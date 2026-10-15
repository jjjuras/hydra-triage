"""Triage: rank *components* (not accounts) by evidence, precision first.

Two layers. Evidence rules an investigator can read, each with a weight, produce an explainable score.
An isolation forest over the behavioral features gives a second opinion: it never adds to the score,
it only breaks ties between cases whose evidence is otherwise identical (and can be switched off to
measure what that is worth). The output is a case list with the evidence spelled out per case.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

from .features import FEATURE_COLS
from .graph import ABLATION_LINKS, BEHAVIOR_LINKS

RULES = {
    # name: (predicate over account features → bool, weight, human reason)
    "narrow_topics_high_volume": (
        lambda f: (f["topic_entropy"] < 1.6) & (f["total_requests"] > 2000),
        2.0,
        "volume concentrated in narrow topics",
    ),
    "repetitive_templates": (lambda f: f["template_rate"] > 0.6, 1.5, "highly repetitive prompt structures"),
    "cot_elicitation": (lambda f: f["cot_rate"] > 0.35, 2.0, "chain-of-thought elicitation at scale"),
    "long_outputs": (lambda f: f["out_in_ratio"] > 2.5, 1.0, "output-heavy exchanges (training data shape)"),
    "release_redirect": (
        lambda f: (f["release_redirect"] > 0.5) & (f["release_window_volume"] > 100),
        1.5,
        "most traffic redirected to the new model on its first day online after release",
    ),
    "flat_around_clock": (
        lambda f: (f["hod_entropy"] > 4.4) & (f["total_requests"] > 20000),
        1.5,
        "flat around-the-clock volume (proxy/reseller shape)",
    ),
    "many_ips": (lambda f: f["n_ip"] > 500, 1.5, "hundreds of end-user IPs behind one key"),
    "new_account_promo": (
        lambda f: (f["account_age_days"] < 12) & (f["promo_share"] > 0.8),
        1.0,
        "new account living on promotional credit",
    ),
    "card_test_burst": (
        lambda f: (f["hours_active"] <= 3) & (f["total_requests"] > 5) & (f["promo_share"] > 0.8),
        1.0,
        "tiny burst then silence (card test / banked account)",
    ),
}


def _fires(mask: pd.Series) -> pd.Series:
    """A rule fires only on measured values. Comparisons against NaN are already False in pandas; this
    makes the convention explicit and guarantees a boolean series."""
    return mask.fillna(False).astype(bool)


def account_scores(feats: pd.DataFrame, seed: int = 0) -> pd.DataFrame:
    f = feats.copy()
    f["evidence"] = [[] for _ in range(len(f))]
    f["rule_score"] = 0.0
    for pred, w, reason in RULES.values():
        hit = _fires(pred(f))
        f.loc[hit, "rule_score"] += w
        for idx in f.index[hit]:
            f.at[idx, "evidence"].append(reason)
    X = f[FEATURE_COLS].to_numpy(dtype=float)
    # the forest sees unmeasured values as an explicit indicator column plus the column median,
    # so "unmeasured" is itself a feature and never silently becomes zero
    missing = np.isnan(X)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # an all-NaN column is "unmeasured for everyone"
        med = np.nanmedian(X, axis=0)
    med = np.where(np.isnan(med), 0.0, med)
    X = np.where(missing, med, X)
    X = np.log1p(np.abs(X)) * np.sign(X)
    X = np.hstack([X, missing[:, missing.any(axis=0)].astype(float)])
    iso = IsolationForest(n_estimators=200, contamination="auto", random_state=seed).fit(X)
    f["anomaly"] = -iso.score_samples(X)  # higher = more anomalous
    return f


def component_cases(
    f: pd.DataFrame, comp: pd.Series, synchrony: pd.Series, use_anomaly: bool = True
) -> pd.DataFrame:
    """One case per component. case_score = mean rule score + cluster evidence (size, shared identifiers,
    synchrony). Cases are ranked by case_score, then (if `use_anomaly`) by the members' mean isolation-
    forest anomaly, then by volume."""
    link_types = comp.attrs.get("links", {})
    f = f.join(comp)
    rows = []
    for c, g in f.groupby("component"):
        size = len(g)
        syn = synchrony.get(c, np.nan)
        kinds = link_types.get(c, [])
        cluster_bonus = 0.0
        reasons: list[str] = []
        if size >= 5:
            cluster_bonus += 1.0
            reasons.append(f"{size} accounts linked by: " + ", ".join(kinds))
            if all(k in BEHAVIOR_LINKS + ABLATION_LINKS for k in kinds):
                reasons.append("linked by behavior only: no shared identifier")
        if size >= 5 and "payment_method" in kinds:
            cluster_bonus += 1.5
            reasons.append("shared payment methods across accounts")
        if size >= 5 and "device_fp" in kinds:
            cluster_bonus += 1.0
            reasons.append("shared device fingerprints")
        if not np.isnan(syn) and syn > 0.5:
            cluster_bonus += 2.0
            reasons.append(f"synchronized activity (mean pairwise corr {syn:.2f})")
        ev = sorted({r for lst in g["evidence"] for r in lst})
        rows.append(
            {
                "component": c,
                "accounts": size,
                "volume": int(g["total_requests"].sum()),
                "mean_rule_score": float(g["rule_score"].mean()),
                "mean_anomaly": float(g["anomaly"].mean()),
                "synchrony": None if np.isnan(syn) else round(float(syn), 3),
                "case_score": float(g["rule_score"].mean() + cluster_bonus),
                "link_types": list(kinds),
                "evidence": reasons + ev,
            }
        )
    keys = ["case_score", "mean_anomaly", "volume"] if use_anomaly else ["case_score", "volume"]
    cases = pd.DataFrame(rows).sort_values(keys, ascending=False, kind="stable").reset_index(drop=True)
    return cases
