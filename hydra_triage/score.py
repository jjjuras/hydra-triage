"""Triage: rank *components* (not accounts) by evidence, precision first.

Two layers. Evidence rules an investigator can read, each with a weight, produce an explainable score.
An isolation forest over the behavioral features flags accounts that look like nothing else; it is an
input to the queue, never a verdict. The output is a case list with the evidence spelled out per case.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

from .features import FEATURE_COLS

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
        lambda f: (f["release_shift"] > 0.3) & (f["total_requests"] > 500),
        1.5,
        "traffic redirected to the new model within 48h of release",
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


def account_scores(feats: pd.DataFrame, seed: int = 0) -> pd.DataFrame:
    f = feats.copy()
    f["evidence"] = [[] for _ in range(len(f))]
    f["rule_score"] = 0.0
    for pred, w, reason in RULES.values():
        hit = pred(f).fillna(False)
        f.loc[hit, "rule_score"] += w
        for idx in f.index[hit]:
            f.at[idx, "evidence"].append(reason)
    X = f[FEATURE_COLS].to_numpy(dtype=float)
    X = np.log1p(np.abs(X)) * np.sign(X)
    iso = IsolationForest(n_estimators=200, contamination="auto", random_state=seed).fit(X)
    f["anomaly"] = -iso.score_samples(X)  # higher = more anomalous
    return f


def component_cases(f: pd.DataFrame, comp: pd.Series, synchrony: pd.Series) -> pd.DataFrame:
    """One case per component. Score = mean rule score + cluster evidence (size, synchrony, shared identifiers)."""
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
                "evidence": reasons + ev,
            }
        )
    cases = pd.DataFrame(rows).sort_values(["case_score", "volume"], ascending=False).reset_index(drop=True)
    return cases
