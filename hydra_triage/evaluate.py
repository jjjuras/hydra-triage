"""Score the queue against ground truth. Precision first: a false positive here is a paying customer."""

from __future__ import annotations

import pandas as pd

from .synth import ABUSE_TYPES

KS = (5, 10, 25, 50)


def _truth(cases: pd.DataFrame, feats: pd.DataFrame, comp: pd.Series) -> tuple[pd.DataFrame, pd.DataFrame]:
    f = feats.join(comp)
    f["abuse"] = f["actor"].isin(ABUSE_TYPES)
    f["ring"] = f["actor"] == "distillation_ring"
    comp_truth = f.groupby("component").agg(
        abuse_accounts=("abuse", "sum"),
        ring_accounts=("ring", "sum"),
        truth_accounts=("abuse", "size"),
        abuse_volume=("total_requests", lambda s: int(s[f.loc[s.index, "abuse"]].sum())),
        actors=("actor", lambda s: sorted(set(s))),
    )
    # a case is a hit when at least half its accounts are abusive
    comp_truth["is_abuse_case"] = comp_truth["abuse_accounts"] / comp_truth["truth_accounts"] >= 0.5
    ranked = cases.merge(comp_truth, left_on="component", right_index=True, how="left")
    return f, ranked


def _queue_stats(top: pd.DataFrame, f: pd.DataFrame) -> dict:
    total_abuse_volume = int(f.loc[f["abuse"], "total_requests"].sum())
    n_ring = int(f["ring"].sum())
    return {
        "cases": len(top),
        "precision": float(top["is_abuse_case"].mean()) if len(top) else None,
        "abuse_accounts_captured": int(top["abuse_accounts"].sum()),
        "abuse_volume_share": float(top["abuse_volume"].sum() / total_abuse_volume)
        if total_abuse_volume
        else None,
        # legit accounts swept into the queue: the cost that matters
        "legit_accounts_in_queue": int((top["truth_accounts"] - top["abuse_accounts"]).sum()),
        "ring_recall": float(top["ring_accounts"].sum() / n_ring) if n_ring else None,
    }


def queue_to_ring_recall(ranked: pd.DataFrame, n_ring: int, target: float = 0.9) -> dict:
    """How far down the queue an analyst works to reach `target` of the ring, and what it costs: cases
    worked and legitimate accounts seen on the way. None when the queue never gets there."""
    if not n_ring:
        return {"cases": None, "legit_accounts_seen": None}
    cum_ring = ranked["ring_accounts"].cumsum() / n_ring
    hit = cum_ring[cum_ring >= target]
    if hit.empty:
        return {"cases": None, "legit_accounts_seen": None}
    k = int(hit.index[0]) + 1
    top = ranked.head(k)
    return {"cases": k, "legit_accounts_seen": int((top["truth_accounts"] - top["abuse_accounts"]).sum())}


def evaluate(
    cases: pd.DataFrame,
    feats: pd.DataFrame,
    comp: pd.Series,
    ks: tuple[int, ...] = KS,
    threshold: float | None = None,
) -> dict:
    f, ranked = _truth(cases, feats, comp)
    out: dict = {
        "total_abuse_accounts": int(f["abuse"].sum()),
        "total_abuse_volume": int(f.loc[f["abuse"], "total_requests"].sum()),
        "legit_accounts": int((~f["abuse"]).sum()),
        "cases_total": len(cases),
    }
    for k in ks:
        for name, v in _queue_stats(ranked.head(k), f).items():
            if name != "cases":
                out[f"{name}@{k}"] = v
    # recall by actor type at k=25
    top = ranked.head(25)
    flagged = set(top["component"])
    f["flagged"] = f["component"].isin(flagged)
    out["recall_by_actor@25"] = {
        a: float(f.loc[f["actor"] == a, "flagged"].mean()) for a in ABUSE_TYPES if (f["actor"] == a).any()
    }
    out["legit_flag_rate@25"] = float(f.loc[~f["abuse"], "flagged"].mean())
    ring90 = queue_to_ring_recall(ranked, int(f["ring"].sum()), 0.9)
    out["cases_to_ring_recall_90"] = ring90["cases"]
    out["legit_seen_to_ring_recall_90"] = ring90["legit_accounts_seen"]
    if threshold is not None:
        # stopping rule: work every case at or above the threshold, stop there
        out["threshold"] = threshold
        out["at_threshold"] = _queue_stats(ranked[ranked["case_score"] >= threshold], f)
    return out


def fmt(results: dict) -> str:
    lines = []
    for k, v in results.items():
        if isinstance(v, dict):
            lines.append(
                f"{k}: "
                + ", ".join(f"{a}={b:.2f}" if isinstance(b, float) else f"{a}={b}" for a, b in v.items())
            )
        elif isinstance(v, float):
            lines.append(f"{k}: {v:.3f}")
        else:
            lines.append(f"{k}: {v}")
    return "\n".join(lines)
