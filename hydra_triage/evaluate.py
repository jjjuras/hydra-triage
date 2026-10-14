"""Score the queue against ground truth. Precision first: a false positive here is a paying customer."""

from __future__ import annotations

import pandas as pd

from .synth import ABUSE_TYPES


def evaluate(cases: pd.DataFrame, feats: pd.DataFrame, comp: pd.Series, ks: tuple[int, ...] = (5, 10, 25, 50)) -> dict:
    f = feats.join(comp)
    f["abuse"] = f["actor"].isin(ABUSE_TYPES)
    comp_truth = f.groupby("component").agg(abuse_accounts=("abuse", "sum"), truth_accounts=("abuse", "size"), abuse_volume=("total_requests", lambda s: int(s[f.loc[s.index, "abuse"]].sum())), actors=("actor", lambda s: sorted(set(s))))
    comp_truth["is_abuse_case"] = comp_truth["abuse_accounts"] / comp_truth["truth_accounts"] >= 0.5
    total_abuse_volume = int(f.loc[f["abuse"], "total_requests"].sum())
    total_abuse_accounts = int(f["abuse"].sum())
    out: dict = {"total_abuse_accounts": total_abuse_accounts, "total_abuse_volume": total_abuse_volume, "legit_accounts": int((~f["abuse"]).sum())}
    ranked = cases.merge(comp_truth, left_on="component", right_index=True, how="left")
    for k in ks:
        top = ranked.head(k)
        out[f"precision@{k}"] = float(top["is_abuse_case"].mean()) if len(top) else None
        out[f"abuse_accounts_captured@{k}"] = int(top["abuse_accounts"].sum())
        out[f"abuse_volume_share@{k}"] = float(top["abuse_volume"].sum() / total_abuse_volume) if total_abuse_volume else None
        # legit accounts swept into the top-k cases: the cost that matters
        out[f"legit_accounts_in_queue@{k}"] = int((top["truth_accounts"] - top["abuse_accounts"]).sum())
    # recall by actor type at k=25
    top = ranked.head(25)
    flagged = set(top["component"])
    f["flagged"] = f["component"].isin(flagged)
    out["recall_by_actor@25"] = {a: float(f.loc[f["actor"] == a, "flagged"].mean()) for a in ABUSE_TYPES if (f["actor"] == a).any()}
    out["legit_flag_rate@25"] = float(f.loc[~f["abuse"], "flagged"].mean())
    return out


def fmt(results: dict) -> str:
    lines = []
    for k, v in results.items():
        if isinstance(v, dict):
            lines.append(f"{k}: " + ", ".join(f"{a}={b:.2f}" for a, b in v.items()))
        elif isinstance(v, float):
            lines.append(f"{k}: {v:.3f}")
        else:
            lines.append(f"{k}: {v}")
    return "\n".join(lines)
