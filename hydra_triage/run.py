"""End to end: synthesize → fingerprint → features → link (identifiers, then behavior) → score → evaluate,
for every attacker level, over several seeds, with ablations and a tau sweep.

python -m hydra_triage.run --seed 1 --out runs/              # one seed, every level, sweep included
python -m hydra_triage.run --seeds 1-5 --out runs/           # the tables in the README
python -m hydra_triage.run --seed 1 --behavior none          # identifier linking only
python -m hydra_triage.run --seed 1 --behavior activity+topic # one edge kind
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

from .evaluate import evaluate
from .features import account_features
from .fingerprint import assign_families
from .graph import (
    ABLATION_LINKS,
    BEHAVIOR_LINKS,
    apply_links,
    behavior_edges,
    component_synchrony,
    link_accounts,
)
from .score import account_scores, component_cases
from .synth import LEVELS, SynthConfig, generate

BEHAVIORS = BEHAVIOR_LINKS
TAUS = (0.3, 0.4, 0.5, 0.6, 0.7, 0.8)
DEFAULT_TAU = 0.5  # activity cosine
DEFAULT_TAU_TOPIC = 0.8
DEFAULT_MIN_RULE_SCORE = 1.5
DEFAULT_THRESHOLD = 6.0


@dataclass
class World:
    cfg: SynthConfig
    accounts: pd.DataFrame
    requests: pd.DataFrame
    families: pd.DataFrame
    scored: pd.DataFrame  # per-account features + rule scores + anomaly (independent of linking)
    id_comp: pd.Series  # components from identifier edges only


def prepare(cfg: SynthConfig, prompts_per_account: int = 60) -> World:
    """Everything that does not depend on the behavior-linking knobs."""
    accounts, requests = generate(cfg)
    families = assign_families(requests, per_account=prompts_per_account, seed=cfg.seed)
    feats = account_features(accounts, requests, cfg.days, families)
    scored = account_scores(feats, seed=cfg.seed)
    id_comp = link_accounts(accounts, requests, families)
    return World(cfg, accounts, requests, families, scored, id_comp)


def rank(
    w: World,
    *,
    behavior: tuple[str, ...] = BEHAVIORS,
    tau: float = DEFAULT_TAU,
    tau_topic: float = DEFAULT_TAU_TOPIC,
    min_rule_score: float = DEFAULT_MIN_RULE_SCORE,
    use_anomaly: bool = True,
    threshold: float | None = DEFAULT_THRESHOLD,
) -> tuple[dict, pd.DataFrame, pd.DataFrame]:
    """Behavior linking (after identifier linking), synchrony, cases, evaluation."""
    comp = w.id_comp
    for kind in behavior:
        pairs = behavior_edges(
            w.requests,
            w.scored,
            kind=kind,
            days=w.cfg.days,
            min_rule_score=min_rule_score,
            tau_activity=tau,
            tau_topic=tau_topic,
        )
        comp = apply_links(comp, pairs, kind)
    syn = component_synchrony(w.requests, comp, w.cfg.days)
    cases = component_cases(w.scored, comp, syn, use_anomaly=use_anomaly)
    results = evaluate(cases, w.scored, comp, threshold=threshold)
    return results, cases, w.scored.join(comp)


def pipeline(cfg: SynthConfig, **kw) -> tuple[dict, pd.DataFrame, pd.DataFrame]:
    return rank(prepare(cfg), **kw)


# ----------------------------------------------------------------------------- reporting

METRICS = {
    "precision@5": "p@5",
    "precision@10": "p@10",
    "precision@25": "p@25",
    "precision@50": "p@50",
    "legit_accounts_in_queue@10": "legit@10",
    "legit_accounts_in_queue@25": "legit@25",
    "legit_accounts_in_queue@50": "legit@50",
    "ring_recall@25": "ring recall@25",
    "ring_recall@50": "ring recall@50",
    "abuse_volume_share@25": "abuse vol@25",
    "cases_to_ring_recall_90": "cases to 90% ring",
    "legit_seen_to_ring_recall_90": "legit seen to 90% ring",
}
THRESHOLD_METRICS = {
    "cases": "cases ≥T",
    "precision": "p ≥T",
    "legit_accounts_in_queue": "legit ≥T",
    "ring_recall": "ring recall ≥T",
}


def _flat(res: dict) -> dict:
    row = {k: res.get(k) for k in METRICS}
    for k in THRESHOLD_METRICS:
        row[f"T.{k}"] = res.get("at_threshold", {}).get(k)
    return row


def _cell(vals: pd.Series) -> str:
    v = vals.dropna().astype(float)
    if v.empty:
        return "—"
    is_int = bool(np.allclose(v, v.round()))
    mean = f"{v.mean():.0f}" if is_int else f"{v.mean():.2f}"
    if len(v) < 2 or v.min() == v.max():
        return mean
    lo, hi = (f"{v.min():.0f}", f"{v.max():.0f}") if is_int else (f"{v.min():.2f}", f"{v.max():.2f}")
    return f"{mean} [{lo}–{hi}]"


def summarize(rows: pd.DataFrame, by: list[str], cols: dict[str, str]) -> tuple[pd.DataFrame, str]:
    """Mean and [min–max] over seeds per group. Returns (long csv frame, markdown table)."""
    g = rows.groupby(by, sort=False)
    stats = g[list(cols)].agg(["mean", "min", "max"])
    stats.columns = [f"{c}.{s}" for c, s in stats.columns]
    stats["n_seeds"] = g.size()
    md = [
        "| " + " | ".join(by + list(cols.values()) + ["n"]) + " |",
        "|" + "---|" * (len(by) + len(cols) + 1),
    ]
    for key, grp in g:
        key = key if isinstance(key, tuple) else (key,)
        md.append(
            "| " + " | ".join([str(k) for k in key] + [_cell(grp[c]) for c in cols] + [str(len(grp))]) + " |"
        )
    return stats.reset_index(), "\n".join(md)


def _parse_range(s: str) -> list[int]:
    out: list[int] = []
    for part in s.split(","):
        if "-" in part:
            a, b = part.split("-")
            out.extend(range(int(a), int(b) + 1))
        else:
            out.append(int(part))
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seed", type=int, default=None, help="single seed (shorthand for --seeds N)")
    ap.add_argument("--seeds", default="1-5", help="e.g. 1-5 or 1,3,7")
    ap.add_argument("--levels", default="0-4", help="attacker levels to run")
    ap.add_argument("--days", type=int, default=30)
    ap.add_argument("--out", default="runs")
    ap.add_argument(
        "--behavior",
        default=",".join(BEHAVIORS),
        help="comma list from activity+topic,release+topic (or single-axis activity,topic,release) or 'none'",
    )
    ap.add_argument(
        "--tau", type=float, default=DEFAULT_TAU, help="activity-cosine threshold for behavior edges"
    )
    ap.add_argument(
        "--tau-topic", type=float, default=DEFAULT_TAU_TOPIC, help="topic-cosine threshold for behavior edges"
    )
    ap.add_argument(
        "--min-rule-score",
        type=float,
        default=DEFAULT_MIN_RULE_SCORE,
        help="accounts below this carry no behavior edges",
    )
    ap.add_argument(
        "--threshold",
        type=float,
        default=DEFAULT_THRESHOLD,
        help="stopping rule: work cases with case_score ≥ T",
    )
    ap.add_argument(
        "--no-anomaly", action="store_true", help="drop the isolation-forest tie-break from the main ranking"
    )
    ap.add_argument("--no-sweep", action="store_true", help="skip the tau sweep and edge-type ablation")
    ap.add_argument("--sweep-levels", default="3,4")
    ap.add_argument("--prompts-per-account", type=int, default=60)
    a = ap.parse_args(argv)
    seeds = [a.seed] if a.seed is not None else _parse_range(a.seeds)
    levels = _parse_range(a.levels)
    behavior = () if a.behavior == "none" else tuple(a.behavior.split(","))
    sweep_levels = [] if a.no_sweep else [lv for lv in _parse_range(a.sweep_levels) if lv in levels]
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    summary: dict = {"args": vars(a), "levels": LEVELS}
    main_rows, abl_rows, sweep_rows = [], [], []
    for seed in seeds:
        for lvl in levels:
            label = LEVELS[lvl]
            cfg = SynthConfig(seed=seed, days=a.days, adapted=lvl)
            w = prepare(cfg, a.prompts_per_account)
            variants = {
                "identifiers only": {"behavior": (), "use_anomaly": not a.no_anomaly},
                "identifiers + behavior": {"behavior": behavior, "use_anomaly": not a.no_anomaly},
                "identifiers + behavior, no anomaly tie-break": {
                    "behavior": behavior,
                    "use_anomaly": a.no_anomaly,
                },
            }
            for name, kw in variants.items():
                res, cases, scored = rank(
                    w,
                    tau=a.tau,
                    tau_topic=a.tau_topic,
                    min_rule_score=a.min_rule_score,
                    threshold=a.threshold,
                    **kw,
                )
                summary.setdefault(name, {}).setdefault(label, {})[str(seed)] = res
                abl_rows.append({"level": label, "variant": name, "seed": seed, **_flat(res)})
                if name == "identifiers + behavior":
                    main_rows.append({"level": label, "seed": seed, **_flat(res)})
                    cases.to_json(out / f"cases_{label}_seed{seed}.json", orient="records", indent=1)
                    scored.reset_index().drop(columns=["evidence"]).to_csv(
                        out / f"accounts_{label}_seed{seed}.csv", index=False
                    )
                    r = res
                    print(
                        f"seed {seed} {label}: p@10={r['precision@10']:.2f} ring_recall@25={r['ring_recall@25']:.2f} cases_to_90%_ring={r['cases_to_ring_recall_90']} legit_seen={r['legit_seen_to_ring_recall_90']} | T≥{a.threshold}: {r['at_threshold']}"
                    )
                    print("  top cases:")
                    print(
                        cases.head(6)[
                            ["accounts", "volume", "case_score", "synchrony", "link_types", "evidence"]
                        ].to_string(index=False, max_colwidth=70)
                    )
            if seed == seeds[0]:
                # analyst view in SQL: the same table an investigator would query
                con = duckdb.connect()
                con.register("accounts", scored.reset_index().drop(columns=["evidence"]))
                q = con.execute(
                    "SELECT actor, count(*) AS n, round(avg(rule_score),2) AS rule, round(avg(anomaly),3) AS anomaly, "
                    "round(avg(cot_rate),2) AS cot, round(avg(template_rate),2) AS tmpl, round(avg(release_redirect),2) AS redirect "
                    "FROM accounts GROUP BY actor ORDER BY rule DESC"
                ).fetchdf()
                print(q.to_string(index=False))
            if lvl in sweep_levels:
                for tau in TAUS:
                    res, _, _ = rank(
                        w,
                        behavior=behavior,
                        tau=tau,
                        tau_topic=a.tau_topic,
                        min_rule_score=a.min_rule_score,
                        threshold=a.threshold,
                        use_anomaly=not a.no_anomaly,
                    )
                    sweep_rows.append({"level": label, "tau": tau, "seed": seed, **_flat(res)})
                for edge in BEHAVIORS + ABLATION_LINKS:
                    res, _, _ = rank(
                        w,
                        behavior=(edge,),
                        tau=a.tau,
                        tau_topic=a.tau_topic,
                        min_rule_score=a.min_rule_score,
                        threshold=a.threshold,
                        use_anomaly=not a.no_anomaly,
                    )
                    abl_rows.append(
                        {"level": label, "variant": f"identifiers + {edge} only", "seed": seed, **_flat(res)}
                    )

    tcols = {f"T.{k}": v for k, v in THRESHOLD_METRICS.items()}
    main_df = pd.DataFrame(main_rows)
    csv, md = summarize(main_df, ["level"], METRICS | tcols)
    csv.to_csv(out / "results.csv", index=False)
    text = [
        f"# Results — {len(seeds)} seed(s) {seeds}, tau={a.tau}, min_rule_score={a.min_rule_score}, T={a.threshold}, behavior={behavior or 'none'}",
        "",
        "Mean [min–max] over seeds.",
        "",
        md,
    ]
    abl_df = pd.DataFrame(abl_rows)
    acsv, amd = summarize(
        abl_df,
        ["level", "variant"],
        {
            k: METRICS[k]
            for k in (
                "precision@10",
                "ring_recall@25",
                "cases_to_ring_recall_90",
                "legit_seen_to_ring_recall_90",
            )
        },
    )
    acsv.to_csv(out / "ablation.csv", index=False)
    text += [
        "",
        "## Ablation: identifier links only vs + behavior links; anomaly tie-break on vs off",
        "",
        amd,
    ]
    if sweep_rows:
        sw_df = pd.DataFrame(sweep_rows)
        scsv, smd = summarize(
            sw_df,
            ["level", "tau"],
            {
                k: METRICS[k]
                for k in (
                    "ring_recall@25",
                    "cases_to_ring_recall_90",
                    "legit_seen_to_ring_recall_90",
                    "precision@10",
                )
            },
        )
        scsv.to_csv(out / "sweep_tau.csv", index=False)
        (out / "sweep_tau.md").write_text(smd + "\n")
        text += [
            "",
            f"## Tau sweep (activity-cosine threshold; topic threshold fixed at {a.tau_topic})",
            "",
            smd,
        ]
    (out / "results.md").write_text("\n".join(text) + "\n")
    (out / "summary.json").write_text(json.dumps(summary, indent=1, default=str))
    print("\n".join(text))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
