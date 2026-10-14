"""End to end: synthesize → features → link → score → evaluate, for the base world and the adapted attacker.

    python -m hydra_triage.run --seed 1 --out runs/
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import duckdb

from .evaluate import evaluate, fmt
from .features import account_features
from .graph import component_synchrony, link_accounts
from .score import account_scores, component_cases
from .synth import SynthConfig, generate


def pipeline(cfg: SynthConfig) -> tuple[dict, pd.DataFrame, pd.DataFrame]:  # noqa: F821
    accounts, requests = generate(cfg)
    feats = account_features(accounts, requests, cfg.days)
    comp = link_accounts(accounts, requests)
    syn = component_synchrony(requests, comp, cfg.days)
    scored = account_scores(feats, seed=cfg.seed)
    cases = component_cases(scored, comp, syn)
    results = evaluate(cases, scored, comp)
    return results, cases, scored.join(comp)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--days", type=int, default=30)
    ap.add_argument("--out", default="runs")
    a = ap.parse_args(argv)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    summary = {}
    for label, adapted in (("base", 0), ("adapted", 1), ("no_identifiers", 2), ("no_identifiers_no_templates", 3)):
        cfg = SynthConfig(seed=a.seed, days=a.days, adapted=adapted)
        results, cases, scored = pipeline(cfg)
        summary[label] = results
        cases.to_json(out / f"cases_{label}.json", orient="records", indent=1)
        scored.reset_index().to_csv(out / f"accounts_{label}.csv", index=False)
        print(f"=== {label} ===")
        print(fmt(results))
        print("top cases:")
        print(cases.head(8)[["component", "accounts", "volume", "case_score", "synchrony", "evidence"]].to_string(index=False, max_colwidth=90))
        # analyst view in SQL: the same table an investigator would query
        con = duckdb.connect()
        con.register("accounts", scored.reset_index())
        q = con.execute(
            "SELECT actor, count(*) AS n, round(avg(rule_score),2) AS rule, round(avg(anomaly),3) AS anomaly, round(avg(cot_rate),2) AS cot "
            "FROM accounts GROUP BY actor ORDER BY rule DESC"
        ).fetchdf()
        print(q.to_string(index=False))
    (out / "summary.json").write_text(json.dumps(summary, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
