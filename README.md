# hydra-triage

![CI](https://github.com/jjjuras/hydra-triage/actions/workflows/ci.yml/badge.svg) · MIT

**Question.** Anthropic's February 2026 report described industrial-scale distillation of Claude run through "hydra cluster" networks of fraudulent accounts: synchronized traffic, identical prompt structures, shared payment methods, coordinated timing, volume concentrated in narrow areas, chain-of-thought elicitation, and traffic that redirected within 24 hours of a model release. Its September 2026 threat report added reseller proxies that rotate stolen keys, account factories banking verified accounts, and resellers that silently proxy to a different model. The unit of abuse is a network, not an account: ban one head and another grows. Can a triage pipeline rank *networks* from behavior alone, precision first, and what does it lose as the attacker adapts?

**What this is.** A small, runnable system on labeled synthetic logs:

1. `synth.py` — 30 days of hourly API usage for ~3,800 accounts: three legitimate archetypes plus distillation rings, reseller proxies, account farms, and stolen keys, each built from the published signals above. Four attacker levels: base; desynchronized timing + 10× more payment methods + diluted templates; no shared identifiers at all; and no shared prompt templates either (every prompt paraphrased).
2. `features.py` — per-account features an investigator can read: volume, burstiness, hour-of-day entropy, topic entropy, template reuse, chain-of-thought rate, output/input ratio, distinct IPs and ASNs, release-day shift, account age, promotional-credit share.
3. `graph.py` — link accounts on hard identifiers (payment method, device) and habitual soft ones (a prompt-template family that is ≥10% of *both* accounts' volume; a shared IP on ≥3 distinct hours) into components, then measure each component's **synchrony** (mean pairwise correlation of members' hourly activity).
4. `score.py` — explainable evidence rules with weights, an isolation forest as a second opinion (never a verdict), and a **case per component** ranked by evidence. Every case carries its reasons in words.
5. `evaluate.py` — precision@k over cases, share of abuse volume captured, legitimate accounts swept into the queue, recall by actor type, and the flag rate on legitimate users. Unmeasured values are `null`, never 0.

## Results (seed 1, 3,810 accounts, 850 abusive)

| attacker level | p@5 | p@10 | p@25 | abuse volume in top 10 | legit accounts in top 10 | ring recall @25 cases |
|---|---|---|---|---|---|---|
| base | 1.00 | 0.80 | 0.32 | 100% | 2 | 1.00 |
| desync + more cards + diluted templates | 1.00 | 0.80 | 0.32 | 100% | 2 | 1.00 |
| no shared identifiers | 1.00 | 0.80 | 0.32 | 100% | 3 | 1.00 |
| no identifiers, no shared templates | 1.00 | 1.00 | 1.00 | 78% | 0 | **0.06** |

Reading the last row: precision looks perfect because each single-account case really is abusive, but the rings have shattered into 360 one-account cases. A 25-case queue reaches 6% of the ring. The body is invisible; the queue can only cut heads. That is the research problem: **recovering the network from behavior alone when the attacker has removed every shared identifier.** Candidate signals this code does not yet use: pairwise synchrony across all high-evidence singletons (not only within linked components), topic and output-shape similarity, and co-movement on release day.

A second lesson from the build: linking on "shared prompt template" merged ~680 legitimate accounts into a ring the first time the attacker diluted its templates. The fix was to require the template to be a *habit* on both sides. Incidental overlap must never link strangers.

## Run

```bash
uv venv .venv --python 3.13 && uv pip install --python .venv/bin/python -e ".[dev]"
.venv/bin/python -m pytest -q
.venv/bin/python -m hydra_triage.run --seed 1 --out runs/
```

No API key, no network, no real data. The analyst view is a DuckDB SQL query over the scored table, because that is what an investigator would actually run.

## Prior work this builds on

Model extraction and distillation: Hinton, Vinyals & Dean, *Distilling the Knowledge in a Neural Network* ([arXiv:1503.02531](https://arxiv.org/abs/1503.02531)); Tramèr et al., *Stealing Machine Learning Models via Prediction APIs*, USENIX Security 2016 ([arXiv:1609.02943](https://arxiv.org/abs/1609.02943)); Orekondy, Schiele & Fritz, *Knockoff Nets*, CVPR 2019 ([arXiv:1812.02766](https://arxiv.org/abs/1812.02766)); Krishna et al., *Thieves on Sesame Street! Model Extraction of BERT-based APIs*, ICLR 2020 ([arXiv:1910.12366](https://arxiv.org/abs/1910.12366)); Carlini et al., *Stealing Part of a Production Language Model*, ICML 2024 ([arXiv:2403.06634](https://arxiv.org/abs/2403.06634)). Output watermarking as a countermeasure: Kirchenbauer et al., *A Watermark for Large Language Models*, ICML 2023 ([arXiv:2301.10226](https://arxiv.org/abs/2301.10226)).

Coordinated-account detection: Cao, Yang, Yu & Palow, *Uncovering Large Groups of Active Malicious Accounts in Online Social Networks* (SynchroTrap), CCS 2014 ([doi:10.1145/2660267.2660269](https://doi.org/10.1145/2660267.2660269)); Beutel et al., *CopyCatch: Stopping Group Attacks by Spotting Lockstep Behavior*, WWW 2013 ([doi:10.1145/2488388.2488400](https://doi.org/10.1145/2488388.2488400)); Hooi et al., *FRAUDAR: Bounding Graph Fraud in the Face of Camouflage*, KDD 2016 ([doi:10.1145/2939672.2939747](https://doi.org/10.1145/2939672.2939747)); Liu, Ting & Zhou, *Isolation Forest*, ICDM 2008 ([doi:10.1109/ICDM.2008.17](https://doi.org/10.1109/ICDM.2008.17)). The synchrony measure here is the lockstep idea from SynchroTrap and CopyCatch applied to hourly API activity; the "camouflage" problem FRAUDAR names is exactly the hydra's cover traffic.

Threat reporting that defines the archetypes: Anthropic, *Detecting and preventing distillation attacks* (Feb 23, 2026, [link](https://www.anthropic.com/news/detecting-and-preventing-distillation-attacks)); Anthropic, *Countering misuse of AI: September 2026* ([link](https://www.anthropic.com/threat-intelligence-report-september-2026)); CNBC, Sept 11 2026 ([link](https://www.cnbc.com/2026/09/11/chinese-ai-labs-moonshot-deepseek-alibaba-anthropic.html)); TechCrunch, Feb 23 2026 ([link](https://techcrunch.com/2026/02/23/anthropic-accuses-chinese-ai-labs-of-mining-claude-as-us-debates-ai-chip-exports/)).

## Honest limits

The data is synthetic, so the numbers measure the method against the archetypes I wrote, not against real attackers. The archetypes are drawn from published descriptions, not from logs. The rules' thresholds were set by hand on this data; on real traffic they would be learned and re-tuned as actors adapt. None of this is a claim about any real account.
