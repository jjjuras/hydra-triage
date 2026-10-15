# hydra-triage

![CI](https://github.com/jjjuras/hydra-triage/actions/workflows/ci.yml/badge.svg) · MIT

**Question.** Anthropic's February 2026 report described industrial-scale distillation of Claude run through "hydra cluster" networks of fraudulent accounts: synchronized traffic, identical prompt structures, shared payment methods, coordinated timing, volume concentrated in narrow areas, chain-of-thought elicitation, and traffic that redirected within 24 hours of a model release. Its September 2026 threat report added reseller proxies that rotate stolen keys, account factories banking verified accounts, and resellers that silently proxy to a different model. The unit of abuse is a network, not an account: ban one head and another grows. Can a triage pipeline rank *networks* from behavior alone, precision first, and what does it lose as the attacker adapts?

**What this is.** A small, runnable system on labeled synthetic logs. Nothing in it joins on a label.

1. `synth.py` — 30 days of hourly API usage for 3,810 accounts: legitimate developers, enterprises, researchers, and five legitimate *companies* (one product, one timezone, one shift, shared prompt templates, a switch to the new model on a morning IT picks: honest correlation, written in so the detector has something to wrongly link), plus three distillation rings of 120, two reseller proxies, two account farms of 200, and 40 stolen keys, each built from the published signals. Every request row carries a prompt *as text* and the model it went to; a new model ships on day 20. Five attacker levels, each of which stops sharing one more thing: L0 shares everything; L1 jitters the schedule, spreads cards, dilutes templates; L2 stops sharing cards, devices and IPs; L3 stops sharing prompt templates (every prompt a fresh structure); L4 stops sharing a schedule. At L4 the ring still shares its topics and still redirects to the new model within hours of release, because that is what it is for.
2. `fingerprint.py` — structure reuse recovered from prompt text, with no template label: tokenize, mask rare and digit-bearing tokens (ids, names, numbers) and lone words between them, collapse runs, 3-shingle MinHash, LSH banding into *families*. Two prompts from one template with different payloads land in one family; a light paraphrase usually does; a fresh structure does not.
3. `features.py` — per-account features an investigator can read: volume, burstiness, hour-of-day entropy, topic entropy, reuse of *uncommon* prompt structures, chain-of-thought rate, output/input ratio, distinct IPs and ASNs, release-day shift, share of the account's first day online after release that went to the new model, account age, promotional-credit share. Unmeasured is `NaN`, never 0.
4. `graph.py` — *identifier* edges: shared payment method, shared device, a prompt structure that is a habit on both sides (≥10% of each account's volume, used by <10% of all accounts), a shared IP on ≥3 distinct hours. *Behavior* edges, for accounts that already carry rule evidence and at least a day of activity: the same hours **and** the same topics (mean-centred cosine of binarized hourly activity ≥0.5 and topic-vector cosine ≥0.8), or the same switch to the new model **and** the same topics (both redirected ≥50% of their first day online after release, both had traffic before release, first touched the new model within 3 hours of each other). One axis is never an edge on its own; single-axis edges exist only as ablations. Each component gets a synchrony score.
5. `score.py` — explainable evidence rules with weights; one case per component ranked by mean rule score plus cluster evidence (size, shared identifiers, synchrony); an isolation forest as a tie-break that can be switched off to measure what it is worth. Every case carries its reasons in words.
6. `evaluate.py` — precision@k, legitimate accounts swept into the queue, ring recall, and the two numbers that matter to an analyst: **cases worked and legitimate accounts seen to reach 90% of the ring**, plus a stopping rule (work every case with score ≥ 6). Unmeasured values are `null`.

## Results (seeds 1–5; 3,810 accounts, 850 abusive in 8 networks plus 40 stolen keys)

Mean [min–max] over five seeds. "Identifiers" is linking on shared cards, devices, IPs and habitual prompt structures; "+ behavior" adds the two-axis edges. The threshold queue works every case scoring ≥ 6 and stops. Full tables: `runs/results.md`, `runs/ablation.csv`, `runs/sweep_tau.csv`, per-seed `runs/summary.json`.

| level | the ring stops sharing | cases to 90% of the ring, identifiers | cases to 90%, + behavior | legit accounts seen on the way | threshold queue: cases · precision · legit · ring recall |
|---|---|---|---|---|---|
| L0 | nothing | 3 | 3 | 0 | 4 [3–6] · 1.00 · 0 [0–1] · 1.00 |
| L1 | schedule jittered, cards spread, templates diluted | 3 | 3 | 0 | 5 [4–6] · 1.00 · 0 · 1.00 |
| L2 | cards, devices, IPs | 4 [3–5] | 4 [3–4] | 0 | 5 [4–5] · 1.00 · 0 · 1.00 |
| L3 | + prompt templates | 288 [268–308] | **5 [1–19]** | 0 [0–1] | 4 [2–8] · 1.00 · 0 [0–1] · 0.96 [0.88–1.00] |
| L4 | + schedule | 263 [159–315] | **57 [19–119]** | 2 [0–9] | 6 [3–9] · 1.00 · 0 [0–1] · 0.77 [0.59–0.86] |

**Reading the table.** Through L2 the identifiers do the work: the whole ring is three or four cases, and behavior edges add nothing. At L3 the ring has shattered into hundreds of one-account cases for the identifier linker (288 cases to reach 90% of it); two-axis behavior edges reassemble it into about five cases with no legitimate account seen on the way. At L4, with no shared schedule, the only axis left is release-day co-movement, and it recovers 77–81% of the ring (the threshold queue; 0.81 [0.64–0.92] at 25 cases) at a cost of two legitimate accounts across five seeds. The remainder is ring members whose first hour on the new model fell more than three hours from any other member's. Precision@10 is 0.66–0.74 at L0–L2 and 0.90–1.00 at L3–L4; the "misses" at low levels are the five legitimate companies, which look like a 40-seat ring on identifiers alone (shared templates, one shift, synchrony ≈ 0.5) and rank sixth to tenth. The threshold queue never contains more than one of their seats.

**What each axis costs on its own** (L3 and L4, five seeds; `runs/ablation.csv`):

| edge | L3 ring recall@25 | L3 legit seen to 90% ring | L4 ring recall@25 | L4 legit seen to 90% ring |
|---|---|---|---|---|
| activity alone | 0.89 [0.75–1.00] | 0 [0–1] | 0.25 [0.10–0.54] | 2 [0–9] |
| topic alone | 1.00, as **one** case | 1 [0–3] | 1.00, as one case | 1 [0–1] |
| release alone | 0.84 [0.69–0.96] | **134 [3–278]** | 0.82 [0.65–0.93] | **53 [3–188]** |
| activity + topic | 0.89 [0.75–1.00] | 0 [0–1] | 0.25 [0.10–0.54] | 2 [0–9] |
| release + topic | 0.84 [0.69–0.95] | 0 [0–1] | 0.81 [0.64–0.92] | 2 [0–9] |
| both two-axis edges | 0.98 [0.92–1.00] | 0 [0–1] | 0.81 [0.64–0.92] | 2 [0–9] |

Topic similarity alone recovers every ring by fusing all three rings, the resellers and the stolen keys into a single 400-plus-account case that no analyst can act on (precision@10 falls to 0.38). Release-day co-movement alone sweeps in whichever company upgraded on release morning: up to 278 legitimate accounts in one seed. Requiring the topic axis as well removes that cost entirely and keeps the recall. The activity axis is clean on its own at its threshold and is simply dead once schedules are independent. The tau sweep (`runs/sweep_tau.md`) shows the same shape: at activity cosine 0.3–0.4 the rings fuse into one case; at 0.5 they are five cases; at 0.6 and above recall drops to 0.84. The isolation-forest tie-break changed no number in any table; it is kept because the ablation that shows it is worth nothing is cheaper than the argument.

## What the build taught me

- **Incidental overlap must never link strangers.** The first template link, "any shared template", merged ~680 legitimate accounts into a ring the moment the attacker diluted its templates. The rule became a habit on both sides.
- **A structure everyone uses is not a signature.** When the text fingerprint replaced the label, every legitimate account suddenly looked 60% "templated", because the small prompt grammar repeats, and a 1,600-account blob swallowed the queue. `template_rate` now counts only structures used by fewer than 10% of accounts.
- **Co-movement needs something to move.** Account-farm accounts live for one to three hours, all on the new model, each on one topic; they co-switched with everyone and chained two rings and the stolen keys into one 560-account case. Behavior edges now require a day of activity and a pre-release history to redirect from.
- **Precision@25 was a ceiling, not a score.** With eight networks in the world it cannot exceed 0.32 however good the ranking is. The metric that answers the analyst's question is how many cases they work, and how many customers they see, to reach the ring.
- **One axis of resemblance is a false-positive machine; two independent axes are not.** This is the finding, and it is the same lesson as SynchroTrap's: lockstep on one dimension is common, lockstep on two is rare.

## Run

```bash
uv venv .venv --python 3.13 && uv pip install --python .venv/bin/python -e ".[dev]"
.venv/bin/python -m pytest -q                                  # 10 tests, no network
.venv/bin/python -m hydra_triage.run --seed 1 --out runs/      # one seed, every level, ablations and sweep (~3 min)
.venv/bin/python -m hydra_triage.run --seeds 1-5 --out runs/   # the tables above (~15 min)
.venv/bin/python -m hydra_triage.run --seed 1 --behavior none  # identifier linking only
```

No API key, no network, no real data. The analyst view is a DuckDB SQL query over the scored table, because that is what an investigator would actually run.

## References

Model extraction and distillation: Hinton, Vinyals & Dean, *Distilling the Knowledge in a Neural Network* ([arXiv:1503.02531](https://arxiv.org/abs/1503.02531)); Tramèr et al., *Stealing Machine Learning Models via Prediction APIs*, USENIX Security 2016 ([arXiv:1609.02943](https://arxiv.org/abs/1609.02943)); Orekondy, Schiele & Fritz, *Knockoff Nets*, CVPR 2019 ([arXiv:1812.02766](https://arxiv.org/abs/1812.02766)); Krishna et al., *Thieves on Sesame Street! Model Extraction of BERT-based APIs*, ICLR 2020 ([arXiv:1910.12366](https://arxiv.org/abs/1910.12366)); Carlini et al., *Stealing Part of a Production Language Model*, ICML 2024 ([arXiv:2403.06634](https://arxiv.org/abs/2403.06634)). Output watermarking as a countermeasure: Kirchenbauer et al., *A Watermark for Large Language Models*, ICML 2023 ([arXiv:2301.10226](https://arxiv.org/abs/2301.10226)).

Coordinated-account detection: Cao, Yang, Yu & Palow, *Uncovering Large Groups of Active Malicious Accounts in Online Social Networks* (SynchroTrap), CCS 2014 ([doi:10.1145/2660267.2660269](https://doi.org/10.1145/2660267.2660269)); Beutel et al., *CopyCatch: Stopping Group Attacks by Spotting Lockstep Behavior*, WWW 2013 ([doi:10.1145/2488388.2488400](https://doi.org/10.1145/2488388.2488400)); Hooi et al., *FRAUDAR: Bounding Graph Fraud in the Face of Camouflage*, KDD 2016 ([doi:10.1145/2939672.2939747](https://doi.org/10.1145/2939672.2939747)); Liu, Ting & Zhou, *Isolation Forest*, ICDM 2008 ([doi:10.1109/ICDM.2008.17](https://doi.org/10.1109/ICDM.2008.17)). The synchrony measure here is the lockstep idea from SynchroTrap and CopyCatch applied to hourly API activity; the "camouflage" problem FRAUDAR names is exactly the hydra's cover traffic.

Threat reporting that defines the archetypes: Anthropic, *Detecting and preventing distillation attacks* (Feb 23, 2026, [link](https://www.anthropic.com/news/detecting-and-preventing-distillation-attacks)); Anthropic, *Countering misuse of AI: September 2026* ([link](https://www.anthropic.com/threat-intelligence-report-september-2026)); CNBC, Sept 11 2026 ([link](https://www.cnbc.com/2026/09/11/chinese-ai-labs-moonshot-deepseek-alibaba-anthropic.html)); TechCrunch, Feb 23 2026 ([link](https://techcrunch.com/2026/02/23/anthropic-accuses-chinese-ai-labs-of-mining-claude-as-us-debates-ai-chip-exports/)).

## Honest limits

The data is synthetic, so the numbers measure the method against the archetypes I wrote, not against real attackers. The archetypes are drawn from published descriptions, not from logs. Thresholds (cosines, shares, the ≥6 stopping rule) were set by hand on seed 1 and then reported over seeds 1–5, so seed 1 is not a held-out test. The L4 remainder is a limit of the 3-hour co-switch window, not a discovery. Union-find linking is transitive, so one wrong edge fuses two networks; the sweep shows that happening below activity cosine 0.5. The text fingerprint still joins a handful of legitimate accounts to a reseller through a popular prompt structure in some seeds; those cases score below the threshold. None of this is a claim about any real account.
