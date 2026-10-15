# Results — 5 seed(s) [1, 2, 3, 4, 5], tau=0.5, min_rule_score=1.5, T=6.0, behavior=('activity+topic', 'release+topic')

Mean [min–max] over seeds.

| level | p@5 | p@10 | p@25 | p@50 | legit@10 | legit@25 | legit@50 | ring recall@25 | ring recall@50 | abuse vol@25 | cases to 90% ring | legit seen to 90% ring | cases ≥T | p ≥T | legit ≥T | ring recall ≥T | n |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| L0_shares_everything | 1 | 0.66 [0.60–0.80] | 0.26 [0.24–0.32] | 0.13 [0.12–0.16] | 134 [80–160] | 396 [212–623] | 421 [237–648] | 1 | 1 | 1 | 3 | 0 | 4 [3–6] | 1 | 0 [0–1] | 1 | 5 |
| L1_jitters_schedule_spreads_cards_dilutes_templates | 1 | 0.72 [0.60–0.80] | 0.29 [0.24–0.32] | 0.14 [0.12–0.16] | 126 [80–230] | 271 [212–397] | 296 [237–422] | 1 | 1 | 1 | 3 | 0 | 5 [4–6] | 1 | 0 | 1 | 5 |
| L2_stops_sharing_cards_devices_ips | 1 | 0.74 [0.60–0.80] | 0.30 [0.24–0.32] | 0.15 [0.12–0.16] | 114 [90–160] | 257 [223–349] | 282 [248–374] | 1 | 1 | 1 | 4 [3–4] | 0 | 5 [4–5] | 1 | 0 | 1.00 [1.00–1.00] | 5 |
| L3_stops_sharing_cards_devices_ips_templates | 1 | 0.90 [0.50–1.00] | 0.66 [0.20–1.00] | 0.43 [0.10–1.00] | 40 [0–201] | 287 [0–483] | 307 [0–508] | 0.98 [0.92–1.00] | 1.00 [0.99–1.00] | 1.00 [0.98–1.00] | 5 [1–19] | 0 [0–1] | 4 [2–8] | 1 | 0 [0–1] | 0.96 [0.88–1.00] | 5 |
| L4_stops_sharing_cards_devices_ips_templates_schedule | 1 | 1 | 1 | 1 | 2 [0–9] | 2 [0–9] | 2 [0–9] | 0.81 [0.64–0.92] | 0.88 [0.71–0.99] | 0.73 [0.19–0.98] | 57 [19–119] | 2 [0–9] | 6 [3–9] | 1 | 0 [0–1] | 0.77 [0.59–0.86] | 5 |

## Ablation: identifier links only vs + behavior links; anomaly tie-break on vs off

| level | variant | p@10 | ring recall@25 | cases to 90% ring | legit seen to 90% ring | n |
|---|---|---|---|---|---|---|
| L0_shares_everything | identifiers only | 0.66 [0.60–0.80] | 1 | 3 | 0 | 5 |
| L0_shares_everything | identifiers + behavior | 0.66 [0.60–0.80] | 1 | 3 | 0 | 5 |
| L0_shares_everything | identifiers + behavior, no anomaly tie-break | 0.66 [0.60–0.80] | 1 | 3 | 0 | 5 |
| L1_jitters_schedule_spreads_cards_dilutes_templates | identifiers only | 0.72 [0.60–0.80] | 1 | 3 | 0 | 5 |
| L1_jitters_schedule_spreads_cards_dilutes_templates | identifiers + behavior | 0.72 [0.60–0.80] | 1 | 3 | 0 | 5 |
| L1_jitters_schedule_spreads_cards_dilutes_templates | identifiers + behavior, no anomaly tie-break | 0.72 [0.60–0.80] | 1 | 3 | 0 | 5 |
| L2_stops_sharing_cards_devices_ips | identifiers only | 0.88 [0.70–1.00] | 1 | 4 [3–5] | 0 | 5 |
| L2_stops_sharing_cards_devices_ips | identifiers + behavior | 0.74 [0.60–0.80] | 1 | 4 [3–4] | 0 | 5 |
| L2_stops_sharing_cards_devices_ips | identifiers + behavior, no anomaly tie-break | 0.74 [0.60–0.80] | 1 | 4 [3–4] | 0 | 5 |
| L3_stops_sharing_cards_devices_ips_templates | identifiers only | 1 | 0.17 [0.12–0.23] | 288 [268–308] | 0 | 5 |
| L3_stops_sharing_cards_devices_ips_templates | identifiers + behavior | 0.90 [0.50–1.00] | 0.98 [0.92–1.00] | 5 [1–19] | 0 [0–1] | 5 |
| L3_stops_sharing_cards_devices_ips_templates | identifiers + behavior, no anomaly tie-break | 0.90 [0.50–1.00] | 0.98 [0.93–1.00] | 4 [1–16] | 0 [0–1] | 5 |
| L3_stops_sharing_cards_devices_ips_templates | identifiers + activity+topic only | 1 | 0.89 [0.75–1.00] | 31 [4–78] | 0 [0–1] | 5 |
| L3_stops_sharing_cards_devices_ips_templates | identifiers + release+topic only | 1 | 0.84 [0.69–0.95] | 46 [7–97] | 0 [0–1] | 5 |
| L3_stops_sharing_cards_devices_ips_templates | identifiers + activity only | 1 | 0.89 [0.75–1.00] | 31 [4–78] | 0 [0–1] | 5 |
| L3_stops_sharing_cards_devices_ips_templates | identifiers + topic only | 0.38 [0.30–0.50] | 1 | 1 | 1 [0–3] | 5 |
| L3_stops_sharing_cards_devices_ips_templates | identifiers + release only | 1 | 0.84 [0.69–0.96] | 45 [5–96] | 134 [3–278] | 5 |
| L4_stops_sharing_cards_devices_ips_templates_schedule | identifiers only | 1 | 0.25 [0.10–0.54] | 263 [159–315] | 2 [0–9] | 5 |
| L4_stops_sharing_cards_devices_ips_templates_schedule | identifiers + behavior | 1 | 0.81 [0.64–0.92] | 57 [19–119] | 2 [0–9] | 5 |
| L4_stops_sharing_cards_devices_ips_templates_schedule | identifiers + behavior, no anomaly tie-break | 1 | 0.81 [0.64–0.92] | 57 [19–119] | 2 [0–9] | 5 |
| L4_stops_sharing_cards_devices_ips_templates_schedule | identifiers + activity+topic only | 1 | 0.25 [0.10–0.54] | 263 [159–315] | 2 [0–9] | 5 |
| L4_stops_sharing_cards_devices_ips_templates_schedule | identifiers + release+topic only | 1 | 0.81 [0.64–0.92] | 57 [19–119] | 2 [0–9] | 5 |
| L4_stops_sharing_cards_devices_ips_templates_schedule | identifiers + activity only | 1 | 0.25 [0.10–0.54] | 263 [159–315] | 2 [0–9] | 5 |
| L4_stops_sharing_cards_devices_ips_templates_schedule | identifiers + topic only | 0.46 [0.30–0.60] | 1 | 1 | 1 [0–1] | 5 |
| L4_stops_sharing_cards_devices_ips_templates_schedule | identifiers + release only | 1 | 0.82 [0.65–0.93] | 53 [16–115] | 53 [3–188] | 5 |

## Tau sweep (activity-cosine threshold; topic threshold fixed at 0.8)

| level | tau | ring recall@25 | cases to 90% ring | legit seen to 90% ring | p@10 | n |
|---|---|---|---|---|---|---|
| L3_stops_sharing_cards_devices_ips_templates | 0.3 | 1 | 1 | 0 [0–1] | 0.48 [0.40–0.60] | 5 |
| L3_stops_sharing_cards_devices_ips_templates | 0.4 | 1 | 1 | 0 [0–1] | 0.48 [0.40–0.60] | 5 |
| L3_stops_sharing_cards_devices_ips_templates | 0.5 | 0.98 [0.92–1.00] | 5 [1–19] | 0 [0–1] | 0.90 [0.50–1.00] | 5 |
| L3_stops_sharing_cards_devices_ips_templates | 0.6 | 0.84 [0.69–0.95] | 46 [7–97] | 0 [0–1] | 1 | 5 |
| L3_stops_sharing_cards_devices_ips_templates | 0.7 | 0.84 [0.69–0.95] | 46 [7–97] | 0 [0–1] | 1 | 5 |
| L3_stops_sharing_cards_devices_ips_templates | 0.8 | 0.84 [0.69–0.95] | 46 [7–97] | 0 [0–1] | 1 | 5 |
| L4_stops_sharing_cards_devices_ips_templates_schedule | 0.3 | 0.81 [0.64–0.92] | 57 [19–119] | 2 [0–9] | 1 | 5 |
| L4_stops_sharing_cards_devices_ips_templates_schedule | 0.4 | 0.81 [0.64–0.92] | 57 [19–119] | 2 [0–9] | 1 | 5 |
| L4_stops_sharing_cards_devices_ips_templates_schedule | 0.5 | 0.81 [0.64–0.92] | 57 [19–119] | 2 [0–9] | 1 | 5 |
| L4_stops_sharing_cards_devices_ips_templates_schedule | 0.6 | 0.81 [0.64–0.92] | 57 [19–119] | 2 [0–9] | 1 | 5 |
| L4_stops_sharing_cards_devices_ips_templates_schedule | 0.7 | 0.81 [0.64–0.92] | 57 [19–119] | 2 [0–9] | 1 | 5 |
| L4_stops_sharing_cards_devices_ips_templates_schedule | 0.8 | 0.81 [0.64–0.92] | 57 [19–119] | 2 [0–9] | 1 | 5 |
