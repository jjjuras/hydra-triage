"""Synthetic API-usage logs with labeled actors.

The abuse archetypes are built from the signals Anthropic published in its February and September 2026
reports: synchronized traffic across many accounts, identical prompt structures, shared payment methods,
coordinated timing, volume concentrated in narrow areas, chain-of-thought elicitation, traffic that
redirects within a day of a model release, 'hydra' networks that mix extraction traffic with unrelated
requests, and stolen keys rotated through a proxy so they blend with the owner's usage.

Nothing here is real data. The generator exists so the detector can be measured against ground truth.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

ACTOR_TYPES = ("legit_dev", "legit_enterprise", "legit_research", "distillation_ring", "reseller_proxy", "account_farm", "stolen_key")
ABUSE_TYPES = ("distillation_ring", "reseller_proxy", "account_farm", "stolen_key")
TOPICS = 40  # coarse topic ids for prompts
RELEASE_DAY = 20  # a new model ships on this day of the window


@dataclass
class SynthConfig:
    days: int = 30
    n_legit: int = 3000
    n_rings: int = 3  # distillation rings (hydra clusters)
    ring_size: int = 120  # accounts per ring
    n_resellers: int = 2
    reseller_keys: int = 25
    n_farms: int = 2
    farm_size: int = 200
    n_stolen: int = 40
    seed: int = 1
    adapted: int = 0  # 0 base; 1 = desync, 10x payment methods, dilute templates; 2 = +no shared identifiers; 3 = +paraphrase every prompt (no shared templates)


def _hours(days: int) -> int:
    return days * 24


def generate(cfg: SynthConfig) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (accounts, requests). requests has one row per hour-bucket per account with counts and traits."""
    rng = np.random.default_rng(cfg.seed)
    H = _hours(cfg.days)
    accounts: list[dict] = []
    rows: list[dict] = []
    aid = 0

    def add_account(kind: str, cluster: str | None, payment: str, device: str, signup_day: int, promo_share: float) -> int:
        nonlocal aid
        aid += 1
        accounts.append({"account_id": aid, "actor": kind, "cluster": cluster, "payment_method": payment, "device_fp": device, "signup_day": signup_day, "promo_share": promo_share})
        return aid

    def emit(a: int, hour: int, n: int, topics: np.ndarray, template: int | None, cot: float, out_ratio: float, ip: str, asn: str) -> None:
        if n <= 0:
            return
        rows.append({"account_id": a, "hour": hour, "requests": int(n), "topic": int(rng.choice(topics)), "template_id": template, "cot_rate": float(cot), "out_in_ratio": float(out_ratio), "ip": ip, "asn": asn})

    # --- legitimate accounts: diurnal, their own topics, their own templates, own payment/device/ip
    for i in range(cfg.n_legit):
        kind = rng.choice(["legit_dev", "legit_enterprise", "legit_research"], p=[0.6, 0.25, 0.15])
        base = {"legit_dev": 6, "legit_enterprise": 80, "legit_research": 25}[kind] * rng.lognormal(0, 0.6)
        tz = rng.integers(-8, 9)
        topics = rng.choice(TOPICS, size=rng.integers(2, 7), replace=False)
        template = int(rng.integers(10_000, 99_999)) if rng.random() < 0.5 else None
        pay, dev = f"pm_{aid + 1}", f"dev_{aid + 1}"
        a = add_account(kind, None, pay, dev, int(rng.integers(-400, cfg.days)), 0.0)
        ips = [f"ip_{a}_{k}" for k in range(rng.integers(1, 4))]
        asn = f"asn_{rng.integers(1, 60)}"
        cot = 0.03 if kind != "legit_research" else 0.15
        for h in range(H):
            local = (h + tz) % 24
            diurnal = 1.0 if 8 <= local <= 19 else 0.15
            n = rng.poisson(base * diurnal / 24)
            emit(a, h, n, topics, template if rng.random() < 0.4 else None, cot, rng.normal(1.2, 0.3), rng.choice(ips), asn)

    # --- distillation rings: many accounts, shared payment in sub-groups, synchronized bursts,
    #     narrow topics, one template family, high CoT elicitation, redirect to the new model on release day
    for r in range(cfg.n_rings):
        cluster = f"ring_{r}"
        topics = rng.choice(TOPICS, size=2, replace=False)
        family = int(rng.integers(1000, 1999))
        n_pay = max(2, cfg.ring_size // (40 if not cfg.adapted else 4))  # adapted: far more payment methods
        if cfg.adapted >= 2:
            n_pay = cfg.ring_size  # one card per account: nothing hard to link on
        pays = [f"pm_ring{r}_{k}" for k in range(n_pay)]
        asns = [f"asn_proxy_{r}_{k}" for k in range(3)]
        sched = rng.random(H) < 0.35  # the ring's active hours (shared by members → synchrony)
        for i in range(cfg.ring_size):
            pay = pays[i] if cfg.adapted >= 2 else rng.choice(pays)
            dev = f"dev_ring{r}_{i}" if cfg.adapted >= 2 else f"dev_ring{r}_{rng.integers(0, 15)}"
            a = add_account("distillation_ring", cluster, pay, dev, int(rng.integers(cfg.days - 25, cfg.days)), 0.3)
            ips = [f"ip_ring{r}_{i}_{k}" for k in range(3)] if cfg.adapted >= 2 else [f"ip_ring{r}_{k}" for k in rng.integers(0, 50, size=3)]
            jitter = rng.integers(0, 6) if cfg.adapted else 0
            for h in range(H):
                active = sched[(h + jitter) % H]
                if cfg.adapted and rng.random() < 0.5:
                    active = rng.random() < 0.35  # desynchronize half the time
                if not active:
                    continue
                n = rng.poisson(90 if not cfg.adapted else 45)
                if cfg.adapted >= 3:
                    template = None  # every prompt paraphrased: no structure to fingerprint
                else:
                    template = family + int(rng.integers(0, 3)) if (not cfg.adapted or rng.random() < 0.5) else int(rng.integers(10_000, 99_999))
                emit(a, h, n, topics, template, 0.7 if not cfg.adapted else 0.4, rng.normal(3.5, 0.5), rng.choice(ips), rng.choice(asns))
            # hydra cover traffic: mix unrelated requests
            for h in rng.integers(0, H, size=40):
                emit(a, int(h), rng.poisson(5), rng.choice(TOPICS, size=5), None, 0.03, rng.normal(1.2, 0.3), rng.choice(ips), rng.choice(asns))

    # --- reseller proxies: few keys, enormous mixed volume, many end-user IPs/ASNs, flat around the clock
    for r in range(cfg.n_resellers):
        cluster = f"reseller_{r}"
        pay = f"pm_resell_{r}"
        for k in range(cfg.reseller_keys):
            a = add_account("reseller_proxy", cluster, pay if rng.random() < 0.7 else f"pm_resell_{r}_{k}", f"dev_resell_{r}_{k % 5}", int(rng.integers(-60, cfg.days)), 0.0)
            for h in range(H):
                n = rng.poisson(400)
                emit(a, h, n, rng.choice(TOPICS, size=12), None, 0.08, rng.normal(1.4, 0.4), f"ip_user_{rng.integers(0, 20_000)}", f"asn_{rng.integers(1, 400)}")

    # --- account farms: hundreds of new accounts on shared devices, promo credits, tiny card-testing bursts, then banked
    for f in range(cfg.n_farms):
        cluster = f"farm_{f}"
        for i in range(cfg.farm_size):
            a = add_account("account_farm", cluster, f"pm_farm{f}_{rng.integers(0, 30)}", f"dev_farm{f}_{rng.integers(0, 8)}", int(rng.integers(cfg.days - 10, cfg.days)), 0.95)
            h0 = int(rng.integers(max(0, (cfg.days - 10) * 24), H))
            for h in range(h0, min(H, h0 + rng.integers(1, 4))):
                emit(a, h, rng.poisson(12), rng.choice(TOPICS, size=1), 7777, 0.0, rng.normal(0.6, 0.2), f"ip_farm{f}_{rng.integers(0, 6)}", f"asn_resi_{rng.integers(0, 5)}")

    # --- stolen keys: a legit profile that abruptly changes mid-window (new ASN, new topics, 10x volume)
    legit_ids = [x["account_id"] for x in accounts if x["actor"].startswith("legit")]
    for a in rng.choice(legit_ids, size=cfg.n_stolen, replace=False):
        acc = accounts[a - 1]
        acc["actor"] = "stolen_key"
        acc["cluster"] = "stolen"
        h0 = int(rng.integers(H // 3, H - 48))
        topics = rng.choice(TOPICS, size=2, replace=False)
        for h in range(h0, h0 + 48):
            emit(int(a), h, rng.poisson(60), topics, int(rng.integers(3000, 3010)), 0.5, rng.normal(3.0, 0.5), f"ip_thief_{rng.integers(0, 3)}", "asn_thief")

    acc_df = pd.DataFrame(accounts)
    req_df = pd.DataFrame(rows)
    return acc_df, req_df
