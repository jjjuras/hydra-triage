"""Synthetic API-usage logs with labeled actors.

The abuse archetypes are built from the signals Anthropic published in its February and September 2026
reports: synchronized traffic across many accounts, identical prompt structures, shared payment methods,
coordinated timing, volume concentrated in narrow areas, chain-of-thought elicitation, traffic that
redirects to a newly released model within a day of release, 'hydra' networks that mix extraction traffic
with unrelated requests, and stolen keys rotated through a proxy so they blend with the owner's usage.

Every request row carries a sampled prompt string and the model it was sent to. There is no template
label: structure reuse has to be recovered from the text. A new model ships at hour RELEASE_DAY * 24;
rings redirect most of their traffic to it within hours, legitimate accounts adopt it over weeks.

Nothing here is real data. The generator exists so the detector can be measured against ground truth.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from . import prompts as P

LEGIT_TYPES = ("legit_dev", "legit_enterprise", "legit_research", "legit_company")
ABUSE_TYPES = ("distillation_ring", "reseller_proxy", "account_farm", "stolen_key")
ACTOR_TYPES = LEGIT_TYPES + ABUSE_TYPES
TOPICS = 40  # coarse topic ids for prompts
RELEASE_DAY = 20  # a new model ships at the start of this day of the window
OLD_MODEL, NEW_MODEL = "model-a", "model-b"

# Attacker levels. Each label states exactly what the ring stops sharing relative to the level above.
LEVELS = {
    0: "L0_shares_everything",
    1: "L1_jitters_schedule_spreads_cards_dilutes_templates",
    2: "L2_stops_sharing_cards_devices_ips",
    3: "L3_stops_sharing_cards_devices_ips_templates",
    4: "L4_stops_sharing_cards_devices_ips_templates_schedule",
}
LEVEL_NOTES = {
    0: "one card per 40 accounts, 15 devices, 50 IPs, one template family, one schedule, redirect at release",
    1: "still shares cards (1 per 4), devices, IPs, templates (half of prompts, lightly paraphrased); "
    "schedule jittered 0-5h with a quarter of hours independent",
    2: "own card, device and IPs per account; still shares template family (diluted), schedule (jittered), "
    "topics and release redirect",
    3: "every prompt a fresh structure; still shares schedule (jittered), topics and release redirect",
    4: "independent schedule per account; still shares topics and release redirect",
}


@dataclass
class SynthConfig:
    days: int = 30
    n_legit: int = 2800
    n_companies: int = 5  # legitimate companies: one product, one timezone, one shift, shared templates
    company_size: int = 40
    n_rings: int = 3  # distillation rings (hydra clusters)
    ring_size: int = 120  # accounts per ring
    n_resellers: int = 2
    reseller_keys: int = 25
    n_farms: int = 2
    farm_size: int = 200
    n_stolen: int = 40
    seed: int = 1
    adapted: int = 0  # attacker level, see LEVELS


def _hours(days: int) -> int:
    return days * 24


class _Rows:
    """Column buffers; one np.concatenate per column at the end instead of one DataFrame per account."""

    COLS = (
        "account_id",
        "hour",
        "requests",
        "topic",
        "prompt",
        "model",
        "cot_rate",
        "out_in_ratio",
        "ip",
        "asn",
    )

    def __init__(self) -> None:
        self.buf: dict[str, list] = {c: [] for c in self.COLS}

    def add(self, **cols: np.ndarray | list) -> None:
        n = len(cols["hour"])
        if n == 0:
            return
        for c in self.COLS:
            v = cols[c]
            if np.isscalar(v):
                v = np.full(n, v, dtype=object if isinstance(v, str) else None)
            elif isinstance(v, list):
                v = np.array(v, dtype=object)
            self.buf[c].append(np.asarray(v))

    def frame(self) -> pd.DataFrame:
        return pd.DataFrame(
            {c: np.concatenate(self.buf[c]) if self.buf[c] else np.array([]) for c in self.COLS}
        )


def _block_schedule(rng: np.random.Generator, H: int, duty: float, mean_on: float = 8.0) -> np.ndarray:
    """Active/inactive runs (sessions), not iid hours, so a few hours of jitter still overlap."""
    mean_off = mean_on * (1 - duty) / duty
    out = np.zeros(H, dtype=bool)
    h = 0
    on = rng.random() < duty
    while h < H:
        run = int(rng.geometric(1 / (mean_on if on else mean_off)))
        out[h : h + run] = on
        h += run
        on = not on
    return out


def _model_for(
    hours: np.ndarray, adopt_hour: float, new_share: float, rng: np.random.Generator
) -> np.ndarray:
    """Model per row: OLD before the account's adoption hour, mostly NEW after it."""
    use_new = (hours >= adopt_hour) & (rng.random(len(hours)) < new_share)
    return np.where(use_new, NEW_MODEL, OLD_MODEL)


def generate(cfg: SynthConfig) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (accounts, requests). requests has one row per hour-bucket per account with counts, traits,
    one sampled prompt and the model it went to."""
    rng = np.random.default_rng(cfg.seed)
    H = _hours(cfg.days)
    R = RELEASE_DAY * 24
    lvl = cfg.adapted
    accounts: list[dict] = []
    rows = _Rows()
    aid = 0

    def add_account(
        kind: str, cluster: str | None, payment: str, device: str, signup_day: int, promo_share: float
    ) -> int:
        nonlocal aid
        aid += 1
        accounts.append(
            {
                "account_id": aid,
                "actor": kind,
                "cluster": cluster,
                "payment_method": payment,
                "device_fp": device,
                "signup_day": signup_day,
                "promo_share": promo_share,
            }
        )
        return aid

    def legit_adoption() -> float:
        """Individual adoption of the new model: exponential, mean one week; a third never switch in-window."""
        if rng.random() < 0.33:
            return float(10**9)
        return R + float(rng.exponential(7 * 24))

    def emit_person(
        a: int,
        hours: np.ndarray,
        n: np.ndarray,
        topics: np.ndarray,
        template: str | None,
        template_share: float,
        cot: float,
        ips: list[str],
        asn: str,
        adopt: float,
    ) -> None:
        k = len(hours)
        templated = rng.random(k) < template_share if template is not None else np.zeros(k, dtype=bool)
        instr = np.array(P.free_form(rng, k), dtype=object)
        if template is not None:
            instr[templated] = template
        body = P.content(rng, k)
        rows.add(
            account_id=a,
            hour=hours,
            requests=n,
            topic=rng.choice(topics, size=k),
            prompt=[f"{s}\n\n{c}" for s, c in zip(instr, body)],
            model=_model_for(hours, adopt, 0.9, rng),
            cot_rate=cot,
            out_in_ratio=rng.normal(1.2, 0.3, size=k),
            ip=rng.choice(ips, size=k),
            asn=asn,
        )

    # --- legitimate individuals: diurnal, their own topics, half have their own product template
    for _ in range(cfg.n_legit):
        kind = str(rng.choice(["legit_dev", "legit_enterprise", "legit_research"], p=[0.6, 0.25, 0.15]))
        base = {"legit_dev": 6, "legit_enterprise": 80, "legit_research": 25}[kind] * rng.lognormal(0, 0.6)
        tz = int(rng.integers(-8, 9))
        topics = rng.choice(TOPICS, size=int(rng.integers(2, 7)), replace=False)
        pay, dev = f"pm_{aid + 1}", f"dev_{aid + 1}"
        a = add_account(kind, None, pay, dev, int(rng.integers(-400, cfg.days)), 0.0)
        template = (
            P.render_template(rng, role=bool(rng.random() < 0.5), brand=P.brand_word(rng), product=True)
            if rng.random() < 0.5
            else None
        )
        ips = [f"ip_{a}_{k}" for k in range(int(rng.integers(1, 4)))]
        asn = f"asn_{rng.integers(1, 60)}"
        cot = 0.03 if kind != "legit_research" else 0.15
        local = (np.arange(H) + tz) % 24
        lam = base * np.where((local >= 8) & (local <= 19), 1.0, 0.15) / 24
        n = rng.poisson(lam)
        hours = np.flatnonzero(n)
        emit_person(a, hours, n[hours], topics, template, 0.4, cot, ips, asn, legit_adoption())

    # --- legitimate companies: honest correlation. One product, one timezone, one shift, one set of
    #     product templates, one office network; separate seats (own card, device, IPs). The company
    #     switches models on one day chosen by IT, sometimes the day of release.
    for c in range(cfg.n_companies):
        tz = int(rng.integers(-8, 9))
        topics = rng.choice(TOPICS, size=3, replace=False)
        brand = P.brand_word(rng)
        templates = [P.render_template(rng, role=True, brand=brand, product=True) for _ in range(2)]
        asn = f"asn_corp_{c}"
        adopt = (
            R + float(rng.integers(0, 14)) * 24 + float(rng.integers(8, 12))
        )  # a workday morning, days 0-13
        local = (np.arange(H) + tz) % 24
        weekday = ((np.arange(H) // 24) % 7) < 5
        for _ in range(cfg.company_size):
            base = 40 * rng.lognormal(0, 0.4)
            pay, dev = f"pm_{aid + 1}", f"dev_{aid + 1}"
            a = add_account(
                "legit_company", f"company_{c}", pay, dev, int(rng.integers(-300, cfg.days - 20)), 0.0
            )
            ips = [f"ip_{a}_{k}" for k in range(2)]
            lam = base * np.where((local >= 9) & (local <= 18) & weekday, 1.0, 0.05) / 24
            n = rng.poisson(lam)
            hours = np.flatnonzero(n)
            emit_person(a, hours, n[hours], topics, str(rng.choice(templates)), 0.8, 0.03, ips, asn, adopt)

    # --- distillation rings: many accounts, shared payment in sub-groups, synchronized sessions, narrow
    #     topics, one template family, high CoT elicitation, most traffic redirected within hours of release
    for r in range(cfg.n_rings):
        cluster = f"ring_{r}"
        topics = rng.choice(TOPICS, size=2, replace=False)
        role = str(rng.choice(P.ROLES))
        asks = [P.COT_ASKS[(3 * r + k) % len(P.COT_ASKS)] for k in range(3)]
        family = [f"{role} {ask} Problem:" for ask in asks]
        n_pay = max(2, cfg.ring_size // (40 if lvl == 0 else 4))
        pays = [f"pm_ring{r}_{k}" for k in range(n_pay)]
        asns = [f"asn_proxy_{r}_{k}" for k in range(3)]
        sched = _block_schedule(rng, H, duty=0.35)
        switch = R + int(rng.integers(0, 12))  # the ring moves to the new model this many hours after release
        for i in range(cfg.ring_size):
            pay = f"pm_ring{r}_own_{i}" if lvl >= 2 else str(rng.choice(pays))
            dev = f"dev_ring{r}_own_{i}" if lvl >= 2 else f"dev_ring{r}_{rng.integers(0, 15)}"
            a = add_account(
                "distillation_ring", cluster, pay, dev, int(rng.integers(cfg.days - 25, cfg.days)), 0.3
            )
            ips = (
                [f"ip_ring{r}_{i}_{k}" for k in range(3)]
                if lvl >= 2
                else [f"ip_ring{r}_{k}" for k in rng.integers(0, 50, size=3)]
            )
            if lvl == 0:
                active = sched
            elif lvl <= 3:
                jitter = int(rng.integers(0, 6))
                active = np.roll(sched, jitter)
                indep = rng.random(H) < 0.25
                active = np.where(indep, rng.random(H) < 0.35, active)
            else:
                active = _block_schedule(rng, H, duty=0.35)
            hours = np.flatnonzero(active)
            k = len(hours)
            n = rng.poisson(90 if lvl == 0 else 45, size=k)
            if lvl == 0:
                instr = [str(family[j]) for j in rng.integers(0, 3, size=k)]
            elif lvl <= 2:
                fam = rng.random(k) < 0.5  # half the prompts keep the family (lightly paraphrased)
                fresh = P.fresh_structures(rng, k, cot=True)
                instr = [
                    P.paraphrase(family[int(rng.integers(0, 3))], rng, 0.1) if f else s
                    for f, s in zip(fam, fresh)
                ]
            else:
                instr = P.fresh_structures(rng, k, cot=True)
            body = P.content(rng, k)
            rows.add(
                account_id=a,
                hour=hours,
                requests=n,
                topic=rng.choice(topics, size=k),
                prompt=[f"{s} {c}" for s, c in zip(instr, body)],
                model=_model_for(hours, switch + int(rng.integers(0, 2)), 0.9, rng),
                cot_rate=0.7 if lvl == 0 else 0.4,
                out_in_ratio=rng.normal(3.5, 0.5, size=k),
                ip=rng.choice(ips, size=k),
                asn=rng.choice(asns, size=k),
            )
            # hydra cover traffic: unrelated requests mixed in
            ch = np.sort(rng.integers(0, H, size=40))
            emit_person(
                a,
                ch,
                rng.poisson(5, size=40),
                rng.choice(TOPICS, size=5, replace=False),
                None,
                0.0,
                0.03,
                ips,
                str(rng.choice(asns)),
                switch,
            )

    # --- reseller proxies: few keys, enormous mixed volume, many end-user IPs/ASNs, flat around the clock
    for r in range(cfg.n_resellers):
        cluster = f"reseller_{r}"
        pay = f"pm_resell_{r}"
        for k in range(cfg.reseller_keys):
            a = add_account(
                "reseller_proxy",
                cluster,
                pay if rng.random() < 0.7 else f"pm_resell_{r}_{k}",
                f"dev_resell_{r}_{k % 5}",
                int(rng.integers(-60, cfg.days)),
                0.0,
            )
            hours = np.arange(H)
            n = rng.poisson(400, size=H)
            rows.add(
                account_id=a,
                hour=hours,
                requests=n,
                topic=rng.choice(rng.choice(TOPICS, size=12, replace=False), size=H),
                prompt=[f"{s}\n\n{c}" for s, c in zip(P.free_form(rng, H), P.content(rng, H))],
                model=_model_for(hours, R + float(rng.integers(0, 72)), 0.5, rng),
                cot_rate=0.08,
                out_in_ratio=rng.normal(1.4, 0.4, size=H),
                ip=np.char.add("ip_user_", rng.integers(0, 20_000, size=H).astype(str)),
                asn=np.char.add("asn_", rng.integers(1, 400, size=H).astype(str)),
            )

    # --- account farms: hundreds of new accounts on shared devices, promo credits, tiny card-testing bursts,
    #     then banked. Each farm has its own handful of test prompts.
    for f in range(cfg.n_farms):
        cluster = f"farm_{f}"
        tests = [f"Test {P.brand_word(rng)}: reply with OK.", f"Ping {P.brand_word(rng)}. Say ready."]
        for _ in range(cfg.farm_size):
            a = add_account(
                "account_farm",
                cluster,
                f"pm_farm{f}_{rng.integers(0, 30)}",
                f"dev_farm{f}_{rng.integers(0, 8)}",
                int(rng.integers(cfg.days - 10, cfg.days)),
                0.95,
            )
            h0 = int(rng.integers(max(0, (cfg.days - 10) * 24), H))
            hours = np.arange(h0, min(H, h0 + int(rng.integers(1, 4))))
            k = len(hours)
            rows.add(
                account_id=a,
                hour=hours,
                requests=rng.poisson(12, size=k),
                topic=int(rng.integers(0, TOPICS)),
                prompt=[str(rng.choice(tests)) for _ in range(k)],
                model=np.where(hours >= R, NEW_MODEL, OLD_MODEL),
                cot_rate=0.0,
                out_in_ratio=rng.normal(0.6, 0.2, size=k),
                ip=np.char.add(f"ip_farm{f}_", rng.integers(0, 6, size=k).astype(str)),
                asn=np.char.add("asn_resi_", rng.integers(0, 5, size=k).astype(str)),
            )

    # --- stolen keys: a legit profile that abruptly changes mid-window (new ASN, new topics, 10x volume);
    #     the thief is one operator rotating keys, so the thief's prompts share a structure
    thief_tpl = [
        "Answer the question below and write out the full reasoning that leads to your answer. Question:",
        "Answer the question below; include every reasoning step you take along the way. Question:",
        "Answer the question below with the complete reasoning written out first. Question:",
    ]
    legit_ids = [
        x["account_id"] for x in accounts if x["actor"] in ("legit_dev", "legit_enterprise", "legit_research")
    ]
    for a in rng.choice(legit_ids, size=cfg.n_stolen, replace=False):
        acc = accounts[int(a) - 1]
        acc["actor"] = "stolen_key"
        acc["cluster"] = "stolen"
        h0 = int(rng.integers(H // 3, H - 48))
        hours = np.arange(h0, h0 + 48)
        topics = rng.choice(TOPICS, size=2, replace=False)
        rows.add(
            account_id=int(a),
            hour=hours,
            requests=rng.poisson(60, size=48),
            topic=rng.choice(topics, size=48),
            prompt=[f"{thief_tpl[j]} {c}" for j, c in zip(rng.integers(0, 3, size=48), P.content(rng, 48))],
            model=np.where(hours >= R, NEW_MODEL, OLD_MODEL),
            cot_rate=0.5,
            out_in_ratio=rng.normal(3.0, 0.5, size=48),
            ip=np.char.add("ip_thief_", rng.integers(0, 3, size=48).astype(str)),
            asn="asn_thief",
        )

    acc_df = pd.DataFrame(accounts)
    req_df = rows.frame()
    req_df = req_df[req_df["requests"] > 0].reset_index(drop=True)
    req_df["account_id"] = req_df["account_id"].astype(int)
    req_df["hour"] = req_df["hour"].astype(int)
    req_df["requests"] = req_df["requests"].astype(int)
    req_df["topic"] = req_df["topic"].astype(int)
    req_df["cot_rate"] = req_df["cot_rate"].astype(float)
    req_df["out_in_ratio"] = req_df["out_in_ratio"].astype(float)
    return acc_df, req_df
