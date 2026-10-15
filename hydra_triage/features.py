"""Per-account behavioral features over the window. Every feature is explainable to an investigator.

Unmeasured is NaN, never 0. An account with no traffic has total_requests 0 (that is a measurement) but
no burstiness, no entropy and no release behavior to report (those are NaN). Scoring treats NaN
explicitly: a rule cannot fire on a value that was not measured.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .graph import release_profile
from .synth import RELEASE_DAY


def _entropy(counts: pd.Series) -> float:
    p = counts / counts.sum()
    return float(-(p * np.log2(p + 1e-12)).sum())


def template_rate(families: pd.DataFrame, min_rows: int = 3, max_popular_share: float = 0.10) -> pd.Series:
    """Share of an account's sampled volume that sits in *uncommon* prompt structures the account itself
    reuses: a structure seen in at least `min_rows` of its sampled hour-buckets and used by no more than
    `max_popular_share` of all accounts. Popular structures ("Summarize the following ticket as JSON")
    are how everyone prompts; they say nothing about an operator. Computed from the text fingerprint,
    not from any template label."""
    per = families.groupby(["account_id", "family"])["requests"].agg(["sum", "size"])
    n_accounts = families["account_id"].nunique()
    users = families.groupby("family")["account_id"].nunique()
    uncommon = users[users <= max_popular_share * n_accounts].index
    keep = (per["size"] >= min_rows) & per.index.get_level_values(1).isin(uncommon)
    reused = per[keep]["sum"].groupby(level=0).sum()
    total = families.groupby("account_id")["requests"].sum()
    return (reused.reindex(total.index, fill_value=0) / total).rename("template_rate")


def account_features(
    accounts: pd.DataFrame, requests: pd.DataFrame, days: int, families: pd.DataFrame | None = None
) -> pd.DataFrame:
    g = requests.groupby("account_id")
    total = g["requests"].sum().rename("total_requests")
    hours_active = g["hour"].nunique().rename("hours_active")
    # burstiness: coefficient of variation of hourly counts over the whole window (zeros included)
    H = days * 24
    hourly = requests.pivot_table(
        index="account_id", columns="hour", values="requests", aggfunc="sum", fill_value=0
    )
    hourly = hourly.reindex(columns=range(H), fill_value=0)
    cv = (hourly.std(axis=1) / (hourly.mean(axis=1) + 1e-9)).rename("burstiness")
    # hour-of-day entropy: flat around the clock (resellers) vs diurnal (people)
    hod = requests.assign(hod=requests["hour"] % 24).groupby(["account_id", "hod"])["requests"].sum()
    hod_entropy = hod.groupby(level=0).apply(_entropy).rename("hod_entropy")
    topic_entropy = (
        requests.groupby(["account_id", "topic"])["requests"]
        .sum()
        .groupby(level=0)
        .apply(_entropy)
        .rename("topic_entropy")
    )
    cot = (
        (requests["cot_rate"] * requests["requests"]).groupby(requests["account_id"]).sum() / total
    ).rename("cot_rate")
    out_ratio = (
        (requests["out_in_ratio"] * requests["requests"]).groupby(requests["account_id"]).sum() / total
    ).rename("out_in_ratio")
    n_ip = g["ip"].nunique().rename("n_ip")
    n_asn = g["asn"].nunique().rename("n_asn")
    # release-day volume shift: 48h after vs 48h before (NaN when the account was silent in both)
    pre = (
        requests[(requests["hour"] >= (RELEASE_DAY - 2) * 24) & (requests["hour"] < RELEASE_DAY * 24)]
        .groupby("account_id")["requests"]
        .sum()
    )
    post = (
        requests[(requests["hour"] >= RELEASE_DAY * 24) & (requests["hour"] < (RELEASE_DAY + 2) * 24)]
        .groupby("account_id")["requests"]
        .sum()
    )
    both = pre.add(post, fill_value=0)
    release_shift = (
        (post.reindex(both.index, fill_value=0) - pre.reindex(both.index, fill_value=0)) / both
    ).rename("release_shift")
    # release redirect: share of the account's first day online after release that went to the new model,
    # and the volume in that day, so a rule can ask for a redirect of real traffic rather than two requests
    prof = release_profile(requests)
    redirect, release_volume = prof["release_redirect"], prof["release_window_volume"]
    cols = [
        total,
        hours_active,
        cv,
        hod_entropy,
        topic_entropy,
        cot,
        out_ratio,
        n_ip,
        n_asn,
        release_shift,
        redirect,
        release_volume,
    ]
    if families is not None:
        cols.append(template_rate(families))
    f = pd.concat(cols, axis=1)
    f = accounts.set_index("account_id").join(f, how="left")
    f["account_age_days"] = days - f["signup_day"]
    # measured zeros: no rows means no requests, no active hours, no IPs, no ASNs, nothing on release day
    for c in ("total_requests", "hours_active", "n_ip", "n_asn", "release_window_volume"):
        f[c] = f[c].fillna(0).astype(int)
    if "template_rate" not in f:
        f["template_rate"] = np.nan
    return f


FEATURE_COLS = [
    "total_requests",
    "hours_active",
    "burstiness",
    "hod_entropy",
    "topic_entropy",
    "template_rate",
    "cot_rate",
    "out_in_ratio",
    "n_ip",
    "n_asn",
    "release_shift",
    "release_redirect",
    "account_age_days",
    "promo_share",
]
