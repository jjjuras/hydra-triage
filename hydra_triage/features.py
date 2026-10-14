"""Per-account behavioral features over the window. Every feature is explainable to an investigator."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .synth import RELEASE_DAY


def _entropy(counts: pd.Series) -> float:
    p = counts / counts.sum()
    return float(-(p * np.log2(p + 1e-12)).sum())


def account_features(accounts: pd.DataFrame, requests: pd.DataFrame, days: int) -> pd.DataFrame:
    g = requests.groupby("account_id")
    total = g["requests"].sum().rename("total_requests")
    hours_active = g["hour"].nunique().rename("hours_active")
    # burstiness: coefficient of variation of hourly counts over the whole window (zeros included)
    H = days * 24
    hourly = requests.pivot_table(index="account_id", columns="hour", values="requests", aggfunc="sum", fill_value=0)
    hourly = hourly.reindex(columns=range(H), fill_value=0)
    cv = (hourly.std(axis=1) / (hourly.mean(axis=1) + 1e-9)).rename("burstiness")
    # hour-of-day entropy: flat around the clock (resellers) vs diurnal (people)
    hod = requests.assign(hod=requests["hour"] % 24).groupby(["account_id", "hod"])["requests"].sum()
    hod_entropy = hod.groupby(level=0).apply(_entropy).rename("hod_entropy")
    topic_entropy = requests.groupby(["account_id", "topic"])["requests"].sum().groupby(level=0).apply(_entropy).rename("topic_entropy")
    tmpl = requests.dropna(subset=["template_id"]).groupby("account_id")["requests"].sum()
    template_rate = (tmpl / total).fillna(0).rename("template_rate")
    cot = (requests["cot_rate"] * requests["requests"]).groupby(requests["account_id"]).sum() / total
    cot = cot.rename("cot_rate")
    out_ratio = (requests["out_in_ratio"] * requests["requests"]).groupby(requests["account_id"]).sum() / total
    out_ratio = out_ratio.rename("out_in_ratio")
    n_ip = g["ip"].nunique().rename("n_ip")
    n_asn = g["asn"].nunique().rename("n_asn")
    # release-day shift: share of volume in the 48h after the release vs the 48h before
    pre = requests[(requests["hour"] >= (RELEASE_DAY - 2) * 24) & (requests["hour"] < RELEASE_DAY * 24)].groupby("account_id")["requests"].sum()
    post = requests[(requests["hour"] >= RELEASE_DAY * 24) & (requests["hour"] < (RELEASE_DAY + 2) * 24)].groupby("account_id")["requests"].sum()
    release_shift = ((post - pre) / (pre + post + 1e-9)).rename("release_shift")
    f = pd.concat([total, hours_active, cv, hod_entropy, topic_entropy, template_rate, cot, out_ratio, n_ip, n_asn, release_shift], axis=1)
    f = accounts.set_index("account_id").join(f, how="left")
    f["account_age_days"] = days - f["signup_day"]
    num = ["total_requests", "hours_active", "burstiness", "hod_entropy", "topic_entropy", "template_rate", "cot_rate", "out_in_ratio", "n_ip", "n_asn", "release_shift"]
    f[num] = f[num].fillna(0)
    return f


FEATURE_COLS = ["total_requests", "hours_active", "burstiness", "hod_entropy", "topic_entropy", "template_rate", "cot_rate", "out_in_ratio", "n_ip", "n_asn", "release_shift", "account_age_days", "promo_share"]
