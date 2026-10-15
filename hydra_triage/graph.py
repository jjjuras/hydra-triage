"""Link accounts that share hard identifiers, then measure how synchronized each component is.

A hydra cluster is a network, not an account. Banning one head grows another; the unit of action is the body.
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np
import pandas as pd


class UnionFind:
    def __init__(self) -> None:
        self.p: dict[int, int] = {}

    def find(self, x: int) -> int:
        self.p.setdefault(x, x)
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]
            x = self.p[x]
        return x

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.p[rb] = ra


LINK_TYPES = ("payment_method", "device_fp", "template_family", "shared_ip")


def link_accounts(accounts: pd.DataFrame, requests: pd.DataFrame, min_shared_ip_hours: int = 3) -> pd.Series:
    """Return component id per account. Links: shared payment method, shared device fingerprint,
    shared template family, and a shared IP on at least `min_shared_ip_hours` distinct hours.
    The link types that joined each component are recorded on the returned Series' `.attrs["links"]`."""
    uf = UnionFind()
    links: dict[int, set[str]] = defaultdict(set)

    def union(a: int, b: int, kind: str) -> None:
        if uf.find(a) != uf.find(b):
            uf.union(a, b)
        links[uf.find(a)].add(kind)

    for a in accounts["account_id"]:
        uf.find(int(a))
    for col in ("payment_method", "device_fp"):
        for _, grp in accounts.groupby(col)["account_id"]:
            ids = grp.tolist()
            for b in ids[1:]:
                union(int(ids[0]), int(b), col)
    # Shared template family links only when the family is a *habit* on both sides: at least 50 requests
    # AND at least 10% of that account's volume. An attacker who dilutes with random templates touches
    # thousands of families incidentally; incidental overlap must not link strangers together.
    totals = requests.groupby("account_id")["requests"].sum()
    tm = requests.dropna(subset=["template_id"]).copy()
    tm["family"] = (tm["template_id"] // 10).astype(int)  # template ids within a family of ten link
    fam_counts = tm.groupby(["family", "account_id"])["requests"].sum()
    for _fam, grp in fam_counts.groupby(level=0):
        ids = [int(a) for (_, a), n in grp.items() if n >= 50 and n >= 0.10 * totals.get(a, 0)]
        if 2 <= len(ids) <= 5000:
            for b in ids[1:]:
                union(ids[0], b, "template_family")
    ip_hours = requests.groupby(["ip", "account_id"])["hour"].nunique()
    ip_acc: dict[str, list[int]] = defaultdict(list)
    for (ip, a), n in ip_hours.items():
        if n >= min_shared_ip_hours and not str(ip).startswith("ip_user_"):
            ip_acc[str(ip)].append(int(a))
    for ids in ip_acc.values():
        if 2 <= len(ids) <= 2000:
            for b in ids[1:]:
                union(ids[0], b, "shared_ip")
    comp = pd.Series({int(a): uf.find(int(a)) for a in accounts["account_id"]}, name="component")
    # unions can re-root a component; fold link types onto the final root
    final: dict[int, set[str]] = defaultdict(set)
    for root, kinds in links.items():
        final[uf.find(root)] |= kinds
    comp.attrs["links"] = {k: sorted(v) for k, v in final.items()}
    return comp


def component_synchrony(
    requests: pd.DataFrame, comp: pd.Series, days: int, max_members: int = 60
) -> pd.Series:
    """Mean pairwise correlation of members' hourly activity. People sharing a company are correlated a little;
    a ring on one schedule is correlated a lot."""
    H = days * 24
    hourly = requests.pivot_table(
        index="account_id", columns="hour", values="requests", aggfunc="sum", fill_value=0
    )
    hourly = hourly.reindex(columns=range(H), fill_value=0)
    out = {}
    for c, members in comp.groupby(comp).groups.items():
        ids = [m for m in members if m in hourly.index][:max_members]
        if len(ids) < 2:
            out[c] = np.nan  # unmeasured: one account has no synchrony to report
            continue
        m = hourly.loc[ids].to_numpy(dtype=float)
        m = (m > 0).astype(float)
        m = m - m.mean(axis=1, keepdims=True)
        norm = np.linalg.norm(m, axis=1, keepdims=True) + 1e-9
        corr = (m / norm) @ (m / norm).T
        n = len(ids)
        out[c] = float((corr.sum() - n) / (n * (n - 1)))
    return pd.Series(out, name="synchrony")
