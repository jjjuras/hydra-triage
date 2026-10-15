"""Link accounts into components, then measure how synchronized each component is.

A hydra cluster is a network, not an account. Banning one head grows another; the unit of action is the body.

Two kinds of edge. *Identifier* edges join accounts that share a payment method, a device fingerprint,
a prompt structure they both use habitually (a text fingerprint, see fingerprint.py), or an IP on
several distinct hours. *Behavior* edges (`behavior_edges`) join accounts that share no identifier at
all but move alike on two independent axes at once: the same hours AND the same topics, or the same
switch to a new model AND the same topics. One axis is never enough: strangers share topics, offices
share hours, and a whole company can switch models on release morning. Behavior edges are drawn only
between accounts that already carry rule evidence, because resemblance between two blameless accounts
is not evidence of anything. Single-axis edges exist only as ablations, to measure that cost.
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np
import pandas as pd

from .synth import NEW_MODEL, RELEASE_DAY


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


IDENTIFIER_LINKS = ("payment_method", "device_fp", "template_fingerprint", "shared_ip")
BEHAVIOR_LINKS = ("activity+topic", "release+topic")  # the edges the pipeline uses
ABLATION_LINKS = ("activity", "topic", "release")  # single axes, measured only to show what they sweep in
LINK_TYPES = IDENTIFIER_LINKS + BEHAVIOR_LINKS + ABLATION_LINKS


def _components(accounts: pd.DataFrame, uf: UnionFind, links: dict[int, set[str]]) -> pd.Series:
    comp = pd.Series({int(a): uf.find(int(a)) for a in accounts["account_id"]}, name="component")
    final: dict[int, set[str]] = defaultdict(set)
    for root, kinds in links.items():  # unions can re-root a component; fold link types onto the final root
        final[uf.find(root)] |= kinds
    comp.attrs["links"] = {k: sorted(v) for k, v in final.items()}
    return comp


def _from_components(comp: pd.Series) -> tuple[UnionFind, dict[int, set[str]]]:
    uf = UnionFind()
    for a, c in comp.items():
        uf.union(int(c), int(a))
    links: dict[int, set[str]] = defaultdict(set)
    for c, kinds in comp.attrs.get("links", {}).items():
        links[uf.find(int(c))] |= set(kinds)
    return uf, links


def template_links(
    families: pd.DataFrame,
    totals: pd.Series,
    *,
    min_requests: int = 50,
    min_share: float = 0.10,
    max_popular_share: float = 0.10,
) -> list[tuple[int, int, int]]:
    """Pairs (a, b, family) of accounts that both use the same prompt structure *as a habit*.

    `families` is the sampled table from fingerprint.assign_families (account_id, requests, family).
    A structure counts as a habit for an account when it carries at least `min_share` of the account's
    sampled volume AND, extrapolated to the account's total volume, at least `min_requests` requests.
    An attacker who dilutes with random structures touches thousands of them incidentally; incidental
    overlap must never link strangers (the first version of this rule merged ~680 legitimate accounts
    into a ring). A structure that is a habit for more than `max_popular_share` of all accounts is a
    popular prompt, not an operator's signature, and links nobody."""
    per = families.groupby(["family", "account_id"])["requests"].sum()
    sampled = families.groupby("account_id")["requests"].sum()
    share = per / sampled.reindex(per.index.get_level_values(1)).to_numpy()
    est = share * totals.reindex(per.index.get_level_values(1)).to_numpy()
    habit = per[(share >= min_share) & (est >= min_requests)]
    n_accounts = int(totals.shape[0])
    out: list[tuple[int, int, int]] = []
    for fam, grp in habit.groupby(level=0):
        ids = [int(a) for (_, a) in grp.index]
        if len(ids) < 2 or len(ids) > max_popular_share * n_accounts:
            continue
        out.extend((ids[0], b, int(fam)) for b in ids[1:])
    return out


def link_accounts(
    accounts: pd.DataFrame,
    requests: pd.DataFrame,
    families: pd.DataFrame | None = None,
    *,
    min_shared_ip_hours: int = 3,
    template_kw: dict | None = None,
) -> pd.Series:
    """Component id per account from identifier edges: shared payment method, shared device fingerprint,
    a prompt structure both accounts use habitually (text fingerprint), and a shared IP on at least
    `min_shared_ip_hours` distinct hours. Link types per component are on `.attrs["links"]`."""
    uf = UnionFind()
    links: dict[int, set[str]] = defaultdict(set)

    def union(a: int, b: int, kind: str) -> None:
        uf.union(a, b)
        links[uf.find(a)].add(kind)

    for a in accounts["account_id"]:
        uf.find(int(a))
    for col in ("payment_method", "device_fp"):
        for _, grp in accounts.groupby(col)["account_id"]:
            ids = grp.tolist()
            for b in ids[1:]:
                union(int(ids[0]), int(b), col)
    if families is not None:
        totals = (
            requests.groupby("account_id")["requests"].sum().reindex(accounts["account_id"], fill_value=0)
        )
        for a, b, _fam in template_links(families, totals, **(template_kw or {})):
            union(a, b, "template_fingerprint")
    ip_hours = requests.groupby(["ip", "account_id"])["hour"].nunique()
    ip_acc: dict[str, list[int]] = defaultdict(list)
    for (ip, a), n in ip_hours.items():
        if n >= min_shared_ip_hours:
            ip_acc[str(ip)].append(int(a))
    for ids in ip_acc.values():
        if 2 <= len(ids) <= 2000:
            for b in ids[1:]:
                union(ids[0], b, "shared_ip")
    return _components(accounts, uf, links)


def apply_links(comp: pd.Series, pairs: list[tuple[int, int]], kind: str) -> pd.Series:
    """A new partition: `comp` plus the given edges, each tagged with `kind`."""
    uf, links = _from_components(comp)
    for a, b in pairs:
        uf.union(int(a), int(b))
        links[uf.find(int(a))].add(kind)
    accounts = pd.DataFrame({"account_id": comp.index})
    return _components(accounts, uf, links)


def hourly_matrix(requests: pd.DataFrame, ids: list[int], days: int) -> np.ndarray:
    H = days * 24
    sub = requests[requests["account_id"].isin(ids)]
    m = sub.pivot_table(index="account_id", columns="hour", values="requests", aggfunc="sum", fill_value=0)
    m = m.reindex(index=ids, columns=range(H), fill_value=0)
    return m.to_numpy(dtype=float)


def _centered_cosine(m: np.ndarray) -> np.ndarray:
    m = m - m.mean(axis=1, keepdims=True)
    norm = np.linalg.norm(m, axis=1, keepdims=True) + 1e-9
    return (m / norm) @ (m / norm).T


def candidates(scored: pd.DataFrame, min_rule_score: float) -> list[int]:
    """Accounts eligible for behavior edges: those already carrying rule evidence."""
    return [int(a) for a in scored.index[scored["rule_score"] >= min_rule_score]]


def activity_similarity(requests: pd.DataFrame, ids: list[int], days: int) -> np.ndarray:
    """Mean-centred cosine between binarized hourly activity (active / not, per hour): the lockstep
    idea of SynchroTrap / CopyCatch over every pair of candidates, not only inside components that
    identifiers already formed."""
    m = (hourly_matrix(requests, ids, days) > 0).astype(float)
    return _centered_cosine(m)


def topic_similarity(requests: pd.DataFrame, ids: list[int]) -> np.ndarray:
    """Cosine between topic-volume vectors. Rings work the same narrow topics; so do the seats of a
    company, and so, often enough, do two strangers: on its own this links the wrong people."""
    sub = requests[requests["account_id"].isin(ids)]
    v = sub.pivot_table(index="account_id", columns="topic", values="requests", aggfunc="sum", fill_value=0)
    v = v.reindex(index=ids, fill_value=0).to_numpy(dtype=float)
    v = v / (np.linalg.norm(v, axis=1, keepdims=True) + 1e-9)
    return v @ v.T


def release_profile(requests: pd.DataFrame, window_hours: int = 24) -> pd.DataFrame:
    """Per account: the share of its traffic that went to the new model during its first `window_hours`
    online after release (NaN when it never came online after release), the volume in that window, and
    the first hour it sent anything to the new model. Measured from the account's own first hour back,
    not the release hour: a ring member asleep at release still redirects the moment it wakes."""
    R = RELEASE_DAY * 24
    post = requests[requests["hour"] >= R]
    first_on = post.groupby("account_id")["hour"].min()
    w = post[post["hour"] < first_on.reindex(post["account_id"]).to_numpy() + window_hours]
    vol = w.groupby("account_id")["requests"].sum()
    new = w["requests"].where(w["model"] == NEW_MODEL, 0).groupby(w["account_id"]).sum()
    first_new = post.loc[post["model"] == NEW_MODEL].groupby("account_id")["hour"].min()
    return pd.DataFrame(
        {"release_redirect": new / vol, "release_window_volume": vol, "switch_hour": first_new}
    )


def release_comovement(
    requests: pd.DataFrame,
    ids: list[int],
    *,
    min_share: float = 0.5,
    min_window_volume: int = 20,
    max_hours_apart: int = 3,
) -> np.ndarray:
    """Boolean (n, n): both accounts had traffic before release, both moved at least `min_share` of
    their first day online after it (at least `min_window_volume` requests, so two requests are not a
    "redirect") to the new model, and first touched it within `max_hours_apart` hours of each other.
    An account born after release did not redirect anything; it only ever knew the new model."""
    R = RELEASE_DAY * 24
    prof = release_profile(requests).reindex(ids)
    pre = requests[requests["hour"] < R].groupby("account_id")["hour"].nunique().reindex(ids, fill_value=0)
    moved = (
        (prof["release_redirect"] >= min_share)
        & (prof["release_window_volume"] >= min_window_volume)
        & prof["switch_hour"].notna()
        & (pre.to_numpy() >= 1)
    )
    h = prof["switch_hour"].to_numpy(dtype=float)
    close = np.abs(h[:, None] - h[None, :]) <= max_hours_apart
    mv = moved.to_numpy()
    return close & mv[:, None] & mv[None, :]


def behavior_edges(
    requests: pd.DataFrame,
    scored: pd.DataFrame,
    *,
    kind: str,
    days: int,
    min_rule_score: float,
    tau_activity: float = 0.5,
    tau_topic: float = 0.8,
    max_hours_apart: int = 3,
    min_active_hours: int = 24,
) -> list[tuple[int, int]]:
    """Pairs of candidate accounts joined by behavior edge `kind` (one of BEHAVIOR_LINKS or, for
    ablation, ABLATION_LINKS). Two-axis kinds require both conditions on the same pair. Co-movement is
    only measurable on accounts with at least `min_active_hours` of activity: a three-hour burst has no
    rhythm to compare, so it carries no edge (unmeasured, not zero)."""
    ids = candidates(scored, min_rule_score)
    ids = [a for a in ids if int(scored.at[a, "hours_active"]) >= min_active_hours]
    if len(ids) < 2:
        return []
    need = kind.split("+")
    ok = np.ones((len(ids), len(ids)), dtype=bool)
    if "activity" in need:
        ok &= activity_similarity(requests, ids, days) >= tau_activity
    if "topic" in need:
        ok &= topic_similarity(requests, ids) >= tau_topic
    if "release" in need:
        ok &= release_comovement(requests, ids, max_hours_apart=max_hours_apart)
    iu = np.triu_indices(len(ids), k=1)
    hit = ok[iu]
    return [(ids[i], ids[j]) for i, j in zip(iu[0][hit], iu[1][hit])]


def component_synchrony(
    requests: pd.DataFrame, comp: pd.Series, days: int, max_members: int = 60
) -> pd.Series:
    """Mean pairwise correlation of members' binarized hourly activity. People sharing a company are
    correlated a little; a ring on one schedule is correlated a lot. NaN for singletons: unmeasured."""
    H = days * 24
    hourly = requests.pivot_table(
        index="account_id", columns="hour", values="requests", aggfunc="sum", fill_value=0
    )
    hourly = hourly.reindex(columns=range(H), fill_value=0)
    out = {}
    for c, members in comp.groupby(comp).groups.items():
        ids = [m for m in members if m in hourly.index][:max_members]
        if len(ids) < 2:
            out[c] = np.nan
            continue
        m = (hourly.loc[ids].to_numpy(dtype=float) > 0).astype(float)
        corr = _centered_cosine(m)
        n = len(ids)
        out[c] = float((corr.sum() - n) / (n * (n - 1)))
    return pd.Series(out, name="synchrony")
