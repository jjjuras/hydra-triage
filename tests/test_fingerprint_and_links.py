"""The parts the first review found missing: structure reuse recovered from text (no template label),
habit-only template links, two-axis behavior edges, and the honest-correlation foil (a company)."""

import numpy as np
import pandas as pd

from hydra_triage import prompts as P
from hydra_triage.fingerprint import assign_families, structure_families
from hydra_triage.graph import behavior_edges, candidates, link_accounts, release_profile, template_links
from hydra_triage.run import prepare
from hydra_triage.synth import NEW_MODEL, RELEASE_DAY, SynthConfig

SMALL = SynthConfig(
    days=10,
    n_legit=300,
    n_rings=1,
    ring_size=30,
    n_resellers=1,
    reseller_keys=5,
    n_farms=1,
    farm_size=40,
    n_stolen=5,
    seed=3,
)


def test_same_template_different_content_is_one_family_and_fresh_structures_are_not():
    rng = np.random.default_rng(0)
    tpl = "You are a careful analyst. Summarize the following support ticket as JSON. Be concise."
    same = [f"{tpl} {c}" for c in P.content(rng, 40)]
    fresh = [f"{s} {c}" for s, c in zip(P.fresh_structures(rng, 40, cot=True), P.content(rng, 40))]
    fam = structure_families(same + fresh, min_df=5)
    assert (
        pd.Series(fam[:40]).value_counts().iloc[0] >= 36
    )  # one structure, forty payloads (a few carry filler words)
    assert len(set(fam[40:])) > 20  # forty fresh structures do not collapse into one
    assert not (set(fam[:40]) & set(fam[40:]))


def test_light_paraphrase_stays_in_family():
    rng = np.random.default_rng(1)
    tpl = "You are an expert tutor. Explain the following code snippet in plain English. Show your work."
    base = [f"{tpl} {c}" for c in P.content(rng, 30)]
    para = [f"{P.paraphrase(tpl, rng, 0.1)} {c}" for c in P.content(rng, 30)]
    fam = structure_families(base + para, min_df=5)
    assert pd.Series(fam).value_counts().iloc[0] >= 50  # most paraphrases land with the original


def test_template_links_need_a_habit_on_both_sides_and_ignore_popular_structures():
    fams = pd.DataFrame(
        {
            "account_id": [1, 1, 2, 2, 3, 3, 4],
            "requests": [95, 5, 95, 5, 5, 95, 100],
            "family": [7, 1, 7, 1, 7, 2, 7],
        }
    )
    totals = pd.Series({1: 1000, 2: 1000, 3: 1000, 4: 1000})
    pairs = template_links(fams, totals, min_requests=50, min_share=0.1, max_popular_share=0.9)
    assert sorted((a, b) for a, b, _ in pairs) == [
        (1, 2),
        (1, 4),
    ]  # 3 touches family 7 only incidentally; family 1 is nobody's habit
    # a structure that is a habit for most accounts is a popular prompt, not a signature
    assert template_links(fams, totals, min_requests=50, min_share=0.1, max_popular_share=0.5) == []


def test_release_profile_is_per_account_first_day_online():
    R = RELEASE_DAY * 24
    req = pd.DataFrame(
        {
            "account_id": [1, 1, 1, 2, 2],
            "hour": [R + 30, R + 31, R + 60, R - 5, R + 1],
            "requests": [10, 10, 10, 10, 10],
            "model": [NEW_MODEL, NEW_MODEL, "model-a", "model-a", "model-a"],
        }
    )
    prof = release_profile(req)
    assert prof.loc[1, "release_redirect"] == 1.0  # measured from its own first hour back, not from R
    assert prof.loc[1, "switch_hour"] == R + 30
    assert prof.loc[2, "release_redirect"] == 0.0 and np.isnan(prof.loc[2, "switch_hour"])


def test_two_axis_edges_join_the_ring_and_leave_the_company_alone():
    base = {
        "days": 25,
        "n_legit": 200,
        "n_companies": 2,
        "company_size": 20,
        "n_rings": 1,
        "ring_size": 30,
        "n_resellers": 1,
        "reseller_keys": 3,
        "n_farms": 1,
        "farm_size": 30,
        "n_stolen": 3,
        "seed": 5,
    }
    for lvl in (0, 3):
        w = prepare(SynthConfig(adapted=lvl, **base))
        actor = w.accounts.set_index("account_id")["actor"]
        cands = candidates(w.scored, 1.5)
        assert (actor.loc[cands] == "legit_company").any()  # the honest-correlation foil is in the pool
        for kind in ("activity+topic", "release+topic"):
            pairs = behavior_edges(w.requests, w.scored, kind=kind, days=25, min_rule_score=1.5)
            joined = {(actor[a], actor[b]) for a, b in pairs}
            assert all(x == y for x, y in joined), (
                lvl,
                kind,
                joined,
            )  # never a ring member to a company seat
            if lvl == 0:
                assert sum(actor[a] == "distillation_ring" for a, _ in pairs) > 0, kind


def test_identifier_linking_uses_text_fingerprint_not_labels():
    a, r = SMALL, None
    w = prepare(SMALL)
    comp = link_accounts(w.accounts, w.requests, w.families)
    assert "template_fingerprint" in {k for kinds in comp.attrs["links"].values() for k in kinds}
    assert "template_id" not in w.requests.columns  # there is no oracle column to join on
    fams = assign_families(w.requests, per_account=10, seed=0)
    assert set(fams.columns) == {"account_id", "requests", "family"}
    assert a is SMALL and r is None
