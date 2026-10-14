import numpy as np

from hydra_triage.evaluate import evaluate
from hydra_triage.features import account_features
from hydra_triage.graph import component_synchrony, link_accounts
from hydra_triage.score import account_scores, component_cases
from hydra_triage.synth import ABUSE_TYPES, SynthConfig, generate

SMALL = SynthConfig(days=10, n_legit=300, n_rings=1, ring_size=30, n_resellers=1, reseller_keys=5, n_farms=1, farm_size=40, n_stolen=5, seed=3)


def test_generator_is_deterministic_and_labeled():
    a1, r1 = generate(SMALL)
    a2, r2 = generate(SMALL)
    assert a1.equals(a2) and r1.equals(r2)
    assert set(a1["actor"]) >= set(ABUSE_TYPES)
    assert (a1["actor"] == "distillation_ring").sum() == 30


def test_rings_link_into_one_component_and_are_synchronized():
    a, r = generate(SMALL)
    comp = link_accounts(a, r)
    ring_ids = a.loc[a["actor"] == "distillation_ring", "account_id"]
    assert comp.loc[ring_ids].nunique() == 1  # the whole ring is one body
    syn = component_synchrony(r, comp, SMALL.days)
    assert syn.loc[comp.loc[ring_ids].iloc[0]] > 0.5
    # legit accounts stay (mostly) singletons: no shared identifiers
    legit_ids = a.loc[a["actor"].str.startswith("legit"), "account_id"]
    assert comp.loc[legit_ids].nunique() > 0.9 * len(legit_ids)


def test_unmeasured_synchrony_is_nan_not_zero():
    a, r = generate(SMALL)
    comp = link_accounts(a, r)
    syn = component_synchrony(r, comp, SMALL.days)
    singles = comp.value_counts()
    single_comp = singles[singles == 1].index[0]
    assert np.isnan(syn.loc[single_comp])


def test_queue_is_precision_first():
    a, r = generate(SMALL)
    feats = account_features(a, r, SMALL.days)
    comp = link_accounts(a, r)
    syn = component_synchrony(r, comp, SMALL.days)
    scored = account_scores(feats, seed=0)
    cases = component_cases(scored, comp, syn)
    res = evaluate(cases, scored, comp, ks=(3, 5))
    assert res["precision@3"] == 1.0
    assert res["legit_accounts_in_queue@3"] == 0  # the top of the queue must not contain a paying customer
    top = cases.iloc[0]
    assert isinstance(top["evidence"], list) and top["evidence"]  # every case carries readable evidence
