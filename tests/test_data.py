"""Cohort schema, splits, MIMIC outcome definitions, and simulator ground truth."""
import numpy as np
import pandas as pd
import pytest

from infogain.data import outcomes as oc
from infogain.data.mimic_modalities import (
    TemporalPolicy,
    extract_labs,
    extract_meds,
    resolve_lab_itemids,
)
from infogain.data.schema import (
    Cohort, ModalityBlock, assign_patient_splits, assign_temporal_splits,
)


# --------------------------------------------------------------------------- #
# Schema
# --------------------------------------------------------------------------- #
def test_block_zeroes_unobserved_rows():
    b = ModalityBlock("x", np.arange(12, dtype=float).reshape(4, 3),
                      np.array([True, False, True, False]))
    assert np.all(b.values[1] == 0) and np.all(b.values[3] == 0)
    assert b.values[0, 1] == 1.0
    assert b.availability == 0.5


def test_cohort_round_trip(tmp_path, small_cohort):
    cohort, _ = small_cohort
    cohort.save(tmp_path / "c")
    back = Cohort.load(tmp_path / "c")
    assert back.n == cohort.n
    assert back.spec.names == cohort.spec.names
    for m in cohort.spec.names:
        assert np.allclose(back.blocks[m].values, cohort.blocks[m].values)
        assert np.array_equal(back.blocks[m].observed, cohort.blocks[m].observed)
    assert back.outcome_names == cohort.outcome_names


def test_splits_never_span_a_subject():
    subs = np.repeat(np.arange(200), 3)
    sp = assign_patient_splits(subs, seed=1)
    df = pd.DataFrame({"s": subs, "sp": sp})
    assert (df.groupby("s")["sp"].nunique() == 1).all()


def test_temporal_split_holds_out_the_named_years():
    subs = np.arange(100)
    yg = np.where(subs < 60, "2014 - 2016", "2017 - 2019")
    sp = assign_temporal_splits(subs, yg, ("2017 - 2019",), seed=0)
    assert set(sp[yg == "2017 - 2019"]) == {"test"}
    assert "test" not in set(sp[yg == "2014 - 2016"])


def test_cohort_rejects_misaligned_blocks(small_cohort):
    cohort, _ = small_cohort
    bad = dict(cohort.blocks)
    m = cohort.spec.names[0]
    bad[m] = bad[m].subset(np.arange(cohort.n - 1))
    with pytest.raises(ValueError):
        Cohort(cohort.index, bad, cohort.outcomes, cohort.spec)


# --------------------------------------------------------------------------- #
# MIMIC outcome definitions
# --------------------------------------------------------------------------- #
def test_hf_code_matching():
    codes = pd.Series(["I5023", "428.0", "4280", "J189", "I110", "40201"])
    vers = pd.Series([10, 9, 9, 10, 10, 9])
    hits = oc.is_heart_failure(codes, vers).tolist()
    assert hits == [True, True, True, False, True, True]


def test_esrd_and_icu_matching():
    assert oc.is_esrd(pd.Series(["N186"]), pd.Series([10])).iloc[0]
    assert oc.is_esrd(pd.Series(["5856"]), pd.Series([9])).iloc[0]
    cu = pd.Series(["Medical Intensive Care Unit (MICU)", "Medicine", "Coronary Care Unit (CCU)"])
    assert oc.is_icu_careunit(cu).tolist() == [True, False, True]


def test_mortality_30d(mimic_tables):
    t = mimic_tables
    m = oc.mortality_within(t["index"], t["patients"], t["admissions"], 30)
    assert m.tolist()[0] == 1.0          # died on day 3
    assert m.tolist()[1] == 0.0
    assert m.tolist()[2] == 0.0


def test_hf_readmission(mimic_tables):
    t = mimic_tables
    r = oc.hf_readmission(t["index"], t["admissions"], t["diagnoses"], 30)
    assert np.isnan(r.iloc[0])           # died in hospital -> not evaluable
    assert r.iloc[1] == 1.0              # HF readmission 12 days after discharge
    assert r.iloc[2] == 0.0


def test_hf_readmission_outside_window(mimic_tables):
    t = mimic_tables
    adm = t["admissions"].copy()
    adm.loc[adm["hadm_id"] == 103, "admittime"] += pd.Timedelta(days=40)
    r = oc.hf_readmission(t["index"], adm, t["diagnoses"], 30)
    assert r.iloc[1] == 0.0


def test_kdigo_aki(mimic_tables):
    t = mimic_tables
    a = oc.kdigo_aki(t["index"], t["creatinine"], horizon_days=7)
    assert a.iloc[1] == 1.0              # 0.9 -> 1.6 is both >=1.5x and >=+0.3
    assert a.iloc[2] == 0.0              # 1.00 -> 1.05
    assert a.iloc[0] == 0.0


def test_kdigo_excludes_esrd(mimic_tables):
    t = mimic_tables
    a = oc.kdigo_aki(t["index"], t["creatinine"], 7, esrd_subject_ids={2})
    assert np.isnan(a.iloc[1])


def test_icu_transfer(mimic_tables):
    t = mimic_tables
    i = oc.icu_transfer(t["index"], t["transfers"], 48.0)
    assert i.iloc[0] == 1.0              # MICU at +10 h
    assert i.iloc[1] == 0.0


def test_icu_transfer_outside_window(mimic_tables):
    t = mimic_tables
    tr = t["transfers"].copy()
    tr.loc[tr["transfer_id"] == 2, "intime"] += pd.Timedelta(hours=60)
    assert oc.icu_transfer(t["index"], tr, 48.0).iloc[0] == 0.0


def test_summarize_outcomes_counts_non_evaluable():
    df = pd.DataFrame({"a": [1.0, 0.0, np.nan, 1.0]})
    s = oc.summarize_outcomes(df).iloc[0]
    assert s["n_evaluable"] == 3 and s["n_events"] == 2
    assert s["non_evaluable_frac"] == pytest.approx(0.25)


# --------------------------------------------------------------------------- #
# Modality extraction
# --------------------------------------------------------------------------- #
def test_lab_itemid_resolution_prefers_the_dictionary():
    d = pd.DataFrame([{"itemid": 99999, "label": "Creatinine", "fluid": "Blood"},
                      {"itemid": 88888, "label": "Hemoglobin", "fluid": "Blood"}])
    m = resolve_lab_itemids(d)
    assert m["creatinine"] == [99999]
    assert m["hemoglobin"] == [88888]
    assert m["lactate"]  # falls back to the hard-coded hint


def test_extract_labs_respects_the_index_window():
    T = pd.Timestamp("2150-01-01 12:00")
    index = pd.DataFrame([{"hadm_id": 1, "index_time": T}])
    ev = pd.DataFrame([
        {"hadm_id": 1, "itemid": 50912, "charttime": T - pd.Timedelta(hours=2), "valuenum": 1.4},
        # after the index: must be excluded
        {"hadm_id": 1, "itemid": 50912, "charttime": T + pd.Timedelta(hours=2), "valuenum": 9.9},
    ])
    blk = extract_labs(index, ev, {"creatinine": [50912]}, TemporalPolicy())
    j = blk.feature_names.index("lab_creatinine")
    assert blk.values[0, j] == pytest.approx(1.4)
    assert blk.values[0, j + 1] == 1.0


def test_extract_labs_marks_unmeasured():
    T = pd.Timestamp("2150-01-01 12:00")
    index = pd.DataFrame([{"hadm_id": 1, "index_time": T}])
    ev = pd.DataFrame(columns=["hadm_id", "itemid", "charttime", "valuenum"])
    blk = extract_labs(index, ev, {"creatinine": [50912]}, TemporalPolicy())
    assert blk.values[0, blk.feature_names.index("lab_creatinine_measured")] == 0.0
    assert not blk.observed[0]


def test_extract_labs_drops_implausible_values():
    T = pd.Timestamp("2150-01-01 12:00")
    index = pd.DataFrame([{"hadm_id": 1, "index_time": T}])
    ev = pd.DataFrame([{"hadm_id": 1, "itemid": 50912,
                        "charttime": T - pd.Timedelta(hours=1), "valuenum": 400.0}])
    blk = extract_labs(index, ev, {"creatinine": [50912]}, TemporalPolicy())
    assert blk.values[0, blk.feature_names.index("lab_creatinine_measured")] == 0.0


def test_extract_meds_flags_classes():
    T = pd.Timestamp("2150-01-01 12:00")
    index = pd.DataFrame([{"hadm_id": 1, "index_time": T},
                          {"hadm_id": 2, "index_time": T}])
    rx = pd.DataFrame([
        {"hadm_id": 1, "drug": "Furosemide", "starttime": T - pd.Timedelta(hours=5)},
        {"hadm_id": 1, "drug": "Metoprolol Tartrate", "starttime": T - pd.Timedelta(hours=5)},
        {"hadm_id": 2, "drug": "Furosemide", "starttime": T + pd.Timedelta(hours=5)},
    ])
    blk = extract_meds(index, rx, TemporalPolicy())
    j = blk.feature_names.index("med_loop_diuretic")
    assert blk.values[0, j] == 1.0
    assert blk.values[1, j] == 0.0       # started after the index time
    assert blk.values[0, blk.feature_names.index("med_beta_blocker")] == 1.0


# --------------------------------------------------------------------------- #
# Simulator
# --------------------------------------------------------------------------- #
def test_simulator_posterior_normalises(small_cohort):
    _, gt = small_cohort
    lp = gt.log_posterior_z({"vitals"})
    assert np.allclose(np.exp(lp).sum(axis=1), 1.0)


def test_simulator_information_is_monotone(small_cohort):
    cohort, gt = small_cohort
    base = {"demographics", "vitals"}
    i_base = gt.information("mortality_30d", base, respect_observation=False)
    i_more = gt.information("mortality_30d", base | {"labs"}, respect_observation=False)
    i_all = gt.information("mortality_30d", set(cohort.spec.names),
                           respect_observation=False)
    assert i_base <= i_more + 1e-9 <= i_all + 1e-9


def test_simulator_makes_pure_synergy(small_cohort):
    """ecg and cxr each carry a factor that only matters in combination."""
    _, gt = small_cohort
    base = {"demographics", "vitals"}
    d = gt.decompose("mortality_30d", "ecg", set(gt.design.modalities) - {"ecg"},
                     base, respect_observation=False)
    assert d["conditional"] > d["marginal"]
    assert d["synergistic"] > 0 and d["redundant"] == 0


def test_simulator_makes_redundancy(small_cohort):
    """labs and cxr share the congestion factor that drives HF readmission."""
    _, gt = small_cohort
    base = {"demographics", "vitals"}
    d = gt.decompose("hf_readmission_30d", "labs", base | {"cxr"}, base,
                     respect_observation=False)
    assert d["redundant"] > 0


def test_simulator_prevalences_close_to_target(small_cohort):
    cohort, _ = small_cohort
    assert 0.03 < cohort.prevalence("mortality_30d") < 0.13
    assert 0.15 < cohort.prevalence("hf_readmission_30d") < 0.28


def test_patient_gain_is_nonnegative_and_averages_to_the_cohort_gain(small_cohort):
    cohort, gt = small_cohort
    base = {"demographics", "vitals"}
    rows = np.arange(400)
    d = gt.patient_gain("mortality_30d", "labs", base, rows=rows, n_mc=48, seed=0)
    assert (d >= -1e-9).all()
    cohort_gain = gt.conditional_gain("mortality_30d", "labs", base, rows=rows)
    assert d.mean() == pytest.approx(cohort_gain, abs=0.01)
