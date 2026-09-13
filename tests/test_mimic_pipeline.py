"""Dry-run of the real PhysioNet extraction against synthetic MIMIC-shaped files.

The MIMIC-IV path cannot be exercised on real data without credentialing, and it
is the code a new user runs first -- so it is worth testing hardest. This module
writes files with the actual MIMIC-IV / -ECG / -CXR / -Note / -ED table names,
column names, dtypes and compression into a temporary tree and calls
``build_cohort`` end to end. It catches exactly the class of bug that is
otherwise discovered only after a 600 GB download: a renamed column, a mis-typed
itemid, a join that silently produces zero rows.
"""
from __future__ import annotations

import gzip

import numpy as np
import pandas as pd
import pytest

from infogain.data.mimic_cohort import CohortConfig, build_cohort, build_index_frame
from infogain.data.mimic_io import MimicPaths, load_core_tables, read_table
from infogain.data.mimic_modalities import TemporalPolicy

N_PATIENTS = 60
T0 = pd.Timestamp("2150-01-01")


def _write(df: pd.DataFrame, path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt", newline="") as fh:
        df.to_csv(fh, index=False)


@pytest.fixture(scope="module")
def mimic_tree(tmp_path_factory):
    """A miniature MIMIC-IV installation in the layout PhysioNet ships."""
    root = tmp_path_factory.mktemp("mimic")
    rng = np.random.default_rng(0)
    hosp = root / "mimiciv" / "hosp"
    icu = root / "mimiciv" / "icu"
    ecg = root / "mimic-iv-ecg"
    cxr = root / "mimic-cxr-jpg"
    note = root / "mimic-iv-note" / "note"
    ed = root / "mimic-iv-ed" / "ed"

    subject_ids = np.arange(10000001, 10000001 + N_PATIENTS)
    hadm_ids = np.arange(20000001, 20000001 + N_PATIENTS)
    admit = [T0 + pd.Timedelta(days=int(d)) for d in rng.integers(0, 900, N_PATIENTS)]
    los = rng.integers(24, 240, N_PATIENTS)
    disch = [a + pd.Timedelta(hours=int(h)) for a, h in zip(admit, los)]
    died = rng.random(N_PATIENTS) < 0.12

    _write(pd.DataFrame({
        "subject_id": subject_ids, "hadm_id": hadm_ids,
        "admittime": admit, "dischtime": disch,
        "deathtime": [d if k else pd.NaT for d, k in zip(disch, died)],
        "admission_type": rng.choice(["EW EMER.", "URGENT", "ELECTIVE"], N_PATIENTS),
        "admission_location": "EMERGENCY ROOM",
        "discharge_location": "HOME",
        "insurance": rng.choice(["Medicare", "Medicaid", "Other"], N_PATIENTS),
        "language": "ENGLISH", "marital_status": "SINGLE",
        "race": rng.choice(["WHITE", "BLACK/AFRICAN AMERICAN", "ASIAN", "OTHER"],
                           N_PATIENTS),
        "edregtime": [a - pd.Timedelta(hours=3) for a in admit],
        "edouttime": admit,
        "hospital_expire_flag": died.astype(int),
    }), hosp / "admissions.csv.gz")

    _write(pd.DataFrame({
        "subject_id": subject_ids, "gender": rng.choice(["M", "F"], N_PATIENTS),
        "anchor_age": rng.integers(30, 90, N_PATIENTS), "anchor_year": 2150,
        "anchor_year_group": rng.choice(["2011 - 2013", "2014 - 2016", "2017 - 2019"],
                                        N_PATIENTS),
        "dod": [d + pd.Timedelta(days=5) if k else pd.NaT
                for d, k in zip(disch, died)],
    }), hosp / "patients.csv.gz")

    # a mix of HF, pneumonia and ESRD codes across both ICD versions
    dx = []
    for i, (s, h) in enumerate(zip(subject_ids, hadm_ids)):
        code, ver = [("I5023", 10), ("4280", 9), ("J189", 10), ("N186", 10)][i % 4]
        dx.append({"subject_id": s, "hadm_id": h, "seq_num": 1,
                   "icd_code": code, "icd_version": ver})
    _write(pd.DataFrame(dx), hosp / "diagnoses_icd.csv.gz")

    tr = []
    for i, (s, h, a, d) in enumerate(zip(subject_ids, hadm_ids, admit, disch)):
        tr.append({"subject_id": s, "hadm_id": h, "transfer_id": 3 * i,
                   "eventtype": "admit", "careunit": "Medicine",
                   "intime": a, "outtime": d})
        if i % 5 == 0:
            tr.append({"subject_id": s, "hadm_id": h, "transfer_id": 3 * i + 1,
                       "eventtype": "transfer",
                       "careunit": "Medical Intensive Care Unit (MICU)",
                       "intime": a + pd.Timedelta(hours=12), "outtime": d})
    _write(pd.DataFrame(tr), hosp / "transfers.csv.gz")

    _write(pd.DataFrame([
        {"itemid": 50912, "label": "Creatinine", "fluid": "Blood", "category": "Chemistry"},
        {"itemid": 51006, "label": "Urea Nitrogen", "fluid": "Blood", "category": "Chemistry"},
        {"itemid": 50983, "label": "Sodium", "fluid": "Blood", "category": "Chemistry"},
        {"itemid": 51222, "label": "Hemoglobin", "fluid": "Blood", "category": "Hematology"},
        {"itemid": 51301, "label": "White Blood Cells", "fluid": "Blood",
         "category": "Hematology"},
        {"itemid": 50813, "label": "Lactate", "fluid": "Blood", "category": "Blood Gas"},
    ]), hosp / "d_labitems.csv.gz")

    labs = []
    for s, h, a in zip(subject_ids, hadm_ids, admit):
        for itemid, lo, hi in [(50912, 0.5, 3.0), (51006, 8, 60), (50983, 130, 145),
                               (51222, 8, 16), (51301, 4, 20), (50813, 0.5, 5)]:
            for dt, scale in ((-6, 1.0), (30, 1.6)):
                labs.append({"labevent_id": len(labs), "subject_id": s, "hadm_id": h,
                             "specimen_id": len(labs), "itemid": itemid,
                             "charttime": a + pd.Timedelta(hours=dt),
                             "storetime": a + pd.Timedelta(hours=dt + 1),
                             "value": "x",
                             "valuenum": float(rng.uniform(lo, hi) * scale),
                             "valueuom": "mg/dL", "ref_range_lower": lo,
                             "ref_range_upper": hi, "flag": "", "priority": "ROUTINE",
                             "comments": ""})
    _write(pd.DataFrame(labs), hosp / "labevents.csv.gz")

    rx = []
    for s, h, a in zip(subject_ids, hadm_ids, admit):
        for drug in ("Furosemide", "Metoprolol Tartrate", "Aspirin"):
            rx.append({"subject_id": s, "hadm_id": h, "pharmacy_id": len(rx),
                       "starttime": a - pd.Timedelta(hours=12),
                       "stoptime": a + pd.Timedelta(hours=48), "drug_type": "MAIN",
                       "drug": drug, "route": "PO"})
    _write(pd.DataFrame(rx), hosp / "prescriptions.csv.gz")

    _write(pd.DataFrame({
        "subject_id": subject_ids[:20], "hadm_id": hadm_ids[:20],
        "stay_id": np.arange(30000001, 30000021),
        "first_careunit": "MICU", "last_careunit": "MICU",
        "intime": admit[:20], "outtime": disch[:20], "los": 2.0,
    }), icu / "icustays.csv.gz")

    # ---- MIMIC-IV-ED ----
    ed_stay_ids = np.arange(40000001, 40000001 + N_PATIENTS)
    _write(pd.DataFrame({
        "subject_id": subject_ids, "hadm_id": hadm_ids, "stay_id": ed_stay_ids,
        "intime": [a - pd.Timedelta(hours=3) for a in admit], "outtime": admit,
        "gender": "M", "race": "WHITE", "arrival_transport": "AMBULANCE",
        "disposition": "ADMITTED",
    }), ed / "edstays.csv.gz")
    _write(pd.DataFrame({
        "subject_id": subject_ids, "stay_id": ed_stay_ids,
        "temperature": rng.uniform(97, 101, N_PATIENTS),
        "heartrate": rng.uniform(60, 120, N_PATIENTS),
        "resprate": rng.uniform(12, 28, N_PATIENTS),
        "o2sat": rng.uniform(88, 100, N_PATIENTS),
        "sbp": rng.uniform(90, 170, N_PATIENTS),
        "dbp": rng.uniform(50, 100, N_PATIENTS),
        "pain": rng.integers(0, 10, N_PATIENTS),
        "acuity": rng.integers(1, 5, N_PATIENTS), "chiefcomplaint": "Chest pain",
    }), ed / "triage.csv.gz")

    # ---- MIMIC-IV-ECG ----
    study_ids = np.arange(50000001, 50000001 + N_PATIENTS)
    _write(pd.DataFrame({
        "subject_id": subject_ids, "study_id": study_ids, "file_name": "0000",
        "ecg_time": [a - pd.Timedelta(hours=2) for a in admit],
        "path": [f"files/p{s}/s{t}/0000" for s, t in zip(subject_ids, study_ids)],
    }), ecg / "record_list.csv.gz")
    mm = pd.DataFrame({
        "subject_id": subject_ids, "study_id": study_ids, "cart_id": 1,
        "ecg_time": [a - pd.Timedelta(hours=2) for a in admit],
        "rr_interval": rng.uniform(500, 1200, N_PATIENTS),
        "p_onset": rng.uniform(100, 200, N_PATIENTS),
        "p_end": rng.uniform(200, 260, N_PATIENTS),
        "qrs_onset": rng.uniform(260, 300, N_PATIENTS),
        "qrs_end": rng.uniform(340, 420, N_PATIENTS),
        "t_end": rng.uniform(500, 650, N_PATIENTS),
        "p_axis": rng.uniform(-90, 90, N_PATIENTS),
        "qrs_axis": rng.uniform(-90, 90, N_PATIENTS),
        "t_axis": rng.uniform(-90, 90, N_PATIENTS),
    })
    for k in range(18):
        mm[f"report_{k}"] = rng.choice(
            ["Sinus rhythm", "Atrial fibrillation", "Left ventricular hypertrophy",
             "Normal ECG", ""], N_PATIENTS)
    _write(mm, ecg / "machine_measurements.csv.gz")

    # ---- MIMIC-CXR ----
    cxr_study = np.arange(60000001, 60000001 + N_PATIENTS)
    times = [a - pd.Timedelta(hours=4) for a in admit]
    _write(pd.DataFrame({
        "dicom_id": [f"d{i}" for i in range(N_PATIENTS)],
        "subject_id": subject_ids, "study_id": cxr_study,
        "PerformedProcedureStepDescription": "CHEST",
        "ViewPosition": rng.choice(["PA", "AP", "LATERAL"], N_PATIENTS),
        "Rows": 2544, "Columns": 3056,
        "StudyDate": [int(t.strftime("%Y%m%d")) for t in times],
        "StudyTime": [float(t.strftime("%H%M%S")) for t in times],
    }), cxr / "mimic-cxr-2.0.0-metadata.csv.gz")
    chex = pd.DataFrame({"subject_id": subject_ids, "study_id": cxr_study})
    for lab in ("Atelectasis", "Cardiomegaly", "Consolidation", "Edema",
                "Enlarged Cardiomediastinum", "Fracture", "Lung Lesion",
                "Lung Opacity", "No Finding", "Pleural Effusion", "Pleural Other",
                "Pneumonia", "Pneumothorax", "Support Devices"):
        chex[lab] = rng.choice([1.0, 0.0, -1.0, np.nan], N_PATIENTS)
    _write(chex, cxr / "mimic-cxr-2.0.0-chexpert.csv.gz")

    # ---- MIMIC-IV-Note ----
    _write(pd.DataFrame({
        "note_id": [f"r{i}" for i in range(N_PATIENTS)],
        "subject_id": subject_ids, "hadm_id": hadm_ids,
        "note_type": "RR", "note_seq": 1,
        "charttime": [a - pd.Timedelta(hours=4) for a in admit],
        "storetime": [a - pd.Timedelta(hours=3) for a in admit],
        "text": ["Chest radiograph shows mild pulmonary vascular congestion "
                 "with small bilateral pleural effusions and cardiomegaly."] * N_PATIENTS,
    }), note / "radiology.csv.gz")
    _write(pd.DataFrame({
        "note_id": [f"d{i}" for i in range(N_PATIENTS)],
        "subject_id": subject_ids, "hadm_id": hadm_ids,
        "note_type": "DS", "note_seq": 1,
        "charttime": disch, "storetime": disch,
        "text": ["Discharge summary: the patient died of heart failure."] * N_PATIENTS,
    }), note / "discharge.csv.gz")
    return root


def test_paths_resolve_every_module(mimic_tree):
    paths = MimicPaths.from_root(mimic_tree)
    assert all(paths.available().values()), paths.describe()


def test_core_tables_load_with_parsed_dates(mimic_tree):
    tb = load_core_tables(MimicPaths.from_root(mimic_tree))
    assert len(tb.admissions) == N_PATIENTS
    assert str(tb.admissions["admittime"].dtype).startswith("datetime")
    assert tb.d_labitems is not None and len(tb.d_labitems) == 6


def test_reader_is_case_insensitive_about_columns(mimic_tree):
    """MIMIC-CXR ships CamelCase headers; the readers must not care."""
    meta = read_table(MimicPaths.from_root(mimic_tree).cxr,
                      "mimic-cxr-2.0.0-metadata",
                      usecols=["subject_id", "study_id", "ViewPosition"])
    assert set(meta.columns) == {"subject_id", "study_id", "viewposition"}


def test_index_frame_computes_age_and_anchors_at_ed(mimic_tree):
    tb = load_core_tables(MimicPaths.from_root(mimic_tree))
    idx = build_index_frame(tb, CohortConfig())
    assert len(idx) > 0
    assert (idx["age"] >= 18).all()
    # the ED anchor must precede admission
    assert (idx["index_time"] <= idx["admittime"]).all()


def test_hf_restriction_narrows_the_cohort(mimic_tree):
    tb = load_core_tables(MimicPaths.from_root(mimic_tree))
    all_adm = build_index_frame(tb, CohortConfig())
    hf = build_index_frame(tb, CohortConfig(require_hf=True))
    assert 0 < len(hf) < len(all_adm)


@pytest.fixture(scope="module")
def built(mimic_tree):
    paths = MimicPaths.from_root(mimic_tree)
    cfg = CohortConfig(policy=TemporalPolicy(), split_mode="patient", seed=0,
                       note_components=8)
    return build_cohort(paths, cfg)


def test_cohort_builds_end_to_end(built):
    c = built
    assert c.n > 0
    assert set(c.spec.names) >= {"demographics", "vitals", "labs", "meds",
                                 "ecg", "cxr", "notes"}
    assert c.meta["source"] == "mimic-iv"


def test_every_modality_actually_populated(built):
    """A silent join failure shows up as a block that is present but empty."""
    for name in ("labs", "vitals", "ecg", "cxr", "notes", "meds"):
        blk = built.blocks[name]
        assert blk.availability > 0.5, f"{name} observed for only {blk.availability:.0%}"
        assert np.abs(blk.values[blk.observed]).sum() > 0, f"{name} is all zeros"


def test_lab_values_land_in_the_right_columns(built):
    blk = built.blocks["labs"]
    j = blk.feature_names.index("lab_creatinine")
    vals = blk.values[:, j]
    measured = blk.values[:, j + 1] > 0
    assert measured.mean() > 0.8
    assert (vals[measured] > 0).all() and (vals[measured] < 25).all()


def test_ecg_intervals_are_derived_not_copied(built):
    blk = built.blocks["ecg"]
    names = blk.feature_names
    assert "ecg_qrs_duration" in names and "ecg_qtc_bazett" in names
    qrs = blk.values[:, names.index("ecg_qrs_duration")]
    assert (qrs[blk.observed] > 0).all()
    assert any(n.startswith("ecg_rep_") for n in names)


def test_cxr_labels_expand_over_four_states(built):
    names = built.blocks["cxr"].feature_names
    for suffix in ("_pos", "_neg", "_unc", "_mentioned"):
        assert any(n.endswith(suffix) for n in names), suffix


def test_discharge_summaries_excluded_by_default(built):
    """The default policy forbids them; leaking one would be the classic error."""
    assert built.meta["config"]["policy"]["allow_discharge_summary"] is False


def test_outcomes_present_and_plausible(built):
    for oc in ("mortality_30d", "hf_readmission_30d", "aki_7d", "icu_transfer_48h"):
        assert oc in built.outcome_names
        p = built.prevalence(oc)
        assert np.isnan(p) or 0.0 <= p <= 1.0


def test_aki_detected_from_the_rising_creatinine(built):
    """Every synthetic patient's creatinine rises 1.6x, so AKI must fire."""
    y = built.outcomes["aki_7d"].to_numpy(dtype=float)
    evaluable = np.isfinite(y)
    assert evaluable.sum() > 0
    assert np.nanmean(y[evaluable]) > 0.5


def test_icu_transfer_matches_the_seeded_rate(built):
    """One patient in five was given a MICU transfer at +12 h."""
    y = built.outcomes["icu_transfer_48h"].to_numpy(dtype=float)
    ev = np.isfinite(y)
    assert 0.05 < np.nanmean(y[ev]) < 0.45


def test_splits_are_patient_level(built):
    g = built.index.groupby("subject_id")["split"].nunique()
    assert (g == 1).all()


def test_cohort_survives_a_save_load_round_trip(built, tmp_path):
    from infogain.data.schema import Cohort

    built.save(tmp_path / "c")
    back = Cohort.load(tmp_path / "c")
    assert back.n == built.n and back.spec.names == built.spec.names
    for m in built.spec.names:
        assert np.allclose(back.blocks[m].values, built.blocks[m].values)


def test_missing_optional_module_degrades_to_unobserved(mimic_tree, tmp_path):
    """Absent ECG must give an unobserved modality, not a crash."""
    paths = MimicPaths.from_root(mimic_tree)
    paths.ecg = None
    c = build_cohort(paths, CohortConfig(note_components=8))
    assert "ecg" not in c.spec.names
    assert c.n > 0
