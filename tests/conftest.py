import matplotlib
matplotlib.use("Agg")

import numpy as np
import pandas as pd
import pytest

from infogain.data.synthetic import default_design, generate


@pytest.fixture(scope="session")
def small_cohort():
    design = default_design()
    design.modality_dims = {"demographics": 6, "vitals": 8, "labs": 20,
                            "ecg": 12, "cxr": 10, "echo": 8}
    design.n_nuisance_factors = 1
    return generate(design, n=1500, seed=0)


@pytest.fixture(scope="session")
def mimic_tables():
    """MIMIC-IV-shaped tables with hand-checkable outcome truth.

    Four admissions covering the interesting corners: an in-hospital death, a
    heart-failure readmission inside the window, an AKI by the relative
    criterion, and an ICU transfer inside 48 h.
    """
    T = pd.Timestamp("2150-03-01 08:00")
    admissions = pd.DataFrame([
        # died in hospital on day 3 -> mortality_30d = 1, readmission not evaluable
        dict(subject_id=1, hadm_id=101, admittime=T, dischtime=T + pd.Timedelta(days=3),
             deathtime=T + pd.Timedelta(days=3), edregtime=T - pd.Timedelta(hours=2),
             edouttime=T, admission_type="EW EMER.", insurance="Medicare",
             race="WHITE", hospital_expire_flag=1),
        # survived; readmitted with HF on day 12 after discharge -> readmission = 1
        dict(subject_id=2, hadm_id=102, admittime=T, dischtime=T + pd.Timedelta(days=4),
             deathtime=pd.NaT, edregtime=T - pd.Timedelta(hours=1), edouttime=T,
             admission_type="EW EMER.", insurance="Other", race="BLACK/AFRICAN",
             hospital_expire_flag=0),
        dict(subject_id=2, hadm_id=103, admittime=T + pd.Timedelta(days=16),
             dischtime=T + pd.Timedelta(days=20), deathtime=pd.NaT, edregtime=pd.NaT,
             edouttime=pd.NaT, admission_type="EW EMER.", insurance="Other",
             race="BLACK/AFRICAN", hospital_expire_flag=0),
        # survived; no readmission
        dict(subject_id=3, hadm_id=104, admittime=T, dischtime=T + pd.Timedelta(days=2),
             deathtime=pd.NaT, edregtime=T - pd.Timedelta(hours=3), edouttime=T,
             admission_type="ELECTIVE", insurance="Private", race="ASIAN",
             hospital_expire_flag=0),
    ])
    patients = pd.DataFrame([
        dict(subject_id=1, gender="M", anchor_age=70, anchor_year=2150,
             anchor_year_group="2014 - 2016", dod=T + pd.Timedelta(days=3)),
        dict(subject_id=2, gender="F", anchor_age=64, anchor_year=2150,
             anchor_year_group="2017 - 2019", dod=pd.NaT),
        dict(subject_id=3, gender="F", anchor_age=45, anchor_year=2150,
             anchor_year_group="2017 - 2019", dod=pd.NaT),
    ])
    diagnoses = pd.DataFrame([
        dict(subject_id=1, hadm_id=101, seq_num=1, icd_code="I5023", icd_version=10),
        dict(subject_id=2, hadm_id=102, seq_num=1, icd_code="I509", icd_version=10),
        dict(subject_id=2, hadm_id=103, seq_num=1, icd_code="4280", icd_version=9),
        dict(subject_id=3, hadm_id=104, seq_num=1, icd_code="J189", icd_version=10),
    ])
    transfers = pd.DataFrame([
        dict(subject_id=1, hadm_id=101, transfer_id=1, eventtype="admit",
             careunit="Emergency Department", intime=T - pd.Timedelta(hours=2),
             outtime=T),
        dict(subject_id=1, hadm_id=101, transfer_id=2, eventtype="transfer",
             careunit="Medical Intensive Care Unit (MICU)",
             intime=T + pd.Timedelta(hours=10), outtime=T + pd.Timedelta(days=3)),
        dict(subject_id=2, hadm_id=102, transfer_id=3, eventtype="admit",
             careunit="Medicine", intime=T, outtime=T + pd.Timedelta(days=4)),
        dict(subject_id=3, hadm_id=104, transfer_id=4, eventtype="admit",
             careunit="Medicine", intime=T, outtime=T + pd.Timedelta(days=2)),
    ])
    creatinine = pd.DataFrame([
        # subject 2: 0.9 -> 1.6 within 7 days = 1.78x baseline -> AKI
        dict(hadm_id=102, charttime=T - pd.Timedelta(hours=6), valuenum=0.9),
        dict(hadm_id=102, charttime=T + pd.Timedelta(hours=30), valuenum=1.6),
        # subject 3: stable
        dict(hadm_id=104, charttime=T - pd.Timedelta(hours=4), valuenum=1.0),
        dict(hadm_id=104, charttime=T + pd.Timedelta(hours=20), valuenum=1.05),
        dict(hadm_id=101, charttime=T - pd.Timedelta(hours=1), valuenum=1.2),
        dict(hadm_id=101, charttime=T + pd.Timedelta(hours=20), valuenum=1.25),
    ])
    index = pd.DataFrame([
        dict(subject_id=1, hadm_id=101, index_time=T),
        dict(subject_id=2, hadm_id=102, index_time=T),
        dict(subject_id=3, hadm_id=104, index_time=T),
    ])
    return dict(admissions=admissions, patients=patients, diagnoses=diagnoses,
                transfers=transfers, creatinine=creatinine, index=index)
