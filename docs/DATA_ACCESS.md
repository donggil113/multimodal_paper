# Getting the data

INFOGAIN links four credentialed PhysioNet resources by `subject_id`. That
linkage is the reason the study is possible at all: it is the only public
resource where the same patient's electrocardiogram, chest radiograph,
laboratory panel and clinical notes are all present on a common clock, so the
question "given what I already have, is the next test worth ordering" can be
asked of real patients rather than simulated ones.

Nothing in this repository ships patient data, and nothing should be committed
that derives from it. `.gitignore` blocks `data/` for that reason.

## 1. Credentialing

All four modules require a credentialed PhysioNet account:

1. Register at <https://physionet.org/register/>.
2. Complete CITI "Data or Specimens Only Research" training and upload the
   completion report.
3. Sign the PhysioNet Credentialed Health Data Use Agreement (v1.5.0) on each
   project page below.

Approval typically takes days to weeks. There is no way around it, and a
"de-identified sample" from a third party is not a substitute.

## 2. Downloads

| Module | Project page | Approx. size | Needed for |
|---|---|---|---|
| MIMIC-IV v3.1 (`hosp`, `icu`) | <https://physionet.org/content/mimiciv/> | 7 GB | cohort, labs, meds, outcomes |
| MIMIC-IV-ECG v1.0 | <https://physionet.org/content/mimic-iv-ecg/> | 90 GB (1 GB metadata only) | ECG |
| MIMIC-CXR-JPG v2.1.0 | <https://physionet.org/content/mimic-cxr-jpg/> | 570 GB (1 GB labels only) | chest radiograph |
| MIMIC-IV-Note v2.2 | <https://physionet.org/content/mimic-iv-note/> | 5 GB | notes |
| MIMIC-IV-ED v2.2 (optional) | <https://physionet.org/content/mimic-iv-ed/> | 2 GB | triage vitals |

**The headline analysis needs only the small files.** ECG machine measurements
(`machine_measurements.csv`, ~200 MB) and CheXpert labels
(`mimic-cxr-2.0.0-chexpert.csv.gz`, ~10 MB) replace the 660 GB of raw signal and
pixels. Use `infogain.encoders.waveform` / `infogain.encoders.image` only if you
want the learned-representation comparison.

Metadata-only download (recommended first pass):

```bash
export PN_USER=your_physionet_username
mkdir -p data/raw && cd data/raw

# MIMIC-IV core
wget -r -N -c -np --user $PN_USER --ask-password \
  https://physionet.org/files/mimiciv/3.1/

# ECG: record list + machine measurements only
wget -r -N -c -np --user $PN_USER --ask-password \
  -A "record_list.csv,machine_measurements.csv,*.csv.gz" \
  --reject "*.dat,*.hea" \
  https://physionet.org/files/mimic-iv-ecg/1.0/

# CXR: metadata + labels only
wget -r -N -c -np --user $PN_USER --ask-password \
  -A "mimic-cxr-2.0.0-metadata.csv.gz,mimic-cxr-2.0.0-chexpert.csv.gz,mimic-cxr-2.0.0-split.csv.gz,cxr-record-list.csv.gz,cxr-study-list.csv.gz" \
  https://physionet.org/files/mimic-cxr-jpg/2.1.0/

# Notes
wget -r -N -c -np --user $PN_USER --ask-password \
  https://physionet.org/files/mimic-iv-note/2.2/

# ED (optional, gives triage vitals for the whole ED cohort)
wget -r -N -c -np --user $PN_USER --ask-password \
  https://physionet.org/files/mimic-iv-ed/2.2/
```

## 3. Expected layout

`MimicPaths.from_root` accepts several common layouts. The canonical one:

```
data/raw/
├── mimiciv/{hosp,icu}/            # admissions.csv.gz, labevents.csv.gz, ...
├── mimic-iv-ecg/                  # record_list.csv, machine_measurements.csv
├── mimic-cxr-jpg/                 # mimic-cxr-2.0.0-metadata.csv.gz, ...
├── mimic-iv-note/note/            # discharge.csv.gz, radiology.csv.gz
└── mimic-iv-ed/ed/                # edstays.csv.gz, triage.csv.gz, vitalsign.csv.gz
```

Point the pipeline at it either way:

```bash
export MIMIC_ROOT=$PWD/data/raw          # or pass --root
python -m infogain.data.mimic_cohort --root data/raw --out data/cohort_mimic \
    --split temporal
```

Individual modules can also be set separately (`MIMIC_HOSP`, `MIMIC_ICU`,
`MIMIC_ECG`, `MIMIC_CXR`, `MIMIC_NOTE`, `MIMIC_ED`). A missing module is not an
error: its modality is simply "not acquired" for every patient, which is the
same missingness pattern the estimator already handles.

## 4. Cost and runtime notes

- `labevents` is ~158 M rows in v3.1. It is streamed and filtered to the cohort
  once, then cached as Parquet under `<root>/_infogain_cache/`; budget ~20 min
  and 8 GB of RAM for the first pass, seconds thereafter.
- Everything else fits comfortably in memory.
- Converting the `.csv.gz` files to Parquet up front roughly halves total
  extraction time; the readers accept either.

## 5. Running without PhysioNet access

The simulator produces a cohort with the same schema, the same modality names,
realistic prevalences and informative test ordering — plus exact ground-truth
information decomposition, which real data cannot provide:

```bash
python scripts/make_synthetic_cohort.py --n 50000 --out data/cohort_synthetic
python -m infogain.experiments.run_cohort --cohort data/cohort_synthetic \
    --outcome mortality_30d
```

Every analysis, figure and theorem check in the paper runs on either input
unchanged. Results from the simulator validate the estimator; they are not a
substitute for the clinical findings.

## 6. Data use

The PhysioNet DUA prohibits attempting re-identification and sharing the data
with anyone not individually credentialed. Derived cohorts, model weights fitted
on patient rows, and cached Parquet extracts all inherit those terms — keep them
out of version control and off shared storage.
