# PS-21: Kidney Abnormality Detection from CT

Vetriprasath V · RA2311026050176 · Section A

Classifies kidney CT slices as Normal, Cyst, Stone or Tumour (CT Kidney Dataset, Kaggle, 12,446 images), with a
**patient-grouped split** to avoid the slice-level leakage that inflates random-split results.

| Run (test set) | Accuracy | Macro-F1 | Stone recall |
|---|---|---|---|
| ResNet-18, random split (leaky) | 95.07 % | 94.20 % | 96.14 % |
| **ResNet-18, grouped split (honest)** | **80.27 %** | **69.74 %** | **95.05 %** |

## Files
- `RA2311026050176_PS21.ipynb`: the full notebook with all outputs (EDA, leakage analysis, splits, training, evaluation, Grad-CAM)
- `RA2311026050176_PS21_Report.pdf`: the two-page report
- `outputs/`: figures, `splits.csv` (the exact split), `results.json` (every reported number)
- `make_report.py`: rebuilds the report PDF from `outputs/results.json`

## Reproduce
```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
# download the dataset and point data/kidney at the folder holding Normal/ Cyst/ Stone/ Tumor/
KAGGLEHUB_CACHE=data .venv/bin/python -c "import kagglehub; print(kagglehub.dataset_download('nazmul0087/ct-kidney-dataset-normal-cyst-tumor-and-stone'))"
```
Then run all cells of the notebook top to bottom (seed 42). Model weights are not committed (size); they are retrained by the notebook.

Dataset: M. N. Islam et al., *Scientific Reports* 12, 11440 (2022). AI assistance (Claude) was used for code and drafting; see the report.
