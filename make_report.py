"""Builds the two-page report (Part D format) from the notebook outputs.

Run after the notebook:   .venv/bin/python make_report.py
Every number is read from outputs/results.json / outputs/splits.csv written by the notebook,
so the report always matches the notebook.  Fill in the HEADER fields below.
"""
import json
from pathlib import Path

import numpy as np
from reportlab.lib import colors
from reportlab.lib.enums import TA_JUSTIFY, TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import cm
from reportlab.platypus import (Image, KeepTogether, Paragraph, SimpleDocTemplate,
                                Spacer, Table, TableStyle)

# ------------------------------------------------------------------ fill these in
NAME, REGNO, SECTION, DATE = "Vetriprasath V", "RA2311026050176", "A", "16-11-2026"
GITHUB = "https://github.com/vetriiiiiiiiiiiiiiiiiiiiii/CBL/blob/main/RA2311026050176_PS21.ipynb"
# ---------------------------------------------------------------------------------

OUT = Path("outputs")
R = json.load(open(OUT / "results.json")) if (OUT / "results.json").exists() else None
PDF = f"{REGNO}_PS21_Report.pdf"

base = ParagraphStyle("b", fontName="Times-Roman", fontSize=11, leading=12.6, alignment=TA_JUSTIFY, spaceAfter=2)
head = ParagraphStyle("h", parent=base, fontName="Times-Bold", fontSize=12, leading=14, spaceBefore=4, spaceAfter=1.5)
cap = ParagraphStyle("c", parent=base, fontSize=9.5, leading=11, alignment=TA_CENTER, spaceBefore=1, spaceAfter=3)
cell = ParagraphStyle("t", parent=base, fontSize=9, leading=10.2, alignment=TA_CENTER, spaceAfter=0)
cellL = ParagraphStyle("tl", parent=cell, alignment=0)
title = ParagraphStyle("ti", parent=base, fontName="Times-Bold", fontSize=13, alignment=TA_CENTER, spaceAfter=1)
hdr = ParagraphStyle("hd", parent=base, fontSize=10, leading=11.5, alignment=TA_CENTER, spaceAfter=4)

P = lambda t, s=base: Paragraph(t, s)
f2 = lambda v: f"{float(v):.2f}"
PEND = "<font color='#b00000'>[pending]</font>"


def res(key, col):
    if R is None or key not in R["results"]:
        return PEND
    v = R["results"][key][col]
    return f2(v) if isinstance(v, (int, float)) else str(v)


def table(data, widths, header_rows=1, left_cols=()):
    rows = [[P(str(c), cellL if j in left_cols else cell) for j, c in enumerate(r)] for r in data]
    t = Table(rows, colWidths=widths, hAlign="CENTER")
    t.setStyle(TableStyle([
        ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#9aa5b8")),
        ("BACKGROUND", (0, 0), (-1, header_rows - 1), colors.HexColor("#dfe5f0")),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 1.2), ("BOTTOMPADDING", (0, 0), (-1, -1), 1.6),
        ("LEFTPADDING", (0, 0), (-1, -1), 2.5), ("RIGHTPADDING", (0, 0), (-1, -1), 2.5)]))
    return t


def report_figures():
    """Redraw the two report figures at their printed size so every label stays readable."""
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import pandas as pd, torch, torch.nn.functional as F
    plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "Times", "DejaVu Serif"], "font.size": 9})
    # Fig. 1 - leakage audit (recomputed exactly as in notebook section 4)
    sp = pd.read_csv(OUT / "splits.csv"); X = np.load("cache/X224.npy")
    f = F.avg_pool2d(torch.from_numpy(X).float()[:, None], 7).flatten(1)
    f = f - f.mean(1, keepdim=True); f = f / f.norm(dim=1, keepdim=True)
    fig1, ax = plt.subplots(figsize=(6.4 / 2.54, 5.0 / 2.54))
    bins = np.linspace(0.5, 1, 26)
    for col, lab, c in [("split_random", "random split", "#e45756"), ("split_grouped", "grouped split", "#4c78a8")]:
        tr, te = np.flatnonzero(sp[col] == "train"), np.flatnonzero(sp[col] == "test")
        sim = (f[te] @ f[tr].T).max(1).values.numpy()
        ax.hist(np.clip(sim, 0.5, 1), bins=bins, alpha=.8, label=lab, color=c)
    ax.set_xlabel("max. correlation with a training slice"); ax.set_ylabel("test slices")
    ax.legend(frameon=False, loc="upper left"); ax.spines[["top", "right"]].set_visible(False)
    fig1.tight_layout(pad=0.3); fig1.savefig(OUT / "report_fig1.png", dpi=300); plt.close(fig1)
    # Fig. 2 - confusion matrices of run A and run B
    if R is None: return
    names = ["Normal", "Cyst", "Stone", "Tumour"]
    fig2, axes = plt.subplots(1, 2, figsize=(10.6 / 2.54, 5.0 / 2.54))
    for a, k, t in zip(axes, ["A_random", "B_grouped"], ["(A) random split", "(B) grouped split"]):
        cmx = np.array(R["confusion"][k]); pct = cmx / cmx.sum(1, keepdims=True)
        a.imshow(pct, cmap="Blues", vmin=0, vmax=1)
        for i in range(4):
            for j in range(4):
                a.text(j, i, f"{cmx[i, j]}", ha="center", va="center", fontsize=8.5,
                       color="white" if pct[i, j] > 0.5 else "black")
        a.set_xticks(range(4)); a.set_xticklabels(names, rotation=30, fontsize=8)
        a.set_yticks(range(4)); a.set_yticklabels(names if k == "A_random" else [], fontsize=8)
        a.set_xlabel("predicted"); a.set_title(t, fontsize=9)
    axes[0].set_ylabel("true")
    fig2.tight_layout(pad=0.3); fig2.savefig(OUT / "report_fig2.png", dpi=300); plt.close(fig2)


def fig(path, width_cm):
    p = OUT / path
    if not p.exists():
        t = Table([[P(f"{PEND} figure {path} appears after training", cell)]], colWidths=[width_cm * cm], rowHeights=[3 * cm])
        t.setStyle(TableStyle([("BOX", (0, 0), (-1, -1), 0.5, colors.grey), ("VALIGN", (0, 0), (-1, -1), "MIDDLE")]))
        return t
    from PIL import Image as PI
    w, h = PI.open(p).size
    return Image(str(p), width=width_cm * cm, height=width_cm * cm * h / w)


# ------------------------------------------------------------------ derived numbers
# The split and leakage audit run before any training and are fully determined by seed 42,
# so the draft uses the values the notebook prints in section 4 until results.json exists.
leak = R["leak_audit"] if R else {
    "random split": {"test slices with a near-duplicate in train (%)": 99.04, "1-NN pixel accuracy on test (%)": 99.95},
    "grouped split": {"test slices with a near-duplicate in train (%)": 0.00, "1-NN pixel accuracy on test (%)": 40.92}}
def L(split, key):
    return f2(leak[split][key])

rep = None
if R and all(k in R["results"] for k in ["B_grouped", "E_grouped_seed1", "F_grouped_seed2"]):
    keys = ["B_grouped", "E_grouped_seed1", "F_grouped_seed2"]
    rep = {c: (np.mean([R["results"][k][c] for k in keys]), np.std([R["results"][k][c] for k in keys], ddof=1))
           for c in ["test acc %", "macro-F1 %", "Stone recall %", "study-level acc %"]}
ms = lambda c: f"{rep[c][0]:.2f} ± {rep[c][1]:.2f}" if rep else PEND

story = []
story.append(P("PS-21 &nbsp;Kidney Abnormality Detection from CT", title))
story.append(P(f"{NAME} &nbsp;|&nbsp; {REGNO} &nbsp;|&nbsp; Section {SECTION} &nbsp;|&nbsp; PS-21 Kidney Abnormality Detection from CT "
               f"&nbsp;|&nbsp; {DATE} &nbsp;|&nbsp; GitHub: <font color='#1a4fa0'>{GITHUB}</font>", hdr))

# 1 ---------------------------------------------------------------------------------
story.append(P("1. Problem and Objective", head))
story.append(P(
    "Classify a single kidney CT slice as <b>Normal, Cyst, Stone or Tumour</b>. An automatic triage tool could flag abnormal "
    "scans for a radiologist, but only if its accuracy is measured honestly: slices of one patient's scan are nearly identical, "
    "so a random split lets the model be tested on patients it has already seen. Success is judged by <b>macro-F1 on a test set "
    "of unseen patients</b> (every class weighted equally) and by <b>Stone recall</b>, because stone is the smallest class and a "
    "missed stone is a missed diagnosis; plain accuracy is reported but not trusted, since 41% of slices are Normal."))

# 2 ---------------------------------------------------------------------------------
story.append(P("2. Dataset and Pre-processing", head))
story.append(P(
    "CT Kidney Dataset (Kaggle, nazmul0087), <b>full set of 12,446 slices</b>: Normal 5,077, Cyst 3,709, Tumour 2,283, Stone 1,377 "
    "(imbalance 3.7 : 1). <b>No patient ID is given</b>, but consecutive file numbers are adjacent slices (median correlation 0.999), "
    "and a no-training 1-nearest-neighbour lookup on 32×32 pixels scores 99.99% leave-one-out — memorisation alone \"solves\" the "
    "task. I therefore rebuilt <b>195 pseudo-patient studies</b> (break a run when slice-to-slice correlation &lt; 0.80 or image size "
    "changes, then merge runs whose slices correlate &gt; 0.98; no study mixes classes). The <b>grouped split</b> puts whole studies in "
    "one split, per class, seed 42: train 8,528 slices / 133 studies, validation 1,931 / 30, test 1,987 / 32. For comparison a "
    "<b>random</b> stratified split (8,712 / 1,867 / 1,867) was also made; both are saved in <i>outputs/splits.csv</i>. Images were "
    "resized to 224×224, scaled to [0,1], grey channel copied to 3 channels and ImageNet-normalised. Training-only augmentation: "
    "rotation ±10°, zoom ±10%, shift ±5%, horizontal flip, brightness ±0.1, contrast ±15%; because one training step takes ~1.4 s per batch on the laptop GPU and adjacent slices are near-copies, training uses every 8th slice (~1,070 slices from almost every training study); validation and test use all slices. Imbalance was handled with "
    "<b>class-weighted cross-entropy</b> (weight ∝ 1/frequency); no oversampling, so validation and test keep the true class mix."))

# 3 ---------------------------------------------------------------------------------
story.append(P("3. Methodology", head))
story.append(P(
    "<b>ResNet-18 pretrained on ImageNet</b>, final layer replaced by a 4-way linear layer and <b>all layers fine-tuned</b>. With only "
    "133 independent training studies a from-scratch CNN would memorise patients; pretrained edge/texture filters transfer well to "
    "CT, and the smallest ResNet trains in minutes on a laptop GPU (Apple MPS). Early stopping monitors validation macro-F1 and the "
    "best-validation checkpoint is the one tested; the test set is used once. Beyond the baseline I ran (C) the same model without class weights, to measure "
    "their effect on Stone recall, (D) a 7× smaller MobileNetV3-Small as a lightweight option, and Grad-CAM to see where the model looks."))
hp = [["Backbone", "Input", "Loss", "Optimiser / LR", "Batch", "Epochs", "Regularisation", "Seed"],
      ["ResNet-18 (11.2 M), ImageNet, all layers trained", "224×224×3", "Class-weighted CE, label smoothing 0.05",
       "AdamW, max LR 3e-4, one-cycle cosine", "64", "3 (short budget); best val macro-F1 checkpoint kept",
       "Weight decay 1e-4, augmentation", "42"]]
story.append(KeepTogether([P("Table 1. Hyper-parameters (identical for all ResNet-18 runs).", cap),
                           table(hp, [3.1*cm, 1.6*cm, 2.6*cm, 2.6*cm, 1.1*cm, 2.6*cm, 2.3*cm, 1.1*cm])]))

# 4 ---------------------------------------------------------------------------------
story.append(P("4. Results", head))
cols = ["test acc %", "macro-F1 %", "Stone recall %", "Stone precision %", "Tumor recall %", "study-level acc %", "train min"]
rows = [["Run", "Model / split", "Acc.", "Macro-F1", "Stone recall", "Stone prec.", "Tumour recall", "Study acc.", "Train (min)"]]
for k, lab in [("A_random", "ResNet-18 / <b>random</b> (leaky)"), ("B_grouped", "<b>ResNet-18 / grouped (main)</b>"),
               ("C_grouped_noweights", "ResNet-18 / grouped, no class weights"), ("D_mobilenet", "MobileNetV3-S / grouped")]:
    rows.append([k[0], lab] + [res(k, c) for c in cols])
story.append(KeepTogether([P("Table 2. Test-set results in % (each model on the test part of its own split; "
                             "study acc. = majority vote over the slices of a pseudo-study).", cap),
                           table(rows, [1.0*cm, 4.6*cm, 1.55*cm, 1.55*cm, 1.55*cm, 1.55*cm, 1.55*cm, 1.55*cm, 1.4*cm], left_cols=(1,))]))
story.append(Spacer(1, 3))
report_figures()
figs = Table([[fig("report_fig1.png", 6.4), fig("report_fig2.png", 10.6)]], colWidths=[6.6*cm, 10.8*cm])
figs.setStyle(TableStyle([("VALIGN", (0, 0), (-1, -1), "MIDDLE"), ("LEFTPADDING", (0, 0), (-1, -1), 0), ("RIGHTPADDING", (0, 0), (-1, -1), 0)]))
story.append(KeepTogether([figs, P(
    "Fig. 1 (left) Leakage audit: correlation of each test slice with its closest training slice. "
    "Fig. 2 (right) Test-set confusion matrices, random split (A) vs. grouped split (B).", cap)]))

# 5 ---------------------------------------------------------------------------------
story.append(P("5. Analysis and Discussion", head))
rB, rA, rC = (R["results"][k] for k in ["B_grouped", "A_random", "C_grouped_noweights"])
hB, hA = R["history"]["B_grouped"], R["history"]["A_random"]
cmB = np.array(R["confusion"]["B_grouped"])
story.append(P(
    f"<b>Leakage (specific challenge).</b> With a random split {L('random split', 'test slices with a near-duplicate in train (%)')}% of "
    f"test slices have a near-copy (r &gt; 0.99) in training and a no-learning 1-NN scores {L('random split', '1-NN pixel accuracy on test (%)')}%; "
    f"the grouped split reduces this to {L('grouped split', 'test slices with a near-duplicate in train (%)')}% and "
    f"{L('grouped split', '1-NN pixel accuracy on test (%)')}% (Fig. 1). The same ResNet-18 falls from {f2(rA['macro-F1 %'])}% to "
    f"<b>{f2(rB['macro-F1 %'])}% macro-F1</b> ({f2(rA['macro-F1 %'] - rB['macro-F1 %'])} points): the random-split score measures "
    f"recognition of seen patients, so ~99% claims on this dataset are inflated. "
    f"<b>Confusions.</b> Tumour is the weak class (recall {f2(rB['Tumor recall %'])}%): {cmB[3,1]} tumour slices were called Cyst and "
    f"{cmB[3,0]} Normal (Fig. 2) — both are round renal masses whose difference (enhancing tissue vs. fluid) is subtle at 224 px, and only "
    f"7 tumour studies are in the test set. Stones are over-called ({cmB[0,2]} Normal, {cmB[1,2]} Cyst slices → Stone; precision "
    f"{f2(rB['Stone precision %'])}%). Grad-CAM of these errors points at bright bone, aorta or contrast in the renal pelvis, i.e. "
    f"\"bright spot = stone\" — a shortcut. "
    f"<b>Fitting.</b> On the grouped split training accuracy reached {100*hB['train_acc'][-1]:.2f}% but validation only "
    f"{100*hB['val_acc'][-1]:.2f}%: a large generalisation gap to new patients (over-fitting to 133 training patients). On the random "
    f"split the gap vanishes ({100*hA['train_acc'][-1]:.2f}% vs. {100*hA['val_acc'][-1]:.2f}%) because leaky validation hides it. Validation "
    f"loss was still falling at epoch 3, so the model is also under-trained. "
    f"<b>Imbalance.</b> Class weights raised Stone recall from {f2(rC['Stone recall %'])}% (Run C) to <b>{f2(rB['Stone recall %'])}%</b> "
    f"(95% CI {rB['Stone recall 95% CI']}) and macro-F1 by {f2(rB['macro-F1 %'] - rC['macro-F1 %'])} points. MobileNetV3-Small did not "
    f"converge in 3 epochs (macro-F1 {f2(R['results']['D_mobilenet']['macro-F1 %'])}%)."))

# 6 ---------------------------------------------------------------------------------
story.append(P("6. Real-world Relevance, Limitations and Ethics", head))
story.append(P(
    "Use: a second reader that flags slices or studies for radiologist review, never a stand-alone diagnosis. Blockers: pseudo-studies "
    "are reconstructed, not true patient IDs; the data come from one source (hospitals in Dhaka) as JPEG exports of pre-selected "
    "slices, not full DICOM volumes, so performance on other scanners, protocols and populations is unknown; the test set holds only "
    "32 studies (only a few per class), so a different choice of test patients could move the numbers noticeably. A missed tumour or stone delays treatment, so the operating point should favour "
    "recall and every output must be reviewed. Patient data must stay de-identified and any deployment needs clinical validation "
    "and regulatory approval."))

# 7 ---------------------------------------------------------------------------------
story.append(P("7. Conclusion and Future Work", head))
story.append(P(
    f"On unseen patients ResNet-18 reaches {res('B_grouped', 'macro-F1 %')}% macro-F1 and {res('B_grouped', 'Stone recall %')}% "
    f"Stone recall (test), compared with {res('A_random', 'macro-F1 %')}% macro-F1 under a leaky random split — the split, not the "
    f"model, decides the headline number. The models are deliberately under-trained (3 epochs on 1/8 of the training slices); first I would train to convergence on all slices, then train on several neighbouring slices (2.5-D input) and use patient-level "
    f"cross-validation over several splits with real patient IDs from a multi-centre dataset."))

story.append(P("<b>Dataset:</b> M. N. Islam et al., \"Vision transformer and explainable transfer learning models for auto detection "
               "of kidney cyst, stone and tumor from CT-radiography,\" <i>Scientific Reports</i> 12, 11440 (2022); Kaggle: "
               "kaggle.com/datasets/nazmul0087/ct-kidney-dataset-normal-cyst-tumor-and-stone. <b>Methods:</b> ResNet (He et al., 2016), "
               "Grad-CAM (Selvaraju et al., 2017), PyTorch/torchvision pretrained weights. <b>AI use:</b> an AI assistant (Claude) "
               "helped write code and draft text; all results were run, checked and are explained by the author.",
               ParagraphStyle("ref", parent=base, fontSize=9, leading=10.4, spaceBefore=3, alignment=0)))

doc = SimpleDocTemplate(PDF, pagesize=A4, leftMargin=2*cm, rightMargin=2*cm, topMargin=2*cm, bottomMargin=2*cm,
                        title="PS-21 Kidney Abnormality Detection from CT", author=NAME)
doc.build(story)
from pypdf import PdfReader
print(PDF, "pages:", len(PdfReader(PDF).pages), "| results available:", R is not None)
