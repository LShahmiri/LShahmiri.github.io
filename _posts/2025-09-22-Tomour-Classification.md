---
layout: post
title: "Brain Tumour Classification – MIGT vs Random Dataset Splitting"
image: "/posts/Brain-Tomour.png"
tags: [DeepLearning, ComputerVision, CNN, Xception, MIGT, MedicalAI, Python, TensorFlow]
---

This project presents a **comparative study between Mutual Information Guided Training (MIGT)** and **random dataset splitting** for **brain tumour classification from MRI images** using a deep convolutional neural network.

---

# Table of Contents
- [00. Project Overview](#overview)
  - [Context](#context)
  - [Actions](#actions)
  - [Results](#results)
  - [Growth / Next Steps](#growth)
- [01. Dataset & Splitting Strategy](#dataset)
- [02. MIGT Methodology](#migt)
- [03. Model Architecture](#model)
- [04. Training Strategy](#training)
- [05. Evaluation & Comparison](#evaluation)
- [06. Discussion](#discussion)

---

# 00. Project Overview <a name="overview"></a>

## Context <a name="context"></a>

Brain tumour classification from MRI scans is a critical medical imaging task where **data bias and improper dataset splitting** can significantly inflate performance metrics.

Traditional random splitting may:
- Overrepresent similar images in training and test sets
- Lead to overly optimistic evaluation
- Hide generalization issues

This project investigates whether **Mutual Information Guided Training (MIGT)** can provide a **more robust and principled data partitioning strategy** compared to random splitting.

---

## Actions <a name="actions"></a>

I implemented two complete pipelines:

1. **Random Split Baseline**
   - Standard shuffled train / validation / test split

2. **MIGT-Based Split**
   - Computes mutual information (MI) between images
   - Sorts samples by MI values
   - Groups images into MI-consistent bins
   - Ensures balanced representation of low-, mid-, and high-MI samples
   - Applies consistent splitting within each bin

Both pipelines were trained using **identical models, hyperparameters, and evaluation protocols** to ensure a fair comparison.

---

## Results <a name="results"></a>

| Method | Test Accuracy | Test AUC |
|------|---------------|----------|
| MIGT | **97.61%** | **0.9939** |
| Random | 97.57% | 0.9936 |

Key observations:
- MIGT achieves **slightly better performance**
- Training curves are **more stable**
- Validation loss shows **less variance**
- Evaluation is more robust to data leakage

---

## Growth / Next Steps <a name="growth"></a>

Future extensions include:
- Applying MIGT to multi-class tumour grading
- Integrating explainability (Grad-CAM, SHAP)
- Testing MIGT on multi-institution MRI datasets
- Combining MIGT with clustering-based sampling
- Clinical validation with radiologist feedback

---

# 01. Dataset & Splitting Strategy <a name="dataset"></a>

- Modality: Brain MRI
- Classes:
  - Brain Tumour
  - Healthy
- Image size: 224 × 224
- Split ratio:
  - 50% Training
  - 10% Validation
  - 40% Test

Two dataset versions were created:
- **Random-split dataset**
- **MIGT-split dataset**

Both contain the same images, differing only in **how samples are allocated**.

---

# 02. MIGT Methodology <a name="migt"></a>

MIGT (Mutual Information Guided Training) works as follows:

1. Select a reference image per class
2. Compute mutual information between the reference and each image
3. Sort images based on MI values
4. Divide images into MI-consistent bins
5. Enforce minimum samples per bin
6. Split each bin into train / validation / test subsets

This ensures:
- Reduced sampling bias
- Balanced feature diversity
- Better generalization assessment

---

# 03. Model Architecture <a name="model"></a>

**Backbone:** Xception (ImageNet pretrained)

**Architecture:**
- Input: 224×224 RGB
- Data augmentation
- Xception feature extractor (frozen)
- Global Average Pooling
- Dropout
- Dense layer (sigmoid output)

**Total parameters:** ~20.8M  
**Trainable parameters:** ~2K (classification head)

---

# 04. Training Strategy <a name="training"></a>

- Optimizer: Adam
- Loss: Binary Cross-Entropy
- Metrics: Accuracy, AUC
- Early stopping on validation loss
- Fine-tuning stage with partial unfreezing
- Same training configuration for both MIGT and Random experiments

This guarantees that **dataset splitting is the only variable**.

---

# 05. Evaluation & Comparison <a name="evaluation"></a>

Evaluation performed on a **held-out test set**:

- Accuracy
- ROC-AUC
- Precision / Recall / F1-score
- Confusion matrix analysis

Both methods perform strongly, but MIGT shows:
- Better stability
- Reduced overfitting risk
- More reliable generalization

---

# 06. Discussion <a name="discussion"></a>

This project demonstrates that:

- High accuracy alone is not sufficient in medical AI
- Dataset splitting strategy directly impacts model validity
- MIGT provides a principled alternative to random sampling
- Mutual information can effectively guide data partitioning
- Robust evaluation is essential for clinical applicability

The approach aligns with **trustworthy and explainable medical AI principles**.

---

### 🔗 GitHub Repository  
👉[GitHub](https://github.com/LShahmiri/Brain-tumour-classification/tree/main)


