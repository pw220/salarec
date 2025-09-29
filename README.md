
> **Dual-Alignment Contrastive Learning with Preference-Aware Adversarial Augmentation for Sequential Recommendation**

---

## 🔎 Overview

SALARec is a **sequential recommendation framework** that enhances Transformer-based models with:
- **Dual Alignment Contrastive Learning**  
  - *LISA*: Layer-wise Intra-view Self Alignment  
  - *LCA*: Layer-wise Cross-view Alignment
- **Preference-Aware Adversarial Augmentation (PAN)**  
  Adaptive noise injection guided by user preferences to generate semantically meaningful perturbations.

It is designed to address **data sparsity** in real-world recommendation, producing robust and generalizable user/item representations.

---

## ✨ Features
- Transformer encoder with **Gated Feed-Forward Network (GFFN)**
- **PAN augmentation**: gradient-based adversarial perturbations + stochastic noise
- Support for standard benchmarks: Amazon-Beauty, Amazon-Toys, MovieLens-1M
- Evaluation metrics: HR@K, NDCG@K (full-ranking protocol)
- Modular codebase (`salarec/`) for easy extension

---
cd SALARec
pip install -r requirements.txt
