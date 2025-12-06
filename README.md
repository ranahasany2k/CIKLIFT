# **CIKLIFT: Contrastive & Integrated KL–Driven Interlingual Fine-Tuning**

CIKLIFT is a lightweight and efficient method designed to adapt **CLIP/SigLIP-style Vision-Language Models** to **Urdu** through a simple and scalable **text-only fine-tuning pipeline**. The vision encoder remains frozen throughout the process, allowing the entire framework to run efficiently even on a single consumer GPU.

This repository contains notebooks, model weights, and evaluation resources used to reproduce the experiments and results presented in the thesis.

---

## 🌟 **Key Highlights**

- **Text-only adaptation** of the model's text encoder  
- **Preserves English capabilities** while enhancing Urdu understanding  
- **Parameter-efficient** using LoRA  
- **Two-stage adaptation** with large-scale parallel Urdu–English text and Urdu captions  
- **Lightweight training requirements** (12GB GPU compatible)  
- **High-quality notebooks** for training, testing, and visualization  
- **No need to retrain or modify the vision encoder**

---

## 📂 **Repository Structure**
```
CIKLIFT/
│
├── all_notebooks/          # All training, testing, visualization, PCA/UMAP notebooks
├── all_weights/            # All checkpoints, LoRA adapters, best-performing model weights
│
├── src/
│   ├── datasets/
│   ├── models/
│   ├── training/
│   └── evaluation/
│
├── configs/                # Training configs for text-only fine-tuning
│
├── requirements.txt
└── README.md
```

---

## 📥 **Dataset Access**

The datasets used in this project can be downloaded from the following link:

**🔗 Google Drive (Dataset Link):**  
_Add your link here_

You may place the datasets anywhere and adjust the notebook paths accordingly.

---

## 🏛 **High-Level Architecture**

![High-Level Architecture](assets/high_level_architecture.png)

This diagram explains the two-stage text-only adaptation process and how the model aligns Urdu embeddings with the original English-visual embedding space.

---

## 📊 **Embedding Space Visualizations**

### PCA Projection (Final Model)
![PCA](assets/pca_final_model.png)

### UMAP Projection (Final Model)
![UMAP](assets/umap_final_model.png)

These illustrate how the Urdu embedding space becomes geometrically aligned with the English encoder after the CIKLIFT adaptation.

---

## 📈 **Performance Summary**

| Model              | Urdu Recall@1 | English Recall@1 | Notes                                          |
| ------------------ | ------------- | ---------------- | ---------------------------------------------- |
| CLIP ViT-B/16      | 0.5%          | 82.4%            | Fails on Urdu                                  |
| SigLIP2            | 10.2%         | 84.1%            | Best zero-shot baseline                        |
| Full Fine-Tuning   | 33.2%         | 29.1%            | English collapses                              |
| LoRA (standard)    | 22.3%         | 46.3%            | Balanced but low Urdu                          |
| **CIKLIFT (Ours)** | **57.0%**     | **74.5%**        | Best Urdu performance while preserving English |

---

## 🧪 **Notebooks Included**

The `all_notebooks/` folder contains fully documented notebooks for:

* text-only alignment
* Retrieval testing
* PCA and UMAP visualization
* Model comparison tools

These can be executed directly in Colab, Kaggle, or a local Python 3.10 environment.

---

## 💾 **Model Weights**

All trained adapters and final weights are stored in the `all_weights/` folder, including:

* LoRA adapters
* Final adapted text encoder
* Best checkpoint for inference

Organized by experiment and training stage.

---

## 📦 **Requirements**

This project uses:
```
Python 3.10
```

Install dependencies:
```bash
pip install -r requirements.txt
```

---

## 🤝 **Contributing**

You are welcome to open issues or contribute improvements.

---

---

**Made with ❤️ for advancing Urdu NLP research**
