# CIKLIFT: Contrastive & Integrated KL-Driven Interlingual Fine-Tuning

[![Paper](https://img.shields.io/badge/Paper-ESWA_Submission-blue)](https://github.com/ranahasany2k/CIKLIFT)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Python 3.10](https://img.shields.io/badge/Python-3.10-blue.svg)](https://www.python.org/)
[![PyTorch 2.0](https://img.shields.io/badge/PyTorch-2.0-red.svg)](https://pytorch.org/)

**CIKLIFT** is a lightweight, parameter-efficient adaptation framework that bridges the cross-lingual modality gap in Vision-Language Models (VLMs) for **low-resource, non-Latin script languages**. It adapts English-centric models like **SigLIP2** to **Urdu** through a curriculum-based, text-only fine-tuning pipeline — running entirely on a single consumer GPU (12 GB VRAM).

> **Key Result:** CIKLIFT achieves **57.0% Recall@1** on Flickr1k-Urdu (up from 10.2% zero-shot), while retaining **74.9% English Recall@1** — outperforming both full fine-tuning and multilingual baselines like M-CLIP and AltCLIP.

---

## Architecture

![CIKLIFT Architecture](Archtecture.jpg)

The framework consists of three stages:

1. **Stage 1 — Noisy Corpus Pre-Training:** Cross-lingual alignment on 500K English–Urdu parallel sentence pairs (Mendeley corpus).
2. **Stage 2 — Cleaned Corpus Refinement:** Continued training on 200K agreement-filtered high-quality pairs.
3. **Stage 3 — Image-Grounded Fine-Tuning:** Alignment with visual concepts using 50K translated Conceptual Captions.

The **vision encoder remains frozen** throughout. Only the text encoder is adapted via **LoRA** (rank=64, ~300K trainable parameters, 0.75% of the encoder), using a combined loss:

$$\mathcal{L}_{\text{total}} = \lambda_{CE} \cdot \mathcal{L}_{\text{contrastive}} + \lambda_{KL} \cdot \mathcal{L}_{\text{KL}}$$

with $\lambda_{CE} = 0.4$ and $\lambda_{KL} = 0.6$.

---

## Results

### Cross-Lingual Retrieval Performance (Flickr1k-Urdu, Image→Text)

| Model | Backbone | Trainable Params | Urdu R@1 | Urdu R@5 | English R@1 | English Retention |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| CLIP (Zero-shot) | ViT-B/16 | 0 | 0.5% | 1.0% | 82.4% | 100% |
| SigLIP2 (Zero-shot) | ViT-B/16 | 0 | 10.2% | 22.2% | 92.6% | 100% |
| M-CLIP | ViT-B/32 | 0 | 31.6% | 53.3% | — | — |
| AltCLIP | ViT-L/14 | 0 | 5.4% | 13.2% | — | — |
| Full Fine-Tuning | ViT-B/16 | 40M (100%) | 33.2% | 61.3% | 49.1% | 53.0% |
| LoRA + CL Only | ViT-B/16 | 300K (0.75%) | 22.3% | 46.0% | 46.3% | 50.0% |
| **CIKLIFT (Ours)** | **ViT-B/16** | **300K (0.75%)** | **57.0%** | **85.4%** | **74.9%** | **80.9%** |

### English Performance Retention (Flickr1k-English, Image→Text)

| Configuration | R@1 | R@5 | R@10 |
| :--- | :---: | :---: | :---: |
| SigLIP2 (Zero-shot) | 92.60% | 99.00% | 99.80% |
| Full Fine-Tuning | 49.10% | 74.30% | 83.50% |
| LoRA + CL Only | 46.30% | 71.60% | 80.80% |
| **CIKLIFT (Ours)** | **74.90%** | **93.40%** | **96.60%** |

### Data Quality vs. Quantity (Flickr1k-Urdu R@1)

| Training Data | I→T R@1 | T→I R@1 |
| :--- | :---: | :---: |
| Mendeley Only (500K noisy) | 29.6% | 16.1% |
| CC50K Only (50K clean) | 41.5% | 26.5% |
| **Mendeley → CC50K (Curriculum)** | **57.0%** | **38.8%** |

---

## Embedding Space Visualization

### Before Adaptation (SigLIP2 Pre-trained)

| PCA Projection | UMAP Projection |
| :---: | :---: |
| ![PCA Pretrained](ciklift_en_pca.png) | ![UMAP Pretrained](ciklift_en_umap.png) |

### After Adaptation (CIKLIFT Fine-tuned)

| PCA Projection | UMAP Projection |
| :---: | :---: |
| ![PCA CIKLIFT](ciklift_ur_pca.png) | ![UMAP CIKLIFT](ciklift_ur_umap.png) |

After fine-tuning, English and Urdu embeddings overlap substantially, demonstrating successful cross-lingual alignment while preserving coherent relationships with the visual modality.

---

## Repository Structure

```
CIKLIFT/
├── all_notebooks/
│   ├── CiKLift.ipynb              # Training notebook (all 3 stages)
│   └── results.ipynb              # Evaluation, retrieval testing, visualization
├── all_weights/
│   └── best_generalized/
│       ├── adapter_config.json    # LoRA adapter configuration
│       ├── adapter_model.safetensors  # Trained LoRA weights (~14 MB)
│       └── README.md              # Model card
├── Archtecture.jpg                # Architecture diagram
├── ciklift_en_pca.png             # PCA: pre-trained embeddings
├── ciklift_en_umap.png            # UMAP: pre-trained embeddings
├── ciklift_ur_pca.png             # PCA: adapted embeddings
├── ciklift_ur_umap.png            # UMAP: adapted embeddings
└── README.md                      # This file
```

---

## Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/ranahasany2k/CIKLIFT.git
cd CIKLIFT

# Install dependencies
pip install torch torchvision transformers peft accelerate datasets pillow scikit-learn umap-learn matplotlib
```

### Inference with Pre-trained Adapter

```python
import torch
from transformers import AutoModel, AutoTokenizer
from peft import PeftModel

# Load base model
base_model = AutoModel.from_pretrained("google/siglip2-base-patch16-224")
tokenizer = AutoTokenizer.from_pretrained("google/siglip2-base-patch16-224")

# Load CIKLIFT LoRA adapter
text_encoder = base_model.text_model
adapted_encoder = PeftModel.from_pretrained(text_encoder, "./all_weights/best_generalized")
adapted_encoder.eval()

# Encode Urdu text
urdu_text = "ایک بلی گھاس پر بیٹھی ہے"  # A cat is sitting on the grass
inputs = tokenizer(urdu_text, return_tensors="pt", padding=True, truncation=True)

with torch.no_grad():
    urdu_embedding = adapted_encoder(**inputs).last_hidden_state[:, -1, :]  # EOS token
    urdu_embedding = torch.nn.functional.normalize(urdu_embedding, dim=-1)

print(f"Urdu embedding shape: {urdu_embedding.shape}")
```

---

## Datasets

The training and evaluation datasets used in this project are available on Google Drive:

**[📦 Download Datasets](https://drive.google.com/drive/folders/1wadJRUCqYXYSPzoVApBxuxrQxMsRlO3k?usp=sharing)**

| Dataset | Size | Purpose |
| :--- | :---: | :--- |
| Mendeley En-Ur Corpus | 500K pairs | Stage 1 & 2 training |
| Conceptual Captions (Urdu) | 50K pairs | Stage 3 image-grounded training |
| Flickr1k-Urdu | 1K images, 5K captions | Evaluation benchmark |
| COCO-Val-Urdu | 5K images, 25K captions | Evaluation benchmark |

---

## Training Hyperparameters

| Parameter | Value |
| :--- | :--- |
| Base Model | SigLIP2 ViT-B/16 |
| LoRA Rank | 64 |
| LoRA Alpha | 16 |
| LoRA Dropout | 0.2 |
| Target Modules | `q_proj`, `v_proj` |
| Batch Size | 128 |
| Learning Rate | 1e-4 |
| Optimizer | AdamW (weight decay 0.01) |
| LR Scheduler | Cosine Annealing |
| λ_CE / λ_KL | 0.4 / 0.6 |
| τ_CE (contrastive) | 1.0 |
| τ_target, τ_pred (KL) | 0.1 |
| GPU | NVIDIA RTX 3060 (12 GB) |
| Framework | PyTorch 2.0 + HuggingFace Transformers 4.35 |

---

## Citation

If you find this work useful, please cite:

```bibtex
@article{mehmood2026ciklift,
  title={CIKLIFT: Contrastive and Integrated KL-Driven Interlingual Fine-Tuning for Cross-Lingual Vision-Language Models},
  author={Mehmood, Rana Hasan and Bashir, Maryam},
  journal={Expert Systems With Applications (under review)},
  year={2026},
  institution={FAST School of Computing, National University of Computer and Emerging Sciences, Lahore, Pakistan}
}
```

---

## License

This project is released under the [MIT License](LICENSE).

---

## Acknowledgements

- [SigLIP2](https://arxiv.org/abs/2502.14786) by Google DeepMind
- [LoRA](https://arxiv.org/abs/2106.09685) by Hu et al.
- [PEFT](https://github.com/huggingface/peft) by Hugging Face
- [Mendeley Parallel Corpus](https://data.mendeley.com/) for English-Urdu pairs

---

**Made with ❤️ for advancing Urdu NLP and inclusive multimodal AI research**
