---
base_model: google/siglip2-base-patch16-224
library_name: peft
tags:
  - lora
  - siglip2
  - cross-lingual
  - urdu
  - vision-language
  - ciklift
language:
  - en
  - ur
license: mit
---

# CIKLIFT — Best Generalized LoRA Adapter

This is the **best-performing LoRA adapter** from the CIKLIFT framework, fine-tuned on the SigLIP2 (ViT-B/16) text encoder for cross-lingual Urdu–English vision-language retrieval.

## Model Details

- **Developed by:** Rana Hasan Mehmood, Maryam Bashir
- **Institution:** FAST School of Computing, National University of Computer and Emerging Sciences, Lahore, Pakistan
- **Model type:** LoRA adapter for SigLIP2 text encoder (SiglipTextTransformer)
- **Language(s):** English (en), Urdu (ur)
- **License:** MIT
- **Fine-tuned from:** [google/siglip2-base-patch16-224](https://huggingface.co/google/siglip2-base-patch16-224)

### Model Sources

- **Repository:** [https://github.com/ranahasany2k/CIKLIFT](https://github.com/ranahasany2k/CIKLIFT)
- **Paper:** *CIKLIFT: Contrastive and Integrated KL-Driven Interlingual Fine-Tuning for Cross-Lingual Vision-Language Models* (under review at Expert Systems With Applications)

## Uses

### Direct Use

This adapter enables **zero-shot cross-lingual image–text retrieval** using Urdu queries on a frozen SigLIP2 vision encoder. It can be used for:

- **Image search with Urdu text queries**
- **Urdu caption ranking / retrieval**
- **Cross-lingual multimodal understanding**

### How to Load

```python
from transformers import AutoModel
from peft import PeftModel

base_model = AutoModel.from_pretrained("google/siglip2-base-patch16-224")
text_encoder = base_model.text_model
adapted = PeftModel.from_pretrained(text_encoder, "./best_generalized")
adapted.eval()
```

## Training Details

### Training Data

- **Stage 1:** Mendeley Parallel English–Urdu Corpus v2 (500K pairs, 10 epochs)
- **Stage 2:** Agreement-filtered Mendeley corpus (200K pairs, 5 epochs)
- **Stage 3:** Translated Conceptual Captions (50K pairs, 10 epochs)

### Training Procedure

Three-stage curriculum-based fine-tuning with combined contrastive + KL-divergence loss:

$$\mathcal{L} = 0.4 \cdot \mathcal{L}_{CE} + 0.6 \cdot \mathcal{L}_{KL}$$

### Training Hyperparameters

- **Training regime:** fp32
- **LoRA rank:** 64
- **LoRA alpha:** 8
- **LoRA dropout:** 0.2
- **Target modules:** q_proj, v_proj, k_proj, o_proj, query, key, value, proj, dense
- **Batch size:** 128
- **Learning rate:** 1e-4
- **Optimizer:** AdamW (weight decay 0.01)
- **Hardware:** Single NVIDIA RTX 3060 (12 GB VRAM)

## Evaluation

### Metrics

Bidirectional retrieval: Recall@1, Recall@5, Recall@10

### Results (Flickr1k-Urdu, Image→Text)

| Configuration | Urdu R@1 | Urdu R@5 | Urdu R@10 |
| :--- | :---: | :---: | :---: |
| SigLIP2 Zero-shot | 10.2% | 22.2% | 30.7% |
| **CIKLIFT (This adapter)** | **57.0%** | **85.4%** | **91.7%** |

### English Retention (Flickr1k-English, Image→Text)

| Configuration | R@1 | R@5 | R@10 |
| :--- | :---: | :---: | :---: |
| SigLIP2 Zero-shot | 92.60% | 99.00% | 99.80% |
| **CIKLIFT (This adapter)** | **74.90%** | **93.40%** | **96.60%** |

## Environmental Impact

- **Hardware Type:** NVIDIA RTX 3060 (12 GB)
- **Training Time:** ~8 hours (all 3 stages)
- **Cloud Provider:** Local (consumer hardware)

## Citation

```bibtex
@article{mehmood2026ciklift,
  title={CIKLIFT: Contrastive and Integrated KL-Driven Interlingual Fine-Tuning for Cross-Lingual Vision-Language Models},
  author={Mehmood, Rana Hasan and Bashir, Maryam},
  journal={Expert Systems With Applications (under review)},
  year={2026}
}
```

### Framework Versions

- PEFT 0.16.0
- Transformers 4.35.0
- PyTorch 2.0