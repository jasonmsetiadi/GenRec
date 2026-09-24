# GenRec Notes

## Encoding Methods

| Type | Models | Description |
|---|---|---|
| **Text** | SentenceTransformers, OpenAI Embedding API | Encode item text metadata |
| **Image** | CLIP ViT | Encode item images |
| **Collaborative Filtering** | SASRec | Learn embeddings from user-item interactions |
| **Vision-Language** | VLM (HuggingFace causal LM) | Fuse text + image jointly |
| **Fusion** | Concat, MLP | Combine embeddings from multiple modalities |

## Quantization Families

| Family | Methods | Mechanism |
|---|---|---|
| **Residual** | RQ-VAE, RQ-VAE (LETTER), RVQ, MM-RQVAE | Learned quantization with residual refinement |
| **Product** | PQ, OPQ | Split embedding into subspaces, quantize each |
| **Vector** | VQ-VAE, MQVAE | Direct codebook lookup (single or multi-codebook) |
| **Clustering** | RKMeans | K-Means clustering applied residually |

## Backbone Families

| Family | Models | Architecture |
|---|---|---|
| **Encoder-Decoder** | TIGER, OneRec | Seq2seq autoregressive |
| **Decoder-only (AR)** | GPT2, LLM, LCRec | Autoregressive left-to-right |
| **Decoder-only (Non-AR)** | RPG | GPT-2 with parallel prediction heads |
| **Discrete Diffusion** | AdaDiff | BERT + iterative denoising |

## Training Paradigms

| Paradigm | Models | Description |
|---|---|---|
| **Standard (from scratch)** | TIGER, OneRec, GPT2, RPG | Train on interaction sequences from random init |
| **SFT (Supervised Fine-Tuning)** | LLM, LCRec | Fine-tune a pretrained LLM on recommendation as instruction-following |
| **Discrete Diffusion** | AdaDiff | Mask-and-denoise with iterative refinement |

## Inference Methods

| Method | Models | Description |
|---|---|---|
| **Beam Search** | TIGER, OneRec, GPT2, LLM, LCRec | Standard beam search decoding |
| **Beam Search + Prefix-Tree** | TIGER, OneRec, LCRec | Constrain beam search to only generate valid SID sequences |
| **Codebook Scoring** | RPG | Score all items via parallel head logits + codebook lookup, then top-k |
| **Iterative Denoising + Diversity Guidance** | AdaDiff | Multi-step unmasking with diversity penalty + prefix-tree constraints |

## Decoder-Only Input & Output Design

| Model | Input Format | Loss On | Embedding Layer | Output Validity |
|---|---|---|---|---|
| **GPT2** | Flat SID code sequence (left-padded) | All tokens (history + target) | Built-in (code vocab only) | By architecture — only SID tokens exist |
| **LLM** | Flat SID code sequence (left-padded) | Target SID only | Replaced with code vocab (pretrained embeddings discarded) | By architecture — NL tokens removed |
| **LCRec** | NL instruction prompt + SID tokens (no candidate items provided — open-ended generation) | Response only | Pretrained kept + SID tokens added | By decoding constraint — prefix-tree forces valid SIDs |

## Key Problems & Solutions

| Problem | Stage | Solution |
|---|---|---|
| **Collision** (same SID for different items) | Quantization | Dedup layer appends a disambiguation code |
| **Invalid ID** (generated SID doesn't exist) | Inference | Prefix-tree constrains decoding to valid sequences |
| **Collaborative signals** (inject interaction patterns) | Encoding, Quantization, Training | See below |

## Collaborative Signal Injection During Training

| Method | Papers | Description |
|---|---|---|
| **Joint training (contrastive + multi-task)** | ColaRec | Jointly train generation + indexing with auxiliary ranking/contrastive regularization tied to collaborative IDs |
| **Alternating training** | ETEGRec | Alternate encoder/decoder training to thread preference signal through generation |
| **Preference alignment (DPO/RL)** | OneRec, Align3GR, GREAM | DPO/RL-style alignment using real user feedback pairs (spans quantization + training) |
| **Knowledge distillation** | — | Use a trained CF model's output as soft target or auxiliary loss |
| **Cross-user augmentation** | — | Augment training sequences with items from similar users' histories |
