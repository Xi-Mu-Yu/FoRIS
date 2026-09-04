<div align="center">

# FoRIS: Progressive Foreground Refinement for Training-Free In-Context Segmentation

[![arXiv](https://img.shields.io/badge/arXiv-2609.03384-b31b1b.svg)](https://arxiv.org/abs/2609.03384)
[![Project Page](https://img.shields.io/badge/Code-GitHub-blue)](https://github.com/Xi-Mu-Yu/FoRIS)

**Ming Hu** · **Jianfu Yin** · **Mingyu Dou** · **Miaomiao Zhang** · **Yao Wang** · **Cong Hu** · **Bingliang Hu** · **Quan Wang**

Xi'an Institute of Optics and Precision Mechanics, CAS · University of Chinese Academy of Sciences · Xi'an Jiaotong University · Zhongnan Hospital of Wuhan University

</div>

---

**FoRIS** is a training-free in-context segmentation framework built on a **single frozen DINOv3** encoder. Instead of treating segmentation as one-step reference–query matching, FoRIS progressively refines coarse foreground responses into precise, complete masks through three stages: **Foreground Purification (FP)**, **Foreground Localization (FL)**, and **Foreground Consolidation (FC)**.

🚀 **Training-free** — no fine-tuning, no segmentation decoder, no auxiliary models (e.g., SAM)  
🔁 **Progressive refinement** — suppress background, localize targets, and consolidate fragmented responses  
📈 **Strong performance** — average gains of **+4.5** and **+4.8** mIoU over prior methods in 1-shot and 5-shot settings  
🌍 **Broad generalization** — object-level, part-level, and cross-domain segmentation (natural, medical, underwater, aerial)

<p align="center">
  <img src="assets/teaser.png" alt="FoRIS teaser: progressive refinement pipeline and benchmark comparison" width="95%">
</p>

*Left: coarse-to-fine progressive refinement (FP → FL → FC). Right: 1-shot mIoU comparison across benchmarks.*

## Method Overview

FoRIS reinterprets in-context segmentation as a **coarse-to-fine foreground refinement** process:

| Stage | Module | Role |
|-------|--------|------|
| **FP** | Adaptive Positional Debiasing (APD) + Two-stage Foreground Refinement (FR) | Remove positional/background interference and produce an initial purified response \(S^{(1)}\) |
| **FL** | Cross-image candidate voting + multi-cue clustering | Localize discriminative target regions and refine to \(S^{(2)}\) |
| **FC** | Semantic Disagreement Penalization (SDP) + Semantic Reweighting (SR) | Suppress conflicting activations and recover coherent structures in \(S^{(3)}\) |

Each stage builds on the previous one; ablations in the paper show consistent mIoU improvements when FP, FL, and FC are added sequentially (see teaser above).

## Environment Setup

Create a Conda environment and install dependencies. Experiments in the paper use **PyTorch 2.7+ with CUDA 12.6**:

```bash
conda create --name foris python=3.10 -y
conda activate foris
pip install -r requirements.txt
```

**Optional — CRF mask refinement** (used in paper evaluation):

```bash
git clone https://github.com/netw0rkf10w/CRF.git
cd CRF
python setup.py install
cd ..
```

## DINOv3 Weights

FoRIS uses a **frozen DINOv3** backbone. Download pretrained weights from the [official DINOv3 repository](https://github.com/facebookresearch/dinov3):

```bash
mkdir -p pretrain
```

Place the desired checkpoint under `pretrain/`:

```
pretrain/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth   # Large (default)
pretrain/dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth   # Base
pretrain/dinov3_vits16_pretrain_lvd1689m-08c60483.pth   # Small
```

By default, FoRIS uses the **Large** model. Update the encoder hub path and weight paths in `models/__init__.py` to match your local setup before running inference.

## Minimal Usage

Segment a target image given one reference image and its mask:

```python
from models import build_foris
from utils.visualization import visualize_prediction

ref_image_path = "assets/ref_cat_image.jpg"
ref_mask_path = "assets/ref_cat_mask.png"
target_image_path = "assets/target_cat_image.jpg"
output_path = "target_cat_pred.png"

# Build model
model = build_foris()

# Set reference and target
model.set_reference(ref_image_path, ref_mask_path)
model.set_target(target_image_path)

# Predict
pred_mask = model.segment()

# Save visualization
visualize_prediction(
    ref_image_path,
    ref_mask_path,
    target_image_path,
    pred_mask,
    output_path,
)
```

For CRF-based mask refinement: `model = build_foris(mask_refiner="crf")`.

You can also run the bundled example:

```bash
python minimal.py
```

## Data Preparation

Dataset setup follows the same protocol as [INSID3](https://github.com/cuttano/INSID3) / [Matcher](https://github.com/aim-uofa/Matcher). See [docs/data.md](docs/data.md) for download and preprocessing instructions.

Supported benchmarks:

| Type | Datasets |
|------|----------|
| Semantic | COCO-20<sup>i</sup>, LVIS-92<sup>i</sup>, ISIC, SUIM, iSAID, Chest X-ray |
| Part | PASCAL-Part, PACO-Part |
| Slender structures | Fundus (see paper Appendix F) |

## Inference

Run evaluation on a prepared dataset (default data root: `data/`):

```bash
# COCO-20^i — 4 folds
python inference.py --dataset coco --exp-name foris-coco --crf-mask-refinement --fold 0
python inference.py --dataset coco --exp-name foris-coco --crf-mask-refinement --fold 1
python inference.py --dataset coco --exp-name foris-coco --crf-mask-refinement --fold 2
python inference.py --dataset coco --exp-name foris-coco --crf-mask-refinement --fold 3

# LVIS-92^i — 10 folds
python inference.py --dataset lvis --exp-name foris-lvis --crf-mask-refinement --fold 0
# ... repeat for folds 1–9

# Part segmentation
python inference.py --dataset pascal_part --exp-name foris-pascal --crf-mask-refinement --fold 0
python inference.py --dataset paco_part --exp-name foris-paco --crf-mask-refinement --fold 0

# Cross-domain semantic segmentation
python inference.py --dataset isic --exp-name foris-isic --crf-mask-refinement
python inference.py --dataset suim --exp-name foris-suim --crf-mask-refinement
python inference.py --dataset isaid --exp-name foris-isaid --crf-mask-refinement --fold 0
python inference.py --dataset lung --data-root data/LungSegmentation --exp-name foris-lung --crf-mask-refinement
python inference.py --dataset permis --exp-name foris-permis --crf-mask-refinement

# Multi-shot (e.g., 5-shot on ISIC)
python inference.py --dataset isic --exp-name foris-isic-5shot --crf-mask-refinement --shots 5
```

Use `--data-root /path/to/data` if datasets are stored elsewhere. Example shell scripts are provided under `scripts/`.

### Main Arguments

| Argument | Description |
|----------|-------------|
| `--dataset` | `coco`, `lvis`, `pascal_part`, `paco_part`, `isaid`, `isic`, `lung`, `suim`, `permis`, `fundus` |
| `--model-size` | DINOv3 size: `small`, `base`, `large` (default: `large`) |
| `--shots` | Number of reference images per episode (default: `1`) |
| `--fold` | Cross-validation fold for COCO, LVIS, Pascal-Part, PACO-Part, iSAID |
| `--crf-mask-refinement` | Enable CRF-based boundary refinement (used in paper) |
| `--svd-comps` | SVD components for positional debiasing (default: `500`) |
| `--tau` | Agglomerative clustering distance threshold (default: `0.6`) |

See `opts.py` for the full list of hyperparameters.

**Note:** Predicted masks are bilinearly upsampled to the original resolution by default. Enable `--crf-mask-refinement` for additional boundary refinement as in the paper.

## Results (1-shot mIoU, %)

FoRIS achieves state-of-the-art results among training-free methods using only frozen DINOv3 features:

| Method | LVIS | COCO | ISIC | SUIM | iSAID | X-Ray | PASCAL | PACO | **Avg** |
|--------|------|------|------|------|-------|-------|--------|------|---------|
| GF-SAM | 35.2 | 58.7 | 48.7 | 53.1 | 47.1 | 51.0 | 44.5 | 36.3 | 46.8 |
| INSID3 | 41.8 | 57.6 | 54.4 | 54.9 | 52.1 | 78.8 | 50.5 | 38.7 | 53.6 |
| **FoRIS** | **42.8** | **60.9** | **62.9** | **59.1** | **53.6** | **87.6** | **55.8** | **42.3** | **58.1** |

## Why FoRIS Works

Existing training-free ICS methods often transfer foreground semantics through direct reference–query correspondence. This can fail when:

1. **Reference features are contaminated** by background or positional cues.
2. **Target localization is ambiguous** due to visually similar distractors.
3. **Foreground responses are fragmented** across the query image.

FoRIS addresses these issues progressively:

- **FP** adaptively debiases positional layout (APD) and suppresses background via contrastive foreground refinement, yielding cleaner semantic prototypes.
- **FL** combines bidirectional patch matching with multi-cue clustering (DINO features + RGB + coordinates) to localize reliable target regions.
- **FC** penalizes semantic disagreement between intermediate maps and reweights clusters to recover complete object structures.

## Project Structure

```
FoRIS/
├── models/
│   └── foris.py          # FoRIS model (FP → FL → FC)
├── datasets/             # Benchmark data loaders
├── utils/                # Clustering, metrics, CRF refinement, visualization
├── scripts/              # Per-dataset evaluation scripts
├── docs/data.md          # Dataset preparation
├── inference.py          # Benchmark evaluation entry point
├── minimal.py            # Minimal single-image demo
└── opts.py               # CLI arguments
```

## Citation

If you find this work useful, please cite:

```bibtex
@article{hu2026foris,
  title   = {{FoRIS}: Progressive Foreground Refinement for Training-Free In-Context Segmentation},
  author  = {Ming Hu and Jianfu Yin and Mingyu Dou and Miaomiao Zhang and Yao Wang and Cong Hu and Bingliang Hu and Quan Wang},
  journal = {arXiv preprint arXiv:2609.03384},
  year    = {2026}
}
```

## Acknowledgements

This project builds upon and adapts ideas from [INSID3](https://github.com/cuttano/INSID3). We also thank the authors of:

- [DINOv3](https://github.com/facebookresearch/dinov3)
- [Matcher](https://github.com/aim-uofa/Matcher)
- [GF-SAM](https://github.com/ANDYZAQ/GF-SAM)
