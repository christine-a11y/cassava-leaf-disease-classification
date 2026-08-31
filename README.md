# Cassava Leaf Disease Classification
Deep learning pipeline for classifying cassava leaf images into 5 categories, built and trained in Google Colab. The project compares a CNN trained from scratch against three transfer-learning backbones (ResNet50, EfficientNetV2-S, DINOv2 ViT-Base) and picks the best performer by validation metrics.


## Project Overview
This project aims to classify cassava leaf images into five classes:

- CBB — Cassava Bacterial Blight
- CBSD — Cassava Brown Streak Disease
- CGM — Cassava Green Mottle
- CMD — Cassava Mosaic Disease
- Healthy

## Project Goal

Build and evaluate deep learning models for automatic cassava leaf disease classification using image data.

## Team

- [Christine Mkhitaryan](https://github.com/christine-a11y) — Data & Preprocessing 
- [Suzanna Makaryan] — Baseline Learning
- [Ruzanna Torosyan] — Transfer Learning & Model Improvement
- [Victoria Margaryan](https://github.com/viktoryamargaryan) — Evaluation & Explainability

  
## Dataset

- **Source:** [Cassava Leaf Disease Classification](https://www.kaggle.com/competitions/cassava-leaf-disease-classification) (Kaggle competition)
- **Size:** ~21,000 labeled leaf images
- **Classes:**
  | Label | Disease |
  |---|---|
  | 0 | Cassava Bacterial Blight (CBB) |
  | 1 | Cassava Brown Streak Disease (CBSD) |
  | 2 | Cassava Green Mottle (CGM) |
  | 3 | Cassava Mosaic Disease (CMD) |
  | 4 | Healthy |
- **Split:** Stratified 80/20 train/validation split (`train_test_split`, `random_state=42`), saved to `train_split.csv` / `val_split.csv`. A leakage check confirms zero overlap between train and validation image IDs.
- Class distribution is imbalanced (CMD dominates); `compute_class_weight` is used to weight the loss for several models.

## Project Structure

```
Cassava_Project/
├── cassava_data/              # Kaggle dataset (train.csv, train_images/, label map)
├── checkpoints/               # Model weights (.pth) for baseline, ResNet50, EffNetV2, DINOv2
├── history/                   # Training metrics (.json) and curve plots (.png)
├── plots/                     # EDA charts, confusion matrices, and misclassification plots
├── ֆայլեր/                     # Modular Python scripts (dataset.py, model.py, train.py, utils.py, etc.)
└── results/                   #Evaluation outputs and model_comparison.csv
``` 


## Setup

Runs in Google Colab with a GPU runtime.

1. Mount Google Drive:
   ```python
   from google.colab import drive
   drive.mount('/content/drive')
   ```
2. Download the dataset via the Kaggle API (requires `KAGGLE_USERNAME` / `KAGGLE_KEY`):
   ```bash
   kaggle competitions download -c cassava-leaf-disease-classification
   unzip -q cassava-leaf-disease-classification.zip -d cassava_data
   ```
3. Install dependencies:
   ```bash
   pip install torch torchvision pandas scikit-learn opencv-python pillow matplotlib seaborn tqdm
   ```
4. **Copy images to local disk before training** — reading directly from Drive is much slower than local `/content/` storage:
   ```python
   import shutil
   shutil.copytree(
       "/content/drive/MyDrive/Cassava_Project/cassava_data/train_images",
       "/content/images"
   )
   ```


## Exploratory Data Analysis

- Class distribution / imbalance ratio (bar chart, saved to `plots/`)
- Sample images per class
- Image dimension and channel statistics (width/height/format consistency check)
- Data integrity checks: missing labels, invalid labels, missing/corrupted files, duplicate images (MD5 hash), unexpected formats
- Per-class mean color histograms and mean images
- Augmentation preview (`RandomResizedCrop`, flips, rotation, color jitter) to visually sanity-check the pipeline before training

## Models

| Model | Backbone | Input Size | Training Strategy |
|---|---|:---:|---|
| **Baseline CNN** | 3-block custom CNN (from scratch) | 384×384 | Trained end-to-end, 10 epochs |
| **ResNet50** | ImageNet-pretrained | 384×384 | Head-only (5 epochs) → fine-tune `layer4` (5 epochs) |
| **EfficientNetV2-S** | ImageNet-pretrained | 384×384 | Head-only (5 epochs) → fine-tune stages 5+ (8 epochs) |
| **DINOv2 ViT-Base** | Self-supervised ViT (`dinov2_vitb14`) | 224×224 | Full fine-tune (10 epochs) + label-smoothing refinement (3 epochs) |

All transfer models replace the classifier head for 5-class output. Optimizer: AdamW with `CosineAnnealingLR`. Class-weighted `CrossEntropyLoss` is used to counter class imbalance; label smoothing (`0.1`) is used for later fine-tuning passes. DINOv2 training also uses mixed precision (`autocast` + `GradScaler`).


## Training

Each model follows the same checkpoint-aware pattern — if a saved checkpoint exists it's loaded instead of retraining:

```python
if os.path.exists(checkpoint_path):
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()
else:
    history = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=valid_loader,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        epochs=5,
        scheduler=scheduler,
        save_path=checkpoint_path,
        use_amp=True,
        monitor="val_acc"
    )
```

`train_model` saves the best checkpoint by validation accuracy/loss, logs per-epoch train/val loss and accuracy (with a `tqdm` progress bar), and applies gradient clipping (`max_norm=1.0`).

## Evaluation

- `evaluate_model` / `get_metrics` compute accuracy, macro precision, macro recall, and macro F1 on the validation set for each model.
- Results are compiled into a comparison table (`model_comparison.csv`) and bar-chart visualization across all four models.
- The best model by **macro F1** is selected, saved as `final_model.pth` (with model name, class names, and num_classes bundled in), and re-evaluated with a full report: per-class precision/recall/F1, confusion matrix, and misclassified-sample visualizations.
- Error analysis identifies the most commonly confused class pairs and visualizes representative misclassified images for the worst-confused pair.
- A simple attention/importance-map visualization is included for DINOv2, based on patch-embedding norms from the final transformer block.


## Results

| Model | Accuracy | Macro Precision | Macro Recall | Macro F1 |
|---|:---:|:---:|:---:|:---:|
| Baseline CNN | 0.7150| 0.5571 | 0.4425 | 0.4717 |
| ResNet50 (fine-tuned) | 0.8290 | 0.7116 | 0.7606 | 0.7316 |
| EfficientNetV2-S (fine-tuned) | 0.6512 | 0.5021 | 0.5862 | 0.5268 |
| DINOv2 ViT-Base | 0.8600 | 0.7526 | 0.7669 | 0.7582 |


## Requirements

- Python 3.13
- PyTorch + torchvision (CUDA-enabled)
- pandas, numpy, scikit-learn
- opencv-python, Pillow
- matplotlib, seaborn
- tqdm
- kagglehub / Kaggle API credentials


## Notes

- DINOv2 is loaded via `torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14')` — requires internet access from the Colab runtime on first load.
- Training history for each model is saved as JSON (`history/*.json`) with `head_only` and `finetuned` phases separated, for plotting learning curves later.
- Google Drive I/O can be a major bottleneck during training — copying images to local Colab disk (`/content/images`) before building dataloaders is strongly recommended.

## License

Distributed under the MIT License. See [`LICENSE`](LICENSE) for details.

**Contributors:**
* Christine Mkhitaryan
* Ruzanna Torosyan
* Viktoria Margaryan
* Syuzanna Makaryan

