import os
import json
import random
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix

# Disease label mapping dictionary from EDA
LABEL_MAP = {
    0: "Cassava Bacterial Blight (CBB)",
    1: "Cassava Brown Streak Disease (CBSD)",
    2: "Cassava Green Mottle (CGM)",
    3: "Cassava Mosaic Disease (CMD)",
    4: "Healthy"
}

def set_seed(seed=42):
    """
    Sets seed for reproducibility across runs.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def calculate_class_weights(df, label_col='label'):
    """
    Calculates class weights for CrossEntropyLoss to handle Class Imbalance.
    """
    class_counts = df[label_col].value_counts().sort_index().values
    total_samples = len(df)
    num_classes = len(class_counts)

    weights = total_samples / (num_classes * class_counts)
    return torch.tensor(weights, dtype=torch.float32)

def calculate_accuracy(outputs, targets):
    """
    Calculates top-1 classification accuracy.
    """
    preds = torch.argmax(outputs, dim=1)
    correct = (preds == targets).sum().item()
    return correct / len(targets)

def visualize_augmentations(image, transform, num_samples=5, save_path=None):
    """
    Applies transformations to an image multiple times and saves the plot.
    """
    fig, axes = plt.subplots(1, num_samples + 1, figsize=(18, 4))

    axes[0].imshow(image)
    axes[0].set_title("Original", fontsize=10, fontweight='bold')
    axes[0].axis('off')

    for i in range(1, num_samples + 1):
        augmented = transform(image)
        if hasattr(augmented, 'permute'):
            augmented = augmented.permute(1, 2, 0).cpu().numpy()

        axes[i].imshow(augmented)
        axes[i].set_title(f"Augmented {i}", fontsize=10)
        axes[i].axis('off')

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Augmentation preview saved to: {save_path}")

    plt.show()
    plt.close()

def save_model_checkpoint(state, save_path):
    """
    Saves trained model checkpoint weights to Google Drive.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(state, save_path)
    print(f"Model checkpoint saved to: {save_path}")

def plot_training_curves(train_losses, val_losses, train_accs, val_accs, title="Learning Curves", save_path=None):
    """
    Universal function to plot training & validation loss and accuracy curves.
    Works for Baseline, ResNet, EfficientNet, ViT, etc.
    """
    epochs = range(1, len(train_losses) + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Loss Plot
    ax1.plot(epochs, train_losses, 'b-o', label='Train Loss')
    ax1.plot(epochs, val_losses, 'r-o', label='Val Loss')
    ax1.set_title(f'{title} - Loss')
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True)

    # Accuracy Plot
    ax2.plot(epochs, train_accs, 'b-o', label='Train Accuracy')
    ax2.plot(epochs, val_accs, 'r-o', label='Val Accuracy')
    ax2.set_title(f'{title} - Accuracy')
    ax2.set_xlabel('Epochs')
    ax2.set_ylabel('Accuracy')
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved successfully to {save_path}")

    plt.show()

def evaluate_model(model, val_loader, device, model_name="Baseline CNN", save_dir="/content/drive/MyDrive/Cassava_Project/plots"):
    """
    Evaluates a model, prints Classification Report, and plots/saves Confusion Matrix.
    """
    class_names = ['CBB', 'CBSD', 'CGM', 'CMD', 'Healthy']
    all_preds = []
    all_labels = []

    model.eval()
    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())

    # Classification Report
    print(f"\n=== {model_name.upper()} CLASSIFICATION REPORT ===")
    print(classification_report(all_labels, all_preds, target_names=class_names, digits=4))

    # Confusion Matrix Plot
    cm = confusion_matrix(all_labels, all_preds)

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title(f'{model_name} - Confusion Matrix')

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        filename = f"{model_name.lower().replace(' ', '_')}_confusion_matrix.png"
        save_path = os.path.join(save_dir, filename)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f" Confusion matrix saved to: {save_path}")

    plt.show()

def verify_transfer_setup(model, model_name, device, expected_classes=5):
    """Verifies output shape and counts trainable vs total parameters of a model."""
    model.eval()
    dummy_input = torch.randn(1, 3, 384, 384).to(device)

    with torch.no_grad():
        output = model(dummy_input)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())

    print(f"=== {model_name} ===")
    print(
        f"Output shape: {output.shape} (պետք է լինի [1, {expected_classes}])"
    )
    print(
        f"Trainable params: {trainable:,} / {total:,}"
        f" ({trainable/total*100:.2f}%)"
    )

    assert output.shape[1] == expected_classes, "❌ Output classes-ը սխալ է!"
    print("✅ Setup-ը ճիշտ է\n")

def load_checkpoint_if_exists(model, checkpoint_path, device):
    """Loads model weights from checkpoint path if file exists."""
    if os.path.exists(checkpoint_path):
        model.load_state_dict(
            torch.load(checkpoint_path, map_location=device)
        )
        print(f"✅ Weight-երը հաջողությամբ բեռնվեցին: {checkpoint_path}")
        return True
    else:
        print(
            f"⚠️ Զգուշացում․ Checkpoint-ը չգտնվեց, ստուգիր հասցեն: {checkpoint_path}"
        )
        return False

def save_history_json(
    history_dict,
    filename,
    save_dir="/content/drive/MyDrive/Cassava_Project/history"):
    """Saves training/validation history dictionary to a JSON file."""
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, filename)

    with open(save_path, "w") as f:
        json.dump(history_dict, f, indent=2)

    print(f"✅ History-ն ապահով պահպանվեց Drive-ում: {save_path}")
    return save_path

def load_history(filepath):
    """Loads training history dictionary from a JSON file if it exists."""
    if os.path.exists(filepath):
        with open(filepath, "r") as f:
            history = json.load(f)
        print(f"✅ History loaded from JSON: {filepath}")
        return history
    return None

def plot_learning_curves_from_history(
    history,
    model_name="Model",
    save_name=None,
    plots_dir="/content/drive/MyDrive/Cassava_Project/plots",
):
    """Plots and saves Train/Val Loss and Accuracy curves for a given history dict."""
    os.makedirs(plots_dir, exist_ok=True)
    epochs = range(1, len(history["train_loss"]) + 1)

    plt.figure(figsize=(14, 5))

    # Loss Plot
    plt.subplot(1, 2, 1)
    plt.plot(epochs, history["train_loss"], label="Train Loss", marker="o")
    plt.plot(epochs, history["val_loss"], label="Val Loss", marker="o")
    plt.title(f"{model_name} - Loss Curve")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)

    # Accuracy Plot
    plt.subplot(1, 2, 2)
    plt.plot(epochs, history["train_acc"], label="Train Accuracy", marker="o")
    plt.plot(epochs, history["val_acc"], label="Val Accuracy", marker="o")
    plt.title(f"{model_name} - Accuracy Curve")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()

    if save_name is None:
        save_name = (
            model_name.lower().replace(" ", "_") + "_learning_curves.png"
        )
    save_path = os.path.join(plots_dir, save_name)
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"✅ Plot-ը հաջողությամբ պահպանվեց Drive-ում: {save_path}")

    plt.show()


def freeze_backbone(model):
    """Freezes all feature extractor parameters, leaving only the head trainable."""
    for param in model.features.parameters():
        param.requires_grad = False
    for param in model.classifier.parameters():
        param.requires_grad = True
    return model

def setup_gradual_unfreezing(model, unfreeze_from_stage=5):
    """
    Explicitly freezes stages < unfreeze_from_stage,
    and unfreezes stages >= unfreeze_from_stage + classifier head.
    """

    for param in model.features.parameters():
        param.requires_grad = False


    for i, child in enumerate(model.features.children()):
        if i >= unfreeze_from_stage:
            for param in child.parameters():
                param.requires_grad = True


    for param in model.classifier.parameters():
        param.requires_grad = True

    return model
