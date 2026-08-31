# engine.py
import os
import time
import torch
from torch.cuda.amp import GradScaler, autocast
from tqdm.notebook import tqdm


def train_model(
    model,
    train_loader,
    val_loader,
    criterion,
    optimizer,
    device,
    epochs=10,
    scheduler=None,
    save_path=None,
    use_amp=False,
    monitor="val_loss",  # 'val_loss' կամ 'val_acc'
):
    """Universal Training loop supporting both Standard FP32 and AMP FP16 Mixed Precision.

    Args:
        model: PyTorch model.
        train_loader: DataLoader for training data.
        val_loader: DataLoader for validation data.
        criterion: Loss function.
        optimizer: Optimizer.
        device: torch.device ('cuda' or 'cpu').
        epochs: Number of training epochs.
        scheduler: Optional learning rate scheduler.
        save_path: Path to save the best model checkpoint.
        use_amp: If True, uses Automatic Mixed Precision (FP16) via autocast &
          GradScaler.
        monitor: Metric to monitor for saving checkpoint ('val_loss' or
          'val_acc').
    """
    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}

    # Monitor-ի սկզբնական արժեքի ընտրություն
    best_metric = (
        float("inf") if monitor == "val_loss" else -float("inf")
    )
    scaler = GradScaler() if use_amp else None

    if save_path:
        save_dir = os.path.dirname(save_path)
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)

    mode_str = "FP16 (AMP)" if use_amp else "FP32 (Standard)"
    print(f"🚀 Starting Training [{mode_str}] | Monitoring: {monitor}...")

    for epoch in range(epochs):
        start_time = time.time()

        # --- TRAINING PHASE ---
        model.train()
        running_loss, correct_train, total_train = 0.0, 0, 0

        pbar = tqdm(
            train_loader,
            desc=f"Epoch {epoch+1}/{epochs} [Train]",
            leave=False,
        )
        for inputs, labels in pbar:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()

            if use_amp:
                with autocast():
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)

                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), max_norm=1.0
                )
                scaler.step(optimizer)
                scaler.update()
            else:
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), max_norm=1.0
                )
                optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, 1)
            total_train += labels.size(0)
            correct_train += (predicted == labels).sum().item()

            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        if scheduler is not None:
            scheduler.step()

        epoch_train_loss = running_loss / total_train
        epoch_train_acc = correct_train / total_train

        # --- VALIDATION PHASE ---
        model.eval()
        val_running_loss, correct_val, total_val = 0.0, 0, 0

        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)

                if use_amp:
                    with autocast():
                        outputs = model(inputs)
                        loss = criterion(outputs, labels)
                else:
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)

                val_running_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs, 1)
                total_val += labels.size(0)
                correct_val += (predicted == labels).sum().item()

        epoch_val_loss = val_running_loss / total_val
        epoch_val_acc = correct_val / total_val

        # Save Metrics History
        history["train_loss"].append(epoch_train_loss)
        history["val_loss"].append(epoch_val_loss)
        history["train_acc"].append(epoch_train_acc)
        history["val_acc"].append(epoch_val_acc)

        # Checkpoint Saving Logic
        saved_str = ""
        if save_path:
            is_improvement = (
                (epoch_val_loss < best_metric)
                if monitor == "val_loss"
                else (epoch_val_acc > best_metric)
            )

            if is_improvement:
                best_metric = (
                    epoch_val_loss if monitor == "val_loss" else epoch_val_acc
                )
                torch.save(model.state_dict(), save_path)
                saved_str = f" --> Saved Best Checkpoint ({monitor}: {best_metric:.4f})"

        elapsed_time = time.time() - start_time
        print(
            f"Epoch [{epoch+1}/{epochs}] ({elapsed_time:.1f}s) | "
            f"Train Loss: {epoch_train_loss:.4f} - Train Acc: {epoch_train_acc:.4f} | "
            f"Val Loss: {epoch_val_loss:.4f} - Val Acc: {epoch_val_acc:.4f}{saved_str}"
        )

    return history
