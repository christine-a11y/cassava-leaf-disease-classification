
import os
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader

from augmentations import get_transforms
from utils import calculate_class_weights
import torchvision.transforms as T


#Dataset դասը
class CassavaPyTorchDataset(Dataset):
    def __init__(self, df, img_dir, transform=None):
        self.df = self.df = df.reset_index(drop=True).copy()
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_name = self.df.iat[idx, self.df.columns.get_loc('image_id')]
        label = self.df.iat[idx, self.df.columns.get_loc('label')]
        img_path = os.path.join(self.img_dir, img_name)

        with Image.open(img_path) as img:
            image = img.convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, torch.tensor(label, dtype=torch.long)


# 3. DataLoader-ների պատրաստման ֆունկցիա
def get_dataloaders(
    train_df,
    val_df,
    img_dir="cassava_data/train_images",
    img_size=384,
    batch_size=16,
    num_workers=0,
    pin_memory=False
):
    train_transform, valid_transform = get_transforms(img_size=img_size)

    train_dataset = CassavaPyTorchDataset(
        df=train_df,
        img_dir=img_dir,
        transform=train_transform
    )

    valid_dataset = CassavaPyTorchDataset(
        df=val_df,
        img_dir=img_dir,
        transform=valid_transform
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True
    )

    valid_loader = DataLoader(
        valid_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False
    )

    return train_loader, valid_loader

def get_weighted_criterion(train_df, label_col="label", device="cpu"):
    """
    Computes balanced class weights from training DataFrame
    and returns a CrossEntropyLoss weighted by class imbalance.
    """
    classes = np.array(sorted(train_df[label_col].unique()))
    weights = compute_class_weight(
        class_weight="balanced",
        classes=classes,
        y=train_df[label_col].values,
    )
    class_weights = torch.tensor(weights, dtype=torch.float).to(device)

    print("⚖️ Computed Class Weights:", class_weights)

    criterion_weighted = nn.CrossEntropyLoss(weight=class_weights)
    return criterion_weighted
