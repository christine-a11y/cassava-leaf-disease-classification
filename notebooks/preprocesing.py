from google.colab import drive
drive.mount('/content/drive')

import os
import pandas as pd
from sklearn.model_selection import train_test_split



def prepare_and_save_splits(
    csv_path: str,
    save_dir: str = '/content/drive/MyDrive/Cassava_Project',
    test_size: float = 0.2,
    random_state: int = 42
):
    """
    Կատարում է Stratified Train/Val split, ստուգում է Data Leakage-ը
    և պահպանում CSV ֆայլերը Drive-ում:
    """
    df = pd.read_csv(csv_path)

    # Stratified Split (պահպանում է դասերի համամասնությունը)
    train_df, val_df = train_test_split(
        df,
        test_size=test_size,
        random_state=random_state,
        stratify=df['label']
    )

    # Save in Drive
    os.makedirs(save_dir, exist_ok=True)
    train_split_path = os.path.join(save_dir, 'train_split.csv')
    val_split_path = os.path.join(save_dir, 'val_split.csv')

    train_df.to_csv(train_split_path, index=False)
    val_df.to_csv(val_split_path, index=False)

    print(f" Train և Validation CSV-ները պահպանվեցին: {save_dir}")

    # Checking sets size
    print(f"Training set size: {len(train_df)} images ({len(train_df)/len(df)*100:.1f}%)")
    print(f"Validation set size: {len(val_df)} images ({len(val_df)/len(df)*100:.1f}%)")

    # Compare Class Distribution
    train_dist = train_df['label'].value_counts(normalize=True).sort_index() * 100
    val_dist = val_df['label'].value_counts(normalize=True).sort_index() * 100

    dist_df = pd.DataFrame({
        'Train (%)': train_dist,
        'Validation (%)': val_dist
    })
    print("\n--- Class Distribution Comparison ---")
    print(dist_df.round(2))

    # Data Leakage Check
    intersection = set(train_df['image_id']).intersection(set(val_df['image_id']))
    print(f"\nOverlap between Train and Validation: {len(intersection)} images")
    assert len(intersection) == 0, "ERROR: Data leakage detected!"

    return train_df, val_df


if __name__ == "__main__":
    # Ֆայլը ուղղակի աշխատեցնելու (script mode) համար
    CSV_PATH = "/content/drive/MyDrive/Cassava_Project/cassava_data/train.csv"
    SAVE_DIR = "/content/drive/MyDrive/Cassava_Project"
    prepare_and_save_splits(csv_path=CSV_PATH, save_dir=SAVE_DIR)

