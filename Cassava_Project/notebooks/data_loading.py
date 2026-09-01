import json
import pandas as pd
import os
import kagglehub
# DATASET_PATH="/root/.cache/kagglehub/competitions/cassava-leaf-disease-classification"

def load_data(kaggle_token="KGAT_91f798969f32e3baa632e8967eb06f6c"):
    """
    Downloads Cassava dataset from Kaggle and returns DataFrame,
    mapping dictionary, and dataset path.
    """
    # Set Kaggle API token
    os.environ["KAGGLE_API_TOKEN"] = kaggle_token

    # Download competition dataset
    path = kagglehub.competition_download('cassava-leaf-disease-classification')
    print("Path to competition files:", path)

    # Load train.csv
    df = pd.read_csv(os.path.join(path, "train.csv"))

    # Load disease mapping JSON
    json_path = os.path.join(path, 'label_num_to_disease_map.json')
    with open(json_path, 'r') as f:
        mapping = json.load(f)

    return df, mapping, path

if __name__ == "__main__":
    # Test script execution independently
    df, mapping, dataset_path = load_data()
    print("\nFirst few rows of DataFrame:")
    print(df.head())
    print('\nDisease Mapping:', mapping)


# os.environ["KAGGLE_API_TOKEN"] = "KGAT_91f798969f32e3baa632e8967eb06f6c"

# # Download latest version
# path = kagglehub.competition_download('cassava-leaf-disease-classification')

# print("Path to competition files:", path)

# print("\nՖայլեր պապկայում:")

# print(os.listdir(path))

# df = pd.read_csv(os.path.join(path, "train.csv"))

# print("\nՏվյալների աղյուսակի առաջին տողերը:")

# print(df.head())

# with open(f'{DATASET_PATH}/label_num_to_disease_map.json') as f:
#   mapping = json.load(f)

# print('\nՀիվանդությունների ցանկը:', mapping)
