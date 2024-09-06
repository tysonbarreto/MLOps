from dataclasses import dataclass
from pathlib import Path
import pandas as pd

@dataclass
class Dataset:
    features: pd.DataFrame
    labels: pd.Series
    
    def write(self, features_path: Path, labels_path: Path):
        self.features.to_parquet(features_path)
        self.labels.to_parquet(labels_path)
        
    @staticmethod
    def read_from_path(features_path: Path, labels_path: Path)-> "Dataset":
        features = pd.read_parquet(features_path)
        labels = pd.read_parquet(labels_path).squeeze()
        return Dataset(features, labels)