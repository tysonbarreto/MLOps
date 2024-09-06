import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from dataclasses import dataclass, field
from pathlib import Path
import joblib

from bodyfat import load_config

def create_new_features(df:pd.DataFrame) -> pd.DataFrame:
    df['BMI'] = df['weight'] / (df['height'] ** 2)
    df['BAI'] = (df['hip'] / (df['height'] ** 1.5))-18
    df['WHR'] = df['abdomen'] / df['hip']
    return df

@dataclass
class FeatureBuilder:
    train_data: Path
    test_data: Path
    features_dir: Path
    models_dir:Path
    config: dict = field(default_factory=lambda:load_config())
    
    def __post_init__(self):
        self.numerical_features =self.config['features']['numerical']
        self.categorical_features = self.config['features']['categorical']
        self.preprocessor=ColumnTransformer(
            transformers=[
                (
                    "num",StandardScaler(), self.numerical_features
                ),
                (
                    "cat", OneHotEncoder(handle_unknown="ignore"),self.categorical_features
                )
            ]
        )

    def build(self):
        features_train, features_test = self._preprocess_data()
        self._save_features_and_model(
            features_train, features_test, self.features_dir, self.models_dir
        )
    
    def _read_and_create_features(self, data_path:Path)-> pd.DataFrame:
        df = pd.read_parquet(data_path)
        df = create_new_features(df)
        return df[self.numerical_features+self.categorical_features]
    
    def _preprocess_data(self):
        train_df = self._read_and_create_features(self.train_data)
        test_df = self._read_and_create_features(self.test_data)
        features_train = self.preprocessor.fit_transform(train_df)
        features_test = self.preprocessor.fit_transform(test_df)
        return features_train, features_test
    
    def _get_feature_names(self):
        categorical_features_transformed = self.preprocessor.named_transformers_[
            "cat"
        ].get_feature_names_out(self.categorical_features)
        
        return np.concatenate(
            [
                self.numerical_features,categorical_features_transformed
            ]
        )
        
    def _save_features_and_model(self, features_train, features_test,features_dir:Path, models_dir:Path):
        all_features_names = self._get_feature_names()
        features_dir.mkdir(parents=True, exist_ok=True)
        models_dir.mkdir(parents=True, exist_ok=True)
        
        for name, features in [("train", features_train), ("test", features_test)]:
            pd.DataFrame(features, columns=all_features_names).to_parquet(features_dir / f"features_{name}.parquet")
            
            joblib.dump(self.preprocessor, models_dir / "preprocessor.joblib")
        
        
        
    