from bodyfat import Config, load_config
from bodyfat import Dataset
from xgboost import XGBRegressor
from typing import Tuple
from dvclive import Live
from sklearn.metrics import r2_score, root_mean_squared_error

config = load_config()

def evaluate(model:XGBRegressor, datatset: Dataset)-> Tuple[float, float]:
    predictions = model.predict(datatset.features)
    rmse = root_mean_squared_error(datatset.labels,predictions)
    r2 = r2_score(datatset.labels, predictions)
    
    return rmse, r2

def train_model(train_dataset:Dataset, val_dataset:Dataset):
    with Live(Config.Path.EXPERIMENTS_DIR) as live:
        model = XGBRegressor(
            n_estimators=config['model']['n_estimators'],
            max_depth = config['model']['max_depth']
        )
        
        model.fit(train_dataset.features, train_dataset.labels)
        
        train_rmse, train_r2 = evaluate(model=model, datatset=train_dataset)
        live.log_metric("train/rmse", train_rmse)
        live.log_metric("train/r2", train_r2)
        
        val_rmse, val_r2 = evaluate(model=model, datatset=val_dataset)
        live.log_metric("val/rmse", val_rmse)
        live.log_metric("val/r2", val_r2)
        
        model.save_model(Config.Path.MODELS_DIR/"model.json")