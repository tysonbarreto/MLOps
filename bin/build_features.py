from bodyfat import Config,FeatureBuilder


def main():
    data_dir = Config.Path.DATA_DIR / "processed"
    FeatureBuilder(
        train_data= data_dir / "X_train.parquet",
        test_data= data_dir / "X_test.parquet",
        features_dir= Config.Path.FEATURES_DIR,
        models_dir= Config.Path.MODELS_DIR
    ).build()
    
if __name__ == "main":
    main()