from bodyfat import Config, Dataset, train_model

def main():
    train_dataset = Dataset.read_from_path(
        Config.Path.FEATURES_DIR / "features_train.parquet",
        Config.Path.DATA_DIR / "processed" / "labels_train.parquet"
    )
    test_dataset = Dataset.read_from_path(
        Config.Path.FEATURES_DIR / "features_test.parquet",
        Config.Path.DATA_DIR / "processed" / "labels_test.parquet"
    )
    
    train_model(train_dataset, test_dataset)
    

if __name__=="main":
    main()