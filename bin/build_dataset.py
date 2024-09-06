from bodyfat import Config,seed_everything, DatasetBuilder, Dataset

seed_everything()

def main():
    db_path = Config.Path.DATA_DIR / "raw" / "bodyfat.sqlite"
    DatasetBuilder().build(db_path,Config.Path.DATA_DIR/"processed")
    
if __name__ == "__main__":
    main()