import argparse
import os
import shutil
from tqdm import tqdm
import logging
from src.utils.common import read_yaml, create_directories
import random
import pandas as pd


STAGE = "STAGE_001_LOAD_DATA" ## <<< change stage name

logging.basicConfig(
    filename=os.path.join("logs", 'running_logs.log'), 
    level=logging.INFO, 
    format="[%(asctime)s: %(levelname)s: %(module)s]: %(message)s",
    filemode="a"
    )


def main(config_path, params_path):
    ## read config files
    config = read_yaml(config_path)
    params = read_yaml(params_path)
    train_data = config['source_download_dirs']['TRAIN_DATA']
    test_data = config['source_download_dirs']['TEST_DATA']
    train_data_df = pd.read_csv(train_data)
    test_data_df = pd.read_csv(test_data)
    logging.info("Train and test data loaded successfully")
    artifacts = config['artifacts']
    raw_data = artifacts['INPUT_DATA']
    raw_data_dir = os.path.join(artifacts['ARTIFACTS_DIR'], raw_data)
    create_directories([raw_data_dir])
    train_data_path = os.path.join(raw_data_dir,artifacts['TRAIN_DATA'])
    test_data_path = os.path.join(raw_data_dir,artifacts['TEST_DATA'])
    train_data_df.to_csv(train_data_path)
    test_data_df.to_csv(test_data_path)


if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument("--config", "-c", default="configs/config.yaml")
    args.add_argument("--params", "-p", default="params.yaml")
    parsed_args = args.parse_args()

    try:
        logging.info("\n********************")
        logging.info(f">>>>> stage {STAGE} started <<<<<")
        main(config_path=parsed_args.config, params_path=parsed_args.params)
        logging.info(f">>>>> stage {STAGE} completed!<<<<<\n")
    except Exception as e:
        logging.exception(e)
        raise e