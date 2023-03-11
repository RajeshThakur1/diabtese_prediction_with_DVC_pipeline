import argparse
import os
import shutil
from tqdm import tqdm
import logging
from src.utils.common import read_yaml, create_directories
import random
import pandas as pd
import numpy as np

STAGE = "STAGE_002_prepare_data" ## <<< change stage name

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
    artifacts = config['artifacts']
    raw_data = artifacts['INPUT_DATA']
    raw_data_dir = os.path.join(artifacts['ARTIFACTS_DIR'],raw_data)
    train_data_path = os.path.join(raw_data_dir,artifacts['TRAIN_DATA'])
    test_data_path = os.path.join(raw_data_dir,artifacts['TEST_DATA'])
    train_df = pd.read_csv(train_data_path)
    test_df = pd.read_csv(test_data_path)
    train_df[['Glucose','BloodPressure','SkinThickness','Insulin','BMI']] = train_df[['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']].replace(0, np.NaN)
    test_df[['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']] = test_df[
        ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']].replace(0, np.NaN)
    train_df['Glucose'].fillna(train_df['Glucose'].mean(), inplace=True)
    test_df['Glucose'].fillna(test_df['Glucose'].mean(), inplace=True)
    train_df['BloodPressure'].fillna(train_df['BloodPressure'].mean(), inplace=True)
    test_df['BloodPressure'].fillna(test_df['BloodPressure'].mean(), inplace=True)
    train_df['SkinThickness'].fillna(train_df['SkinThickness'].median(), inplace=True)
    test_df['SkinThickness'].fillna(test_df['SkinThickness'].median(), inplace=True)
    train_df['Insulin'].fillna(train_df['Insulin'].median(), inplace=True)
    test_df['Insulin'].fillna(test_df['Insulin'].median(), inplace=True)

    train_df['BMI'].fillna(train_df['BMI'].median(), inplace=True)
    test_df['BMI'].fillna(test_df['BMI'].median(), inplace=True)

    prepared_data = artifacts['PREPARED_DATA']
    prepared_data_dir = os.path.join(artifacts['ARTIFACTS_DIR'],prepared_data)
    create_directories([prepared_data_dir])
    train_path = os.path.join(prepared_data_dir, artifacts['TRAIN_DATA'])
    test_path = os.path.join(prepared_data_dir, artifacts['TEST_DATA'])
    train_df.to_csv(train_path)
    test_df.to_csv(test_path)
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