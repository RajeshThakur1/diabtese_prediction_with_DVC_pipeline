import argparse
import os
import shutil
import logging
from src.utils.common import read_yaml, create_directories, save_json
import random
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve, roc_auc_score
STAGE = "standrized data" ## <<< change stage name

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
    prepare_data = artifacts['PREPARED_DATA']
    prepared_data_dir = os.path.join(artifacts['ARTIFACTS_DIR'], prepare_data)
    train_data_path = os.path.join(prepared_data_dir, artifacts['TRAIN_DATA'])
    test_data_path = os.path.join(prepared_data_dir, artifacts['TEST_DATA'])
    train_df = pd.read_csv(train_data_path)
    test_df = pd.read_csv(test_data_path)

    x_train, x_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.25)
    # let's fit the data into kNN model and see how well it performs:
    knn = KNeighborsClassifier()
    knn.fit(x_train, y_train)
    y_pred = knn.predict(x_test)
    scores = knn.score(x_train, y_train)
    print("The accuracy score is : ", accuracy_score(y_test, y_pred))
    score_data = {'acc':scores}
    scores_json_path = config['metrics']['SCORES']
    save_json(scores_json_path, score_data)


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