"""
Module for applying data preparation, feature extraction, feature selection, and anomaly generation on a time series
dataset. 
"""
# %%
import warnings
import os
import argparse
import yaml
import pickle
from datetime import datetime
import numpy as np
import pandas as pd
from tsfresh.feature_extraction import extract_features
from tsfresh.utilities.dataframe_functions import impute
from tsfresh.feature_extraction import MinimalFCParameters, EfficientFCParameters, ComprehensiveFCParameters
from tsfresh.feature_selection import select_features
from sklearn.model_selection import train_test_split
from imblearn.combine import SMOTETomek
from utils import visualize_dataset, save_data



# %%
def load_data(data_path):
    """Loads the dataset in data_path

    Args:
        data_path (str): data_path (str): path of the raw time series dataset 

    Returns:
        dataset (list): loaded dataset

    Note:
        The dataset is assumed to be a list of dictionaries where each element carries time series measurements, label, 
        time vector(s), and categorical features of a single event.
    """        
    with open(data_path, "rb") as f:
        dataset = pickle.load(f)
    return dataset


# %%
def prepare_data(dataset, time_type, time_tag, ftrs_tag, label_tag):
    """Creates the block dataset needed for feature extraction

    Args:
        dataset (list): time series dataset as a list of dictionaries each one representing one single event
        time_type (str): type of the time vector in the dataset, should be one of the {"fixed", "varying"}
                            - "fixed": all time series of an event share the same time vector
                            - "varying": various time series of an event have different time vectors
        time_tag (None or [str]): key(s) of the time tag in each event
                            - None: if the time vector is not included (only possible when time_type="fixed").
                            - "time_key": the key of the time vector in each event when time_type="fixed"
                            - ["time_key1", "time_key2", ...]: list of time keys at each event when time_type="varying"
        ftrs_tag ([str]): key(s) of the time series measurements in each event
        label_tag (str): key of the label in each event

    Returns:
        X (pd.DataFrame): Block dataset to be passed for feature extraction
        y (pd.Series): Output label vector
    """
    if time_type.lower() == "fixed":
        X = pd.DataFrame()
        y = pd.Series(dtype=np.int64)
        id = 0
        for data_I in dataset:
            X_I = pd.DataFrame()
            X_I["id"] = None  # palceholder for id
            if time_tag is not None:
                X_I["time"] = data_I["time"]
            else:
                X_I["time"] = np.arange(len(data_I[ftrs_tag[0]]))
            for ftr in ftrs_tag:
                X_I[ftr] = data_I[ftr]
            X_I["id"] = id
            X = pd.concat([X, X_I], axis=0)

            y_I = pd.Series(data_I[label_tag], dtype=np.int64)
            y = pd.concat([y, y_I])
            id += 1
        X = X.reset_index(drop=True)
        y = y.reset_index(drop=True)

    elif time_type.lower() == "varying":    
        X = pd.DataFrame()
        y = pd.Series(dtype=np.int64)
        id = 0  # sample block id 
        for data_I in dataset:
            X_I = pd.DataFrame()
            id_ = 0  # time series id in each sample block
            for ftr, t in zip(ftrs_tag, time_tag):
                x_I = pd.DataFrame()
                x_I["id"] = None  # placeholder
                x_I["kind"] = None  # placeholder
                x_I["time"] = data_I[t]
                x_I["value"] = data_I[ftr]
                x_I["id"] = id 
                x_I["kind"] = id_
                X_I = pd.concat([X_I, x_I])
                id_ += 1
            X = pd.concat([X, X_I], axis=0)
            id += 1
            y_I = pd.Series(data_I[label_tag], dtype=np.int64)
            y = pd.concat([y, y_I])
        X = X.reset_index(drop=True)
        y = y.reset_index(drop=True)

    return X, y
    

# %%
def my_extract_features(X, time_type, n_jobs, depth):
    """Extracts features from the block dataset X using tsfresh

    Args:
        X (pd.DataFrame): block dataset
        time_type (str): type of the time vector in the dataset, should be in ["fixed", "varying"]
                            - "fixed": all time series of an event share the same time vector
                            - "varying": various time series of an event have different time vecto
        n_jobs (int): number of CPUs used for feature extraction

    Returns:
        X_t (pd.DataFrame): tabular dataset
    """
    # feature extraction depth to fc_params mapping
    fc_params = dict(
        minimal = MinimalFCParameters(),
        efficient = EfficientFCParameters(),
        comprehensive = ComprehensiveFCParameters()
    )

    if time_type.lower() == "fixed":
        X_t = extract_features(
            X, column_id="id", column_sort="time", n_jobs=n_jobs, 
            impute_function=impute, default_fc_parameters=fc_params[depth.lower()])

    elif time_type.lower() == "varying":    
        X_t = extract_features(
            X, column_id="id", column_kind="kind", column_sort="time", 
            column_value="value", n_jobs=n_jobs, impute_function=impute, 
            default_fc_parameters=fc_params[depth.lower()])
    
    return X_t


# %%
def my_select_features(X_t, y, n_jobs, random_state=42):
    """Select features from the tabular dataset X_t using tsfresh

    Args:
        X_t (pd.DataFrane): tabular dataset
        y (pd.Series): output label vector
        n_jobs (int): number of CPUs used for feature selection
        random_state (int): for reproducing the results

    Returns:
        X_tr_t (pd.DataFrame): design matrix of tabular training set 
        X_ts_t (pd.DataFrame): design matrix of tabular test set
        X_tr_tf (pd.DataFrame): design matrix of filtered tabular training set
        X_ts_tf (pd.DataFrame): design matrix of filtered tabular test set
        y_tr (pd.Series): output label vector of X_tr_t and X_tr_tf
        y_ts (pd.Series): output label vector of X_ts_t and X_ts_tf
    """
    X_tr_t, X_ts_t, y_tr, y_ts = train_test_split(
        X_t, y, stratify=y, random_state=random_state)
    
    X_tr_tf = select_features(X_tr_t, y_tr, n_jobs=n_jobs)
    X_ts_tf = X_ts_t[X_tr_tf.columns]

    return X_tr_t, X_ts_t, X_tr_tf, X_ts_tf, y_tr, y_ts
    

# %% 
def generate_anomalies(X_tr_tf, y_tr):
    """Balances the filtered tabular dataset using SMOTETomek

    Args:
        X_tr_tf (pd.DataFrame): design matrix of filtered tabular training set
        y_tr (pd.Series): output vector of training set

    Returns:
        X_tr_tf_b (pd.DataFrame): design matrix of balanced filtered tabular training set
        y_tr_b (pd.Series): output label vector of X_tr_tf_b 
    """
    X_tr_tf_b, y_tr_b = SMOTETomek().fit_resample(X_tr_tf, y_tr)
    return X_tr_tf_b, y_tr_b


# %%
def main(config_data):
    # Read config_data
    data_path = config_data["preprocess"]["data_path"]
    data_name = config_data["preprocess"]["data_name"]
    tab_path = config_data["preprocess"]["tab_path"]
    ftrs_tag = config_data["preprocess"]["ftrs_tag"]
    label_tag = config_data["preprocess"]["label_tag"]
    time_tag = config_data["preprocess"]["time_tag"]
    time_type = config_data["preprocess"]["time_type"]
    depth = config_data["preprocess"]["depth"]
    n_jobs = config_data["preprocess"]["n_jobs"]
    catg_incl = config_data["preprocess"]["catg_incl"]
    catg_tag = config_data["preprocess"]["catg_tag"]
    random_state = config_data["preprocess"]["random_state"]

    # Create target_path
    now = datetime.now().strftime("%Y_%m_%d - %H_%M")
    target_path = os.path.join(tab_path, now)

    # loading raw dataset
    print(f"loading raw dataset {data_name} in {data_path} ...")
    dataset = load_data(data_path)

    # creating block dataset
    print("creating block dataset ...")
    X, y = prepare_data(dataset, time_type, time_tag, ftrs_tag, label_tag)
    fig = visualize_dataset(y, title=f"{data_name} Class Distribution")
    save_data(fig, data_name, target_path, ".svg")

    # feature extraction
    print("starting feature extraction ...")
    X_t = my_extract_features(X, time_type, n_jobs, depth)

    # feature selection
    print("feature selection ...")
    X_tr_t, X_ts_t, X_tr_tf, X_ts_tf, y_tr, y_ts = my_select_features(X_t, y, n_jobs, random_state)

    # anomaly generation
    print("anomaly generation ...")
    X_tr_tf_b, y_tr_b = generate_anomalies(X_tr_tf, y_tr)
    
    # saving the result
    print(f"saving tabular dataset locally in {target_path} ...")
    data_out = dict(
        X_t = X_t,  
        X_tr_t = X_tr_t,
        X_ts_t = X_ts_t,
        X_tr_tf = X_tr_tf,
        X_ts_tf = X_ts_tf,
        X_tr_tf_b = X_tr_tf_b,
        y_tr = y_tr,
        y_ts = y_ts,
        y_tr_b = y_tr_b
    )
    save_data(data_out, data_name, target_path, ".dat")
    save_data(config_data["preprocess"], "config_data", target_path, ".json")


# %%
if __name__ == "__main__":
    warnings.filterwarnings("ignore")  # to ignore pandas warnings

    parser = argparse.ArgumentParser(description="Recieves the path of the config_data.yaml")
    parser.add_argument("--config_path", type=str, default="../config/config_data.yaml")
    args = parser.parse_args()
    with open(args.config_path, "r") as f:
        config_data = yaml.load(f, Loader=yaml.Loader)
    
    main(config_data)
