"""_summary_
"""
# %%
import warnings
import argparse
import os
import pickle
import yaml
import json
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from sklearn.metrics import recall_score, f1_score, roc_auc_score
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.exceptions import UndefinedMetricWarning


# %%
def warn(*args, **kwargs):
    pass


class NumpyArrayEncoder(json.JSONEncoder):
    """
    Extends json.JSONEncoder to be able to write numpy arrays into json files.
    """

    def default(self, object):
        if isinstance(object, np.ndarray):
            return object.tolist()
        return json.JSONEncoder.default(self, object)


def load_data(data_path, train_on_FE, train_on_FS):
    with open(data_path, "rb") as f:
        dataset = pickle.load(f)
    
    dataset_tr = [dataset["X_tr_tf_b"].to_numpy()]
    label_tr = [dataset["y_tr_b"].to_numpy()]
    dataset_ts = [dataset["X_ts_tf"].to_numpy()]
    label_ts = [dataset["y_ts"].to_numpy()]
    data_names = ["tab_fs_b"]

    if train_on_FE and not train_on_FS:
        dataset_tr.append(dataset["X_tr_t"].to_numpy())
        label_tr.append(dataset["y_tr"].to_numpy())
        dataset_ts.append(dataset["X_ts_t"].to_numpy())
        label_ts.append(dataset["y_ts"].to_numpy())
        data_names = ["tab_fs_b", "tab"]

    elif not train_on_FE and train_on_FS:
        dataset_tr.append(dataset["X_tr_tf"].to_numpy())
        label_tr.append(dataset["y_tr"].to_numpy())
        dataset_ts.append(dataset["X_ts_tf"].to_numpy())
        label_ts.append(dataset["y_ts"].to_numpy())
        data_names = ["tab_fs_b", "tab_fs"]

    elif train_on_FE and train_on_FS:
        dataset_tr.extend([dataset["X_tr_tf"].to_numpy(), dataset["X_tr_t"].to_numpy()])
        label_tr.extend([dataset["y_tr"].to_numpy(), dataset["y_tr"].to_numpy()])
        dataset_ts.extend([dataset["X_ts_tf"].to_numpy(), dataset["X_ts_t"].to_numpy()])
        label_ts.extend([dataset["y_ts"].to_numpy(), dataset["y_ts"].to_numpy()])
        data_names = ["tab_fs_b", "tab_fs", "tab"]

    dataset_tr = list(reversed(dataset_tr))
    dataset_ts = list(reversed(dataset_ts))
    label_tr = list(reversed(label_tr))
    label_ts = list(reversed(label_ts))
    data_names = list(reversed(data_names))

    return dataset_tr, dataset_ts, label_tr, label_ts, data_names


def train_defaultHP(model_name, X, y, data_name, n_estimators, n_jobs, model_path, dataset_name):
    """
    Trains various tree-based ensembles on a given dataset with default HPT.

    Args:
        model_name (str): model name. Should be in ["bagging", "rf", "extra_trees", "ada_boost", "grad_boost"]
        X (np.ndarray): design matrix of the training dataset
        y (np.ndarray): output vector of training dataset
        data_name (str): name of the training data, appears in training message and saved model
        n_estimators (int): number of estimators of the ensemble model
        n_jobs (int): number of CPUs

    Raises:
        Exception: if model_name is not in ["bagging", "rf", "extra_trees", "ada_boost", "grad_boost"]

    Returns:
        model: trained sklearn classifier
    """
    valid_model_names = ["bagging", "rf",
                         "extra_trees", "ada_boost", "grad_boost"]

    if model_name.lower() == "bagging":
        model_base = DecisionTreeClassifier(criterion="gini", max_depth=None)
        model = BaggingClassifier(estimator=model_base,
                                  bootstrap_features=True,
                                  n_estimators=n_estimators,
                                  n_jobs=n_jobs)

    elif model_name.lower() == "rf":
        model = RandomForestClassifier(n_estimators=n_estimators,
                                       n_jobs=n_jobs)

    elif model_name.lower() == "extra_trees":
        model = ExtraTreesClassifier(n_estimators=n_estimators,
                                     n_jobs=n_jobs)

    elif model_name.lower() == "ada_boost":
        model_base = DecisionTreeClassifier(max_depth=1)  # high bias
        model = AdaBoostClassifier(estimator=model_base,
                                   n_estimators=n_estimators)

    elif model_name.lower() == "grad_boost":
        model = HistGradientBoostingClassifier()

    else:
        raise Exception(
            f"Invalid model_name - it should be in \n{valid_model_names}"
        )

    print(f"{model_name} is getting trained on {dataset_name}_{data_name} ...")
    model.fit(X, y)

    # Save the trained model locally
    print(f"Trained model is saved in {model_path} ...")
    if not os.path.isdir(model_path):
        os.mkdir(model_path)
    with open(f"{model_path}/{dataset_name}_{model_name}_{data_name}.pickle", "wb") as f:
        pickle.dump(model, f)

    return model


def report_metrics(model, X, y, model_name, data_name, report_path, dataset_name):
    """
    Computes various performance metrics for a trained classifier.

    Args:
        model (sklearn classifier): trained sklearn classifier object
        X (np.ndarray): design matrix of test dataset
        y (np.ndarray): output vector of test dataset
        model_name (str): name of the model, appears in saved report
        data_name (str): name of the passed dataset, appears in saved report
        report_path (str): directory to save evaluation metrics
        dataset_name (str): given name of the dataset, appears in the saved report

    Returns:
        report (dict): created report holding following keys: 
                        - "model_name": name of the model
                        - "accuracy": model accuracy
                        - "balanced_accuracy": model balanced accuracy
                        - "recall": model recall
                        - "f1": model f1 score
                        - "roc_auc": model ROC-AUC score
                        - "cr": model classification report
                        - "cm": model confusion matrix 
    """
    # Create the report
    report = dict()
    report["model_name"] = model_name

    # accuracy
    acc = accuracy_score(y, model.predict(X))
    report["accuracy"] = acc

    # balanced accuracy
    b_acc = balanced_accuracy_score(y, model.predict(X))
    report["balanced_accuracy"] = b_acc

    # recall
    rec = recall_score(y, model.predict(X), average="weighted")
    report["recall"] = rec

    # f1 score
    f1 = f1_score(y, model.predict(X), average="weighted")
    report["f1"] = f1

    # roc_auc score
    roc = roc_auc_score(y, model.predict_proba(X), average="weighted",
                        multi_class="ovr")
    report["roc"] = roc

    # classification report
    cr = classification_report(y, model.predict(X))
    report["cr"] = cr

    # confusion matrix
    cm = confusion_matrix(y, model.predict(X), normalize="true")
    report["cm"] = cm

    # Save report
    print(f"Evaluation metrics for {model_name} trained on {dataset_name}_{data_name} are locally saved in {report_path} ...")
    if not os.path.isdir(report_path):
        os.mkdir(report_path)

    with open(f"{report_path}/{dataset_name}_{model_name}_{data_name}.json", "w") as f:
        json.dump(report, f, indent=2, cls=NumpyArrayEncoder)

    return report


# %%
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="asks for the path of the train_config.yaml file")
    parser.add_argument("--config_path", type=str, default="../config/train_config_synthetic_fixed.yaml")
    args = parser.parse_args()

    with open(args.config_path, "r") as f:
        config_data = yaml.load(f, Loader=yaml.Loader)

    warnings.warn = warn  # to silence sklearn warnings
    
    data_path = config_data["data_path"]
    dataset_name = config_data["dataset_name"]
    model_path = config_data["model_path"]
    report_path = config_data["report_path"]
    model_names = config_data["model_names"]
    train_on_FE = config_data["train_on_FE"]
    train_on_FS = config_data["train_on_FS"]
    n_estimators = config_data["n_estimators"]
    n_jobs = config_data["n_jobs"]

    dataset_tr, dataset_ts, label_tr, label_ts, data_names = load_data(data_path, train_on_FE, train_on_FS)

    for model_name in model_names:
        for data_name, X_tr, X_ts, y_tr, y_ts in zip(data_names, dataset_tr, dataset_ts, label_tr, label_ts):
            # Train
            model = train_defaultHP(
                model_name, X_tr, y_tr, data_name, n_estimators, n_jobs, model_path, dataset_name)
            # Validate
            report = report_metrics(
                model, X_ts, y_ts, model_name, data_name, report_path, dataset_name)
