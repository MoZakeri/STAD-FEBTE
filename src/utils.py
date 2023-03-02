"""
utility functions used in preprocess_data and train modules.
"""
# %%
import os
import pickle
import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# %%
class NumpyArrayEncoder(json.JSONEncoder):
    """
    Extends json.JSONEncoder to be able to write numpy arrays into json files.
    """
    def default(self, object):
        if isinstance(object, np.ndarray):
            return object.tolist()
        return json.JSONEncoder.default(self, object)


# %%
def visualize_dataset(y, title):
    sns.set_style("whitegrid")
    labels, counts = np.unique(y, return_counts=True)

    fig, ax = plt.subplots()
    ax.bar(x=labels, height=counts)
    ax.bar_label(container=ax.containers[0], labels=(counts/counts.sum()).round(2))
    ax.set_ylabel("counts")
    ax.set_title(title)
    ax.set_xticks(range(len(labels)))

    return fig 


# %%
def save_data(data, data_name, target_path, extension):
    # Create the target_path
    if not Path(target_path).is_dir():
        Path(target_path).mkdir(parents=True)

    # Save the data 
    if extension == ".dat":
        with open(os.path.join(target_path, data_name + extension), "wb") as f:
            pickle.dump(data, f)

    if extension == ".pickle":
        with open(os.path.join(target_path, data_name + extension), "wb") as f:
            pickle.dump(data, f)
    
    if extension == ".yaml":
        extension = ".txt"
        with open(os.path.join(target_path, data_name + extension), "w") as f:
            print(data, file=f)

    if extension == ".json":
        with open(os.path.join(target_path, data_name + extension), "w") as f:
            json.dump(data, f, indent=2, cls=NumpyArrayEncoder)

    if extension == ".svg":
        data.savefig(os.path.join(target_path, data_name + extension), dpi=250, bbox_inches="tight")

    if extension == ".csv":
        data.to_csv(os.path.join(target_path, data_name + extension), index=False)


# %%
def warn(*args, **kwargs):  # to silence sklearn warnings for division by zero
    pass
