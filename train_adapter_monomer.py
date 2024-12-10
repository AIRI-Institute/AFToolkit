import os
import pickle

import numpy as np
import pandas as pd
from tqdm import tqdm

from AFToolKit.models import AdapterMonomer
from AFToolKit.models.utils import get_model_suffix

def train_adapter_monomer(
    base_model, 
    adapter_name,
    train_dataset_path,
    input_features=["pair", "lddt_logits", "plddt"],
    concat_features=True,
    train_dataset_npy_folder="./",
    protein_tasks_folder=None,
    train_mut_types=["ss", "ins", "del"], # ["ss", "multi", "ins", "del"]
    protein_aggregation="mutpos",
    multi_aggregation="sum",
    add_reverse=True,
    shuffle_train=True,
):
    """Train `AdapterMonomer` model in the specified setting and save the trained model.

    Args:
        base_model: adapter model object with `sklearn` interface
        adapter_name: str, name of the adapter used to save trained model
        train_dataset_path: str, path to a `.csv` file with train/test splits
        input_features: list of str, AF2 features used to construct protein representations
        concat_features: boolean, whether to use mutant and wildtype embeddings difference
                         or concatenation as model input
        train_dataset_npy_folder: str, path to where the `.npy`-format dataset can be saved
                                  or loaded from. Dataset will be loaded from this folder,
                                  if appropriate files exist. Otherwise it will be constructed
                                  from `.pkl` files in `protein_tasks_folder`
        protein_tasks_folder: str, folder that contains `.pkl` files of `ProteinTask`s 
                              with pre-calculated AF2 features. Will not be used if `.npy` 
                              dataset can be loaded from `train_dataset_npy_folder`
        train_mut_types: list of str, mutation types to use in training
        protein_aggregation: str, method of aggregation of per-protein embeddings.
                             One of `ProteinTask.PROTEIN_AGGREGATION_OPTIONS`
        multi_aggregation: str, method of aggregation of multiple mutation embeddings
                           into one when `protein_aggregation == "mutpos"`. 
                           One of `ProteinTask.MULTIPLE_AGGREGATION_OPTIONS`
        add_reverse: boolean, whether to add reverse mutations during training
        shuffle_train: boolean, whether to shuffle the dataset before training
    """

    # Load data
    df = pd.read_csv(train_dataset_path, index_col=0)
    train_df = df[df["split"] == "train"]
    test_df = df[df["split"] == "test"]

    multi_not_in_train = "multi" not in train_mut_types
    # Create model instance
    model = AdapterMonomer(
        features_list=input_features,
        base_model=base_model,
        concat_features=concat_features,
        protein_aggregation=protein_aggregation,
        multi_aggregation=multi_aggregation,
        multi_as_singlessum=multi_not_in_train,
    )

    # Define experiment name to load / store data
    model_suffix = get_model_suffix(model, not multi_not_in_train)
    print(model_suffix)

    # Load dataset
    train_X_file = os.path.join(train_dataset_npy_folder, f"train_X" + model_suffix + ".npy")
    train_y_file = os.path.join(train_dataset_npy_folder, f"train_y" + model_suffix + ".npy")
    test_X_file = os.path.join(train_dataset_npy_folder, f"test_X" + model_suffix + ".npy")
    test_y_file = os.path.join(train_dataset_npy_folder, f"test_y" + model_suffix + ".npy")
    if os.path.exists(train_X_file):
        train_X = np.load(train_X_file)
        train_Y = np.load(train_y_file)
        test_X = np.load(test_X_file)
        test_Y = np.load(test_y_file)
    else:
        train_X, train_Y = model.create_npy_dataset(
            train_df[train_df["mut_type"].isin(train_mut_types)],
            protein_tasks_folder,
        )
        np.save(train_X_file, train_X)
        np.save(train_y_file, train_Y)

        test_X, test_Y = model.create_npy_dataset(
            test_df[test_df["mut_type"].isin(train_mut_types)],
            protein_tasks_folder,
        )
        np.save(test_X_file, test_X)
        np.save(test_y_file, test_Y)

    # Run training
    model.train(train_X, train_Y, add_reverse, shuffle_train)#, eval_set=(test_X, test_Y))
    Y_pred, correlation_coefficient = model.test(train_X, train_Y)
    print(f"Train set Spearman correlation: {correlation_coefficient:.3f}")

    Y_pred, correlation_coefficient = model.test(test_X, test_Y)
    print(f"Test set Spearman correlation: {correlation_coefficient:.3f}")

    # Save trained model
    results_filename = os.path.join(
        "data/models/monomer/", 
        "+".join(input_features), 
        f"trained_{adapter_name}" + model_suffix + ".pkl"
    )
    if not os.path.exists(os.path.dirname(results_filename)):
        os.makedirs(os.path.dirname(results_filename))
    with open(results_filename, "wb") as f:
        pickle.dump(model, f)


def train_adapter_monomer_multisetting(    
    base_model, 
    adapter_name,
    train_dataset_path=TRAIN_DATASET_CSV,
    input_features=["pair", "lddt_logits", "plddt"],
    concat_features=True,
    train_dataset_npy_folder="./",
    protein_tasks_folder=None,
    add_reverse=True,
    shuffle_train=True,
):
    """Train `AdapterMonomer` model in 5 aggregation settings and save trained models."""

    experiments_grid = [
        {"train_types": ["ss", "ins", "del"], "protein_aggregation": "mutpos", "multi_aggregation": "sum"},
        {"train_types": ["ss", "multi", "ins", "del"], "protein_aggregation": "mutpos", "multi_aggregation": "sum"},
        {"train_types": ["ss", "multi", "ins", "del"], "protein_aggregation": "mutpos", "multi_aggregation": "mean"},
        {"train_types": ["ss", "multi", "ins", "del"], "protein_aggregation": "mean", "multi_aggregation": "sum"},
        {"train_types": ["ss", "multi", "ins", "del"], "protein_aggregation": "sum", "multi_aggregation": "sum"},
    ]

    for experiment in experiments_grid:
        train_adapter_monomer(
            base_model, 
            adapter_name,
            train_dataset_path,
            input_features=input_features,
            concat_features=concat_features,
            train_dataset_npy_folder=train_dataset_npy_folder,
            protein_tasks_folder=protein_tasks_folder,
            train_mut_types=experiment["train_types"],
            protein_aggregation=experiment["protein_aggregation"],
            multi_aggregation=experiment["multi_aggregation"],
            add_reverse=True,
            shuffle_train=True,
        )