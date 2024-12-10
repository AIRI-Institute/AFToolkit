import os
import pickle

import numpy as np
import pandas as pd
from tqdm import tqdm

from AFToolKit.models.utils import calculate_metrics, get_model_suffix


def test_adapter(
    trained_model_path,
    test_datasets,
    results_folder,
    datasets_npy_folder,
    protein_tasks_folder=None,
):
    """Run trained `AdapterMonomer` or `AdapterMultimer` model on test sets, write results and calculate metrics."""
    # Load trained model
    with open(trained_model_path, "rb") as f:
        trained_model = pickle.load(f)

    model_suffix = get_model_suffix(trained_model, "nomultitrain" not in trained_model_path)
    
    if not os.path.exists(results_folder):
        os.makedirs(results_folder)

    for dataset_name, dataset_df in test_datasets.items():
        # For "multi-to-sum" processing scheme process dataset items one-by-one
        if hasattr(trained_model, "multi_as_singlessum") and trained_model.multi_as_singlessum and ("multi" in dataset_df["mut_type"].unique()):
            print("Making multi-to-sum predictions")
            Y_pred = []
            for idx in tqdm(dataset_df.index):
                row = dataset_df.loc[idx]
                with open(os.path.join(protein_tasks_folder, row["id"] + ".pkl"), "rb") as f:
                    protein_task = pickle.load(f)
                ddg_pred = trained_model.predict(protein_task)
                Y_pred.append(ddg_pred)
            Y_pred = np.array(Y_pred)
        else:
            dataset_X_file = os.path.join(datasets_npy_folder, f"{dataset_name}_X" + model_suffix + ".npy")
            dataset_y_file = os.path.join(datasets_npy_folder, f"{dataset_name}_y" + model_suffix + ".npy")
            if os.path.exists(dataset_X_file):
                X = np.load(dataset_X_file)
                Y = np.load(dataset_y_file)
            else:
                # load the whole dataset into np.array
                X, Y = trained_model.create_npy_dataset(
                    dataset_df, 
                    protein_tasks_folder,
                )
                np.save(dataset_X_file, X)
                np.save(dataset_y_file, Y)

            Y_pred, correlation_coefficient = trained_model.test(X, Y)
            print(f"{dataset_name} direct: {correlation_coefficient}")

        dataset_df["ddg_pred"] = Y_pred

        # Calculate metrics
        dataset_metrics = calculate_metrics(
            dataset_df["ddg_pred"].values, 
            dataset_df["ddg"].values
        )
        print("=" * 15 + f" {dataset_name} metrics " + "=" * 15)
        for k, v in dataset_metrics.items():
            print(f"{k}: {v:.3f}")
        
        dataset_df.to_csv(os.path.join(results_folder, f"results_{dataset_name}.csv"))
