import os

import pandas as pd

from test_adapter import test_adapter

# Set paths
PROTEIN_TASKS_FOLDER = "/PATH/TO/PROTEIN_TASKS/FOLDER/"
DATASETS_NPY_FOLDER = "/PATH/TO/SAVE/NPY/DATASET/FOLDER/"

#svm model path
MODEL_PATH = "data/models/multimer/pair+lddt_logits+plddt/trained_svm_concat_nomultitrain_aggmutpos_multisum.pkl"


RESULTS_FOLDER = os.path.join("svm_results/")
TEST_DATASETS = {
    "c380": pd.read_csv("data/c380.csv", index_col=0),
}

if __name__ == "__main__":
    test_adapter(
        trained_model_path=MODEL_PATH,
        test_datasets=TEST_DATASETS,
        results_folder=RESULTS_FOLDER,
        datasets_npy_folder=DATASETS_NPY_FOLDER,
        protein_tasks_folder=PROTEIN_TASKS_FOLDER,
    )

