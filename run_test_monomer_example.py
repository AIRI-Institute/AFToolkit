import os

import pandas as pd

from test_adapter import test_adapter

# Set paths
PROTEIN_TASKS_FOLDER = "/PATH/TO/PROTEIN_TASKS/FOLDER/"
DATASETS_NPY_FOLDER = "/PATH/TO/SAVE/NPY/DATASET/FOLDER/"

# svm model path
MODEL_PATH = "data/models/monomer/pair+lddt_logits+plddt/trained_svm_concat_nomultitrain_aggmutpos_multisum.pkl"
# mlp model
# MODEL_PATH = "data/models/monomer/pair+lddt_logits+plddt/trained_mlp_concat_nomultitrain_aggmutpos_multisum.pkl"
# catboost model
# MODEL_PATH = "data/models/monomer/pair+lddt_logits+plddt/trained_catboost_concat_nomultitrain_aggmutpos_multisum.pkl"

RESULTS_FOLDER = os.path.join("svm_results/")
TEST_DATASETS = {
    "s669": pd.read_csv("data/s669_mut_idxs.csv", index_col=0),
    "ssym": pd.read_csv("data/ssym_mut_idxs.csv", index_col=0),
    "protherm": pd.read_csv("data/protherm_mut_idxs.csv", index_col=0),
    "denovo": pd.read_csv("data/denovo_mut_idxs.csv", index_col=0),
    # "cdna1": pd.read_csv("data/cdna1_mut_idxs.csv", index_col=0),
    # "cdna2": pd.read_csv("data/cdna2_mut_idxs.csv", index_col=0),
    "cdna_indel": pd.read_csv("data/cdna_indels_mut_idxs.csv", index_col=0),
}

if __name__ == "__main__":
    test_adapter(
        trained_model_path=MODEL_PATH,
        test_datasets=TEST_DATASETS,
        results_folder=RESULTS_FOLDER,
        datasets_npy_folder=DATASETS_NPY_FOLDER,
        protein_tasks_folder=PROTEIN_TASKS_FOLDER,
    )

