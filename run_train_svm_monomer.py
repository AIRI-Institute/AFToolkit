import os

from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.svm import SVR

from train_adapter_monomer import train_adapter_monomer, train_adapter_monomer_multisetting


# Set paths
PROTEIN_TASKS_FOLDER = "/PATH/TO/PROTEIN_TASKS/FOLDER/"
TRAIN_DATASET_CSV = "data/cdna+PROSTATA_mut_idxs.csv" 

# Set parameters
INPUT_FEATURES = ["pair", "lddt_logits", "plddt"]
CONCAT_FEATURES = True
TRAIN_TYPES = ["ss", "ins", "del"]
PROTEIN_AGGREGATION = "mutpos"
MULTI_AGGREGATION = "sum"

TRAIN_DATASET_NPY_FOLDER = os.path.join(
    "/PATH/TO/SAVE/NPY/DATASET/FOLDER/", 
    "+".join(INPUT_FEATURES)
)


if __name__ == "__main__":
    pipeline = Pipeline(
        steps=[
            ('scaler', StandardScaler()),
            ('svm', SVR(cache_size=7000)),
        ],
    )
    train_adapter_monomer(
        base_model=pipeline, 
        adapter_name='svm',
        train_dataset_path=TRAIN_DATASET_CSV, 
        input_features=INPUT_FEATURES,
        concat_features=CONCAT_FEATURES,
        train_dataset_npy_folder=TRAIN_DATASET_NPY_FOLDER,
        protein_tasks_folder=PROTEIN_TASKS_FOLDER,
        train_mut_types=TRAIN_TYPES,
        protein_aggregation=PROTEIN_AGGREGATION,
        multi_aggregation=MULTI_AGGREGATION,
    )
    # Uncomment to train models in 5 aggregation settings at once
    # train_adapter_monomer_multisetting(
    #     pipeline,
    #     "svm",
    #     train_dataset_path=TRAIN_DATASET_CSV, 
    #     train_dataset_npy_folder=TRAIN_DATASET_NPY_FOLDER,
    #     protein_tasks_folder=PROTEIN_TASKS_FOLDER,
    # )