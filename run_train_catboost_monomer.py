import os

from catboost import CatBoostRegressor

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
    catboost_model = CatBoostRegressor(
        random_seed=42, 
        depth=4,
        iterations=500,
        l2_leaf_reg=5,
        rsm=1,
        random_strength=1,
        subsample=1,
    )
    train_adapter_monomer(
        base_model=catboost_model, 
        adapter_name="catboost", 
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
    #     catboost_model,
    #     "catboost",
    #     train_dataset_path=TRAIN_DATASET_CSV, 
    #     train_dataset_npy_folder=TRAIN_DATASET_NPY_FOLDER,
    #     protein_tasks_folder=PROTEIN_TASKS_FOLDER,
    # )
