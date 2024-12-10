import os

from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.neural_network import MLPRegressor

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
    mlp = MLPRegressor(
        hidden_layer_sizes=(100,),
        activation='relu',
        solver='sgd',
        alpha=0.0001,
        batch_size=256,
        learning_rate='invscaling',
        learning_rate_init=0.001,
        power_t=0.5,
        max_iter=50,
        shuffle=True,
        random_state=42,
        tol=0.0001,
        verbose=False,
        warm_start=False,
        momentum=0.9,
        nesterovs_momentum=True,
        early_stopping=True,
        validation_fraction=0.1,
        beta_1=0.9,
        beta_2=0.999,
        epsilon=1e-08,
        n_iter_no_change=10,
        max_fun=15000,
    )
    pipeline = Pipeline(
        steps=[
            ('scaler', StandardScaler()),
            ('mlp', mlp),
        ],
    )
    train_adapter_monomer(
        base_model=pipeline, 
        adapter_name='mlp',
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
    #     "mlp",
    #     train_dataset_path=TRAIN_DATASET_CSV, 
    #     train_dataset_npy_folder=TRAIN_DATASET_NPY_FOLDER,
    #     protein_tasks_folder=PROTEIN_TASKS_FOLDER,
    # )