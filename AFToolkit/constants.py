import os

PROJECT_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_WEIGHTS_DIR = os.path.join(PROJECT_PATH, 'weights')
OF_WEIGHTS = "params_model_2_ptm.npz"
SOURCE_OF_WEIGHTS_URL = "https://storage.googleapis.com/alphafold/alphafold_params_colab_2022-12-06.tar"