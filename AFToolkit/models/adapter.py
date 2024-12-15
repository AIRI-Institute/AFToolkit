import os
import pickle

import numpy as np
from scipy.stats import pearsonr, spearmanr
from tqdm import tqdm


class AdapterMonomer:
    def __init__(
        self,
        features_list,
        base_model,
        concat_features=True,
        protein_aggregation="mutpos",
        multi_aggregation="sum",
        multi_as_singlessum=False,
        random_seed=42,
    ):
        self.features_list = features_list
        self.base_model = base_model
        self.concat_features = concat_features

        self.protein_aggregation = protein_aggregation
        self.multi_aggregation = multi_aggregation
        self.multi_as_singlessum = multi_as_singlessum
        self.random_seed = random_seed

    def __call__(self, protein_task):
        return self.predict(protein_task)

    def predict(self, protein_task, treat_multiple_as_singles=False):
        """Predict for a single `ProteinTask` object using a pre-trained Adapter."""
        if self.multi_as_singlessum or treat_multiple_as_singles:
            # introduce single mutations one-at-a-time and add their effects
            single_predictions = []
            all_mutations = protein_task.task["mutants"]
            for mut_pos, mut_aa in all_mutations.items():
                protein_task.set_task_mutants({mut_pos: mut_aa})
                X = self.protein_task_to_input_features(protein_task)
                single_prediction = self.base_model.predict(X.reshape(1, -1))
                single_predictions.append(single_prediction)
            prediction = sum(single_predictions)
        else:
            X = self.protein_task_to_input_features(protein_task)
            prediction = self.base_model.predict(X)
        return prediction
    
    def train(
        self, 
        X_train,
        y_train,
        add_reverse=True,
        shuffle_train=True,
        **fit_kws,
    ):
        """
        Train Adapter model on input data.

        Args:
            X_train: np.array-format train dataset.
            y_train: np.array-format train targets.
            add_reverse: boolean, whether to add reverse mutations to the dataset
            shuffle_train: boolean, whether to shuffle the dataset before training
        """
        if add_reverse:
            if self.concat_features:
                reverse_X_train = np.concatenate((
                    X_train[:, X_train.shape[1] // 2:], 
                    X_train[:, :X_train.shape[1] // 2]
                ), 1)
                reverse_y_train = -y_train
            else:
                reverse_X_train = -X_train
                reverse_y_train = -y_train
            X_train = np.concatenate((X_train, reverse_X_train))
            y_train = np.concatenate((y_train, reverse_y_train))

        # shuffle the train set
        if shuffle_train:
            np.random.seed(self.random_seed)
            p = np.random.permutation(X_train.shape[0])
            X_train = X_train[p]
            y_train = y_train[p]

        # fit the pipeline
        self.base_model.fit(
            X_train, y_train, **fit_kws,
        )

    def test(self, input_features, targets):
        """
        Test Adapter model on input data.

        Args:
            input_features: np.array-format test dataset \
                            or dict of {protein_name: `ProteinTask` object}.
            targets: np.array-format test targets \
                     or dict of {protein_name: target value}.

        Returns:
            Predictions for the test set and test set Spearman correlation coefficient.
        """
        if isinstance(input_features, dict) and isinstance(targets, dict):
            X, Y = self.create_proteintask_dataset(input_features, targets)
        else:
            X, Y = input_features, targets
        Y_pred = self.base_model.predict(X)

        correlation_coefficient, p_value = spearmanr(Y, Y_pred)
        return Y_pred, correlation_coefficient

    def create_proteintask_dataset(self, features_dict, targets_dict):
        """Create np.array-format dataset of inputs with targets to fit/test the model on \
        from dict of {protein_name: `ProteinTask` object}."""
        X, Y = [], []
        for protein_name, protein_task in features_dict.items():
            X_ = self.protein_task_to_input_features(protein_task)
            Y_ = np.array(targets_dict[protein_name])
            if Y_.ndim == 0:
                Y_ = Y_.reshape(1)
            X.append(X_)
            Y.append(Y_)
        X = np.concatenate(X)
        Y = np.concatenate(Y)
        return X, Y
    
    def create_npy_dataset(self, df, protein_task_folder):
        """Create np.array-format dataset of inputs with targets to fit/test the model on \
        from pd.DataFrame of mutations and folder with pre-calculated `ProteinTask` pkl files. \
        Unlike `create_proteintask_dataset`, this doesn't load the whole dataset \
        of `ProteinTask`s into memory, only the input features for each sample.
        
        Args:
            df: dataframe of mutations with unique `id`s.
            protein_task_folder: folder with `ProteinTask` *.pkl files for each mutation
                                 in the dataframe. File "X.pkl" should match mutation with id "X".
            pick_mutpos: whether to use embeddings of mutation positions only or the whole protein.
                         Mutation positions to be picked are taken from "wt_pos_idxs" and "mut_pos_idxs" \
                         columns of the dataframe.
        """
        X, y = [], []
        for idx in tqdm(df.index):
            row = df.loc[idx]
            with open(os.path.join(protein_task_folder, row["id"] + ".pkl"), "rb") as f:
                protein_task = pickle.load(f)
            
            X.append(self.protein_task_to_input_features(protein_task))
            y.append(np.array(row["ddg"]).reshape(1))
        return np.stack(X), np.stack(y)

    def protein_task_to_input_features(self, protein_task):
        """Get input features (with names from `self.features_list`) \
        from input `ProteinTask` with pre-calculated features."""
        wt_feats, mt_feats = protein_task.get_protein_embeddings(
            self.features_list, 
            self.protein_aggregation,
            self.multi_aggregation,
        )
        if self.concat_features:
            return np.concatenate((wt_feats, mt_feats))
        return mt_feats - wt_feats


class AdapterMultimer(AdapterMonomer):
    def __init__(self, only_complex_features=True, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.only_complex_features = only_complex_features

    def protein_task_to_input_features(self, protein_complex_task):
        """Get SVM input features (with names from `self.features_list`) \
        from input `ProteinTask` with pre-calculated features."""
        wt_feats_complex, mt_feats_complex = protein_complex_task.complex.get_protein_embeddings(
            self.features_list, 
            self.protein_aggregation,
            self.multi_aggregation,
        )
        if self.only_complex_features:
            return np.concatenate((wt_feats_complex, mt_feats_complex))
        else:
            wt_feats_mono, mt_feats_mono = protein_complex_task.protein.get_protein_embeddings(
            self.features_list, 
            self.protein_aggregation,
            self.multi_aggregation,
            )
            return np.concatenate((wt_feats_mono, mt_feats_mono,
            wt_feats_complex, mt_feats_complex))


class AdapterMonomerPerResiduePredictor(AdapterMonomer):
    def protein_task_to_input_features(self, protein_task):
        """Get SVM input features (with names from `self.features_list`) \
        from input `ProteinTask` with pre-calculated features."""
        all_feats = []
        wt_features = protein_task.get_wildtype_protein_of_features()
        position_names = wt_features.keys()
        for position_name in position_names:
            pos_wt_feats = [
                wt_features[position_name][f_name] for f_name in self.features_list
            ]
            all_feats.append(np.concatenate(pos_wt_feats, axis=0))
        return all_feats
