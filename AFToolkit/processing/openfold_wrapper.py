import os
import torch

import numpy as np
import pandas as pd

from functools import partial

import AFToolKit.processing.openfold.np.protein as of_protein
from AFToolKit.processing.openfold.config import model_config
from AFToolKit.processing.openfold.data import feature_pipeline
from AFToolKit.processing.openfold.utils.script_utils import load_models_from_command_line, prep_output
from AFToolKit.processing.openfold.utils.tensor_utils import dict_multimap, tensor_tree_map
from AFToolKit.processing.openfold.np import residue_constants

from AFToolKit.processing.utils import pdb_str_to_dataframe, runcmd
from AFToolKit.constants import MODEL_WEIGHTS_DIR, OF_WEIGHTS, SOURCE_OF_WEIGHTS_URL


class OpenFoldWrapper:
    """
    Class to calculate AlphaFold structure and extract positions embedding
    """

    def __init__(
            self,
            feature_list=None,
            device="mps",
            inference_n_recycle=1,
            always_use_template=False,
            side_chain_mask=False,
            return_all_cycles=False
    ):
        """
        return_all_cycles -- return embeddings for each cycle
        always_use_template -- use 3D structure template across all cycles (True -- standard pipeline),
                            False -- our pipeline (default value))
        side_chain_mask -- mask all side chain atoms (True -- 'gap' pipeline,
                           False -- standard and our pipeline(default value))
        """
        self.device = device
        self.feature_list = feature_list
        self.config = model_config("model_2_ptm", low_prec=True)
        self.feature_processor = feature_pipeline.FeaturePipeline(self.config.data)
        self.output_directory = './'
        self.model = None
        self.protein = pd.DataFrame()
        self.of_output = None
        self.of_protein = None
        self.of_output_per_cycle = None

        self.n_recycle = inference_n_recycle
        self.always_use_template = always_use_template
        self.side_chain_mask = side_chain_mask
        self.return_all_cycles = return_all_cycles

    def init_model(self, weights_dir=MODEL_WEIGHTS_DIR):
        weights_path = os.path.join(weights_dir, OF_WEIGHTS)
        if weights_dir is not None:
            if not os.path.exists(weights_path):
                try:
                    if not os.path.exists(weights_dir):
                        print(f'create {weights_dir}')
                        runcmd(f"mkdir {weights_dir}")
                    print(f"""
                    AlphaFold weights not found. It will be downloaded 
                    into {weights_dir} directory
                    """)
                    tmp_path = f"{weights_dir}/af_colab.tar"
                    runcmd(f"wget -O {tmp_path} {SOURCE_OF_WEIGHTS_URL}")
                    runcmd(f"""
                    tar --extract  --file={tmp_path} --directory="{weights_dir}"
                    """)
                    runcmd(f"rm {tmp_path}")
                except Exception as e:
                    print(f'Error while downloading weights occurs: {e}')
                    raise
            model_generator = load_models_from_command_line(
                self.config, self.device, None, weights_path, None
            )
            for model, output_directory in model_generator:
                self.model = model
                self.output_directory = output_directory
            self.model.to(self.device)
        else:
            print('Warning: path to model weight isn\'t set')

    def set_protein(self, protein_df):
        self.protein = protein_df

    def get_of_protein(self):
        return self.of_protein

    def get_cycles(self):
        return self.of_output_per_cycle

    def get_of_features_by_positions(self, positions):
        pdb_df_pred_ca = self.of_protein[self.of_protein["atom_name"] == "CA"] \
            .reset_index(drop=True)

        assert pdb_df_pred_ca.shape[0] == self.of_output["single"].shape[0], print(
            "OF output and PDB size dimensions doesn't match"
        )
        features = {}
        for name, id in positions.items():
            features[name] = self.extract_residue_features(idx=id, of_output=self.of_output)

        return features

    def get_of_features_by_positions_per_cycle(self, positions):
        pdb_df_pred_ca = self.of_protein[self.of_protein["atom_name"] == "CA"] \
            .reset_index(drop=True)
        of_cylces = []
        for of_cylce in self.of_output_per_cycle:
            assert pdb_df_pred_ca.shape[0] == of_cylce["single"].shape[0], print(
                "OF output and PDB size dimensions doesn't match"
            )
            features = {}
            for name, id in positions.items():
                features[name] = self.extract_residue_features(idx=id, of_output=of_cylce)
            of_cylces.append(features)
        return of_cylces

    @staticmethod
    def prepare_features(target_protein):
        """prepare protein features for AlphaFold calculation"""
        sequence = residue_constants.aatype_to_str_sequence(target_protein.aatype)
        features = {
            "template_all_atom_positions": target_protein.atom_positions[None, ...],
            "template_all_atom_mask": target_protein.atom_mask[None, ...],
            "template_sequence": [sequence],
            "template_aatype": target_protein.aatype[None, ...],
            "template_domain_names": [None],  # f''.encode()]
        }

        # ToDo: look for more elegant way to calculate sequence features
        sequence_features = {}
        num_res = len(sequence)
        sequence_features["aatype"] = residue_constants.sequence_to_onehot(
            sequence=sequence,
            mapping=residue_constants.restype_order_with_x,
            map_unknown_to_x=True,
        )
        sequence_features["between_segment_residues"] = np.zeros(
            (num_res,), dtype=np.int32
        )
        sequence_features["domain_name"] = np.array(
            ["input".encode("utf-8")], dtype=np.object_
        )
        sequence_features["residue_index"] = np.array(
            target_protein.residue_index, dtype=np.int32
        )
        sequence_features["seq_length"] = np.array([num_res] * num_res, dtype=np.int32)
        sequence_features["sequence"] = np.array(
            [sequence.encode("utf-8")], dtype=np.object_
        )
        deletion_matrix = np.zeros(num_res)
        sequence_features["deletion_matrix_int"] = np.array(
            deletion_matrix, dtype=np.int32
        )[None, ...]
        int_msa = [residue_constants.HHBLITS_AA_TO_ID[res] for res in sequence]
        sequence_features["msa"] = np.array(int_msa, dtype=np.int32)[None, ...]
        sequence_features["num_alignments"] = np.array(np.ones(num_res), dtype=np.int32)
        sequence_features["msa_species_identifiers"] = np.array(["".encode()])
        feature_dict = {**sequence_features, **features}
        return feature_dict

    @staticmethod
    def get_batch_features(af_output_batch, af_input_features, batch_id):
        """
        Input: output of alphafold
               features for alphafold inference
               batch_id
        Output:
               output and features for batch_id
        """
        out = {"sm": {}}
        for k, out_ in af_output_batch.items():
            if k == "sm":
                for name in out_:
                    if out_[name].shape[0] == 1:
                        out["sm"][name] = out_[name][batch_id, ...]
                    if out_[name].shape[0] == 8:
                        out["sm"][name] = out_[name][:, batch_id, ...]
                continue
            if len(out_.shape) == 0:
                out[k] = out_
                continue
            out[k] = out_[batch_id, ...]
        features = tensor_tree_map(
            lambda x: np.array(x[batch_id, ..., -1].cpu()), af_input_features
        )

        return out, features

    def extract_residue_features(self, idx, of_output):
        output_upd = {"msa": of_output["msa"][0, idx, :],
                      "pair": of_output["pair"][idx, idx, :],
                      "lddt_logits": of_output["lddt_logits"][idx, :],
                      "distogram_logits": of_output["distogram_logits"][idx, idx, :],
                      "aligned_confidence_probs": of_output["aligned_confidence_probs"][idx, idx, :],
                      "predicted_aligned_error": of_output["predicted_aligned_error"][idx, idx].reshape(-1),
                      "plddt": of_output["plddt"][idx].reshape(-1),
                      "single": of_output["single"][idx, :],
                      "tm_logits": of_output["tm_logits"][idx, idx, :]}
        return output_upd

    def inference_monomer(self, pdb_df):
        """inference openfold with pdb dataframe input
         0 cycle with template features
         1 cycle masked template features
        ...
        n cycle masked template features

        return alphafold output and predicted pdb structure
        """
        pdb = of_protein.from_pdb_df(pdb_df)
        features = self.prepare_features(pdb)

        if self.side_chain_mask:
            pdb_backbone = pdb_df[pdb_df["atom_name"].isin(["N", "C", "CA", "O", "CB"])]
            pdb_backbone.loc[:, "residue_name"] = "UNK"
            pdb_backbone = of_protein.from_pdb_df(pdb_backbone)
            features_backbone = self.prepare_features(pdb_backbone)
            for c in features_backbone:
                if c.startswith("template"):
                    features[c] = features_backbone[c]

        pdb_df_ca = pdb_df[pdb_df["atom_name"] == "CA"].reset_index()
        features["template_all_atom_mask"][:,
        pdb_df_ca[pdb_df_ca["mask"]].index,
        :] *= 0
        processed_feature_dict = self.feature_processor.process_features(
            features,
            mode="predict",
        )
        """ add recycling features with masked template features """
        processed_feature_dict_list = [
            processed_feature_dict
        ]

        for i in range(self.n_recycle):
            processed_feature_dict_list.append(
                {k: p.detach().clone() for k, p in processed_feature_dict.items()}
            )
            if not self.always_use_template:
                processed_feature_dict_list[-1]["template_mask"] *= 0

        cat_fn = partial(torch.cat, dim=-1)
        processed_feature_dict = dict_multimap(cat_fn, processed_feature_dict_list)

        for c, p in processed_feature_dict.items():
            if p.dtype == torch.float64:
                processed_feature_dict[c] = torch.as_tensor(
                    p, dtype=torch.float32, device=self.device
                )
            else:
                processed_feature_dict[c] = torch.as_tensor(p, device=self.device)
            processed_feature_dict[c] = processed_feature_dict[c][None, ...]

        # load alphafold model
        with torch.no_grad():
            out_batch, out_per_cycle = self.model(processed_feature_dict)

        for i in range(len(out_per_cycle)):
            out_per_cycle[i] = tensor_tree_map(
                lambda x: np.array(x[...].detach().cpu()), out_per_cycle[i]
            )
            out_per_cycle[i], _ = self.get_batch_features(out_per_cycle[i], {}, 0)
        out_batch = tensor_tree_map(
            lambda x: np.array(x[...].detach().cpu()), out_batch
        )
        out, ifd = self.get_batch_features(out_batch, processed_feature_dict, 0)
        unrelaxed_protein = prep_output(
            out, ifd, ifd, self.feature_processor, "model_2_ptm", 200, False
        )
        pdb_str = of_protein.to_pdb(unrelaxed_protein)
        pdb_df_pred = pdb_str_to_dataframe(pdb_str, pdb_df)

        if self.return_all_cycles:
            return out, out_per_cycle, pdb_df_pred

        return out, None, pdb_df_pred

    def evaluate(self, protein_df):
        self.set_protein(protein_df=protein_df)
        self.of_output, self.of_output_per_cycle, self.of_protein = self.inference_monomer(
            pdb_df=protein_df)
