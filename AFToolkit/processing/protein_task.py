import pickle
import json

import numpy as np
import pandas as pd

from AFToolKit.processing.utils import (
    load_protein, 
    mutate_protein, 
    AA_3_TO_1_DICT,
    remove_sidechain_for_mutate_position
)


class ProteinTask:
    PROTEIN_AGGREGATION_OPTIONS = ["mutpos", "sum", "mean"]
    MULTIPLE_AGGREGATION_OPTIONS = ["sum", "mean"]

    def __init__(self, task=None, protein_job=None, protein_of=None):
        """
        ProteinTask class contains info about protein and set mutation
        task: json
        """
        if task is None:
            self.task = {
                "input_protein": {
                    "path": None,  # input protein path
                    "chains": None,  # input protein chains
                    "regions": None,  # input protein regions
                },
                "mutants": {},  # input mutants in format (aa_wt, residue_position, aa_mt, chain_id)
                "obs_positions": [],  # positions to calculate embeddings for in original numbering system
            }
        else:
            self.task = task
        if protein_job is None:
            self.protein_job = {
                "protein_wt": {
                    "protein": pd.DataFrame(),  # processed protein dataframe with introduced mutants
                    "obs_positions": {},  # positions in the internal numbering system
                },
                "protein_mt": {
                    "protein": pd.DataFrame(),  # processed protein dataframe with introduced mutants
                    "obs_positions": {}  # positions in the internal numbering system
                }
            }
        else:
            self.protein_job = protein_job
        if protein_of is None:
            self.protein_of = {
                "protein_wt": {
                    'protein': pd.DataFrame(),
                    'features': {}
                },
                "protein_mt": {
                    'protein': pd.DataFrame(),
                    'features': {}
                }
            }
        else:
            self.protein_of = protein_of

    def set_input_protein_task(self, protein_path=None, chains=None, regions=None, task=None):
        if task is None:
            self.task["input_protein"] = {"path": protein_path,
                                          "chains": chains,
                                          "regions": regions}
        else:
            self.task = task

    def set_task_mutants(self, mutants):
        self.task['mutants'] = mutants

    def add_mutation(self, chain_id, resi, aa_mt, aa_wt):
        self.task["mutants"][(aa_wt, resi, chain_id)] = aa_mt

    def get_mutations(self):
        return self.task["mutants"]

    def set_wildtype_protein(self, protein=pd.DataFrame()):
        if protein.empty:
            protein = load_protein(self.task["input_protein"]["path"],
                                   self.task["input_protein"]["chains"],
                                   self.task["input_protein"]["regions"])
        protein_wt = remove_sidechain_for_mutate_position(protein,
                                                          self.task["mutants"])
        self.protein_job["protein_wt"]["protein"] = protein_wt

    def set_mutate_protein(self, protein=pd.DataFrame()):
        if protein.empty:
            protein = mutate_protein(self.get_wildtype_protein(),
                                     self.task["mutants"])
        self.protein_job["protein_mt"]["protein"] = protein

    def set_wildtype_observable_position(self, obs_positions):
        self.protein_job["protein_wt"]["obs_positions"] = obs_positions

    def set_mutate_observable_position(self, obs_positions):
        self.protein_job["protein_mt"]["obs_positions"] = obs_positions

    def set_wildtype_of_protein(self, of_features, of_protein=pd.DataFrame(),
                                of_cycles=None):
        if of_cycles is None:
            of_cycles = {}
        self.protein_of["protein_wt"] = {
            "protein": of_protein,
            "features": of_features,
            "cycles": of_cycles
        }

    def set_mutate_of_protein(self, of_features, of_protein=pd.DataFrame(),
                              of_cycles=None):
        if of_cycles is None:
            of_cycles = {}
        self.protein_of["protein_mt"] = {
            "protein": of_protein,
            "features": of_features,
            "cycles": of_cycles
        }

    def set_observable_positions(self, obs_positions=None):
        if obs_positions is None:
            self.task["obs_positions"] = []
            prot_wt = self.get_wildtype_protein()
            if self.task["input_protein"]['chains'] is not None:
                chains = self.task["input_protein"]['chains']
            else:
                chains = prot_wt.chain_id_original.unique()
            for chain in chains:
                start = min(prot_wt.loc[prot_wt['chain_id_original'] == chain,
                                            "residue_number_original"])
                end = prot_wt.loc[(prot_wt['chain_id_original'] == chain) &
                                      (prot_wt.atom_name == 'CA'),
                                ].shape[0]
                self.task["obs_positions"].append({
                    "resi": start,
                    "shift_from_resi": end - 1,
                    "chain_id": chain})
        else:
            self.task["obs_positions"] = obs_positions

    def add_observable_positions(self, resi=None, shift_from_resi=0, chain_id=None, name=None):
        if name is None:
            name = f"{resi}_{chain_id}"
        self.task["obs_positions"].append({"resi": resi,
                                           "shift_from_resi": shift_from_resi,
                                           "chain_id": chain_id,
                                           "name": name})

    def get_mutate_protein(self):
        return self.protein_job['protein_mt']['protein']

    def get_wildtype_protein(self):
        return self.protein_job['protein_wt']['protein']

    def get_mutate_observable_positions(self):
        return self.protein_job['protein_mt']['obs_positions']

    def get_wildtype_observable_positions(self):
        return self.protein_job['protein_wt']['obs_positions']

    def get_wildtype_protein_of(self):
        return self.protein_of['protein_wt']['protein']

    def get_mutate_protein_of(self):
        return self.protein_of['protein_mt']['protein']

    def get_wildtype_protein_of_features(self):
        return self.protein_of['protein_wt']['features']

    def get_mutate_protein_of_features(self):
        return self.protein_of['protein_mt']['features']

    def get_wildtype_protein_of_cycles(self):
        return self.protein_of['protein_wt']['cycles']

    def get_mutate_protein_of_cycles(self):
        return self.protein_of['protein_mt']['cycles']

    def get_wt_aa(self, position, chain, insertion=''):
        protein = self.get_wildtype_protein()
        aa = protein.loc[(protein['residue_number_original'] == position) &
                         (protein['chain_id_original'] == chain) &
                         (protein['insertion'] == insertion),
                         'residue_name'].unique()[0]
        return AA_3_TO_1_DICT[aa]
    
    def get_mut_type(self):
        if len(self.task["mutants"]) > 1:
            return "multi"
        mutant_pos, mutant_aa = next(iter(self.task["mutants"].items()))
        if mutant_aa == "-":
            return "del"
        elif mutant_pos[0] == "-":
            return "ins"
        else:
            return "ss"
    
    def get_array_protein_of_features(self, features_list, features, positions):
        """Turn a dictionary of OF features into `np.array` with sorted positions."""
        array_feats = []

        sorted_positions =  sorted(positions, key=lambda x: positions[x])

        for position_name in sorted_positions:
            # extract position features
            pos_wt_feats = [
                features[position_name][f_name] for f_name in features_list
            ]
            array_feats.append(np.concatenate(pos_wt_feats))
        array_feats = np.array(array_feats)
        return sorted_positions, array_feats

    def get_protein_embeddings(self, features_list, protein_aggregation="mutpos", multi_aggregation="sum"):
        """
        Get protein embeddings according to specified strategy.

        Args:
            features_list: list of str, specifies what features to concatenate \
                            to get each residue's embedding.
            protein_aggregation: str, specifies how protein embeddings will be aggregated. \
                                "mean" for mean of all residue embeddings, \
                                "sum" for sum of all residue embeddings, \
                                "mutpos" for picking based on mutation positions.
            multi_aggregation: str, specifies how multiple mutation embeddings \
                                will be aggregated when `protein_aggregation == "mutpos"`

        Returns:
            Aggregated embeddings for wildtype and mutated protein.
        """
        if protein_aggregation not in self.PROTEIN_AGGREGATION_OPTIONS:
            raise ValueError(
                f"Expected 'protein_aggregation' to be one of "\
                f"{self.PROTEIN_AGGREGATION_OPTIONS}, got {protein_aggregation} instead."
            )

        if multi_aggregation not in self.MULTIPLE_AGGREGATION_OPTIONS:
            raise ValueError(
                f"Expected 'multi_aggregation' to be one of "\
                f"{self.MULTIPLE_AGGREGATION_OPTIONS}, got {multi_aggregation} instead."
            )

        wt_positions, wt_feats = self.get_array_protein_of_features(
            features_list, 
            self.get_wildtype_protein_of_features(),
            self.get_wildtype_observable_positions(),
        )
        mt_positions, mt_feats = self.get_array_protein_of_features(
            features_list, 
            self.get_mutate_protein_of_features(),
            self.get_mutate_observable_positions(),
        )
        
        mut_type = self.get_mut_type()
        if protein_aggregation == "mutpos":
            if mut_type == "multi":
                # make aggregation like sum or mean
                # TODO: add possibility of some of them being indels
                mut_idxs = []
                for mutant_pos, mutant_aa in self.task["mutants"].items():
                    mut_pos_name = f"{mutant_pos[0]}_{mutant_pos[-1]}_{mutant_pos[1]}"
                    mut_idxs.append(wt_positions.index(mut_pos_name))
                if multi_aggregation == "sum":
                    wt_feats = wt_feats[mut_idxs].sum(0)
                    mt_feats = mt_feats[mut_idxs].sum(0)
                else:
                    wt_feats = wt_feats[mut_idxs].mean(0)
                    mt_feats = mt_feats[mut_idxs].mean(0)
            else: # single mutation
                mutant_pos, mutant_aa = next(iter(self.task["mutants"].items()))
                if mut_type == "del": # pick mean of adjacent positions
                    # find indices of adjacent positions
                    mut_pos_name = f"{mutant_pos[0]}_{mutant_pos[-1]}_{mutant_pos[1]}"
                    mut_pos_wt_idx = wt_positions.index(mut_pos_name)
                    wt_adjacent_idxs = []
                    mt_adjacent_idxs = []
                    if mut_pos_wt_idx != 0:
                        wt_adjacent_idxs.append(mut_pos_wt_idx - 1)
                        mt_adjacent_idxs.append(mut_pos_wt_idx - 1)
                    if mut_pos_wt_idx != len(wt_positions) - 1:
                        wt_adjacent_idxs.append(mut_pos_wt_idx + 1)
                        mt_adjacent_idxs.append(mut_pos_wt_idx)
                    
                    # take mean over available adjacent residues
                    wt_feats = wt_feats[wt_adjacent_idxs].mean(0)
                    mt_feats = mt_feats[mt_adjacent_idxs].mean(0)
                elif mut_type == "ins": # pick mean of adjacent positions
                    # find indices of adjacent positions
                    wt_aa = self.get_wt_aa(mutant_pos[1], mutant_pos[-1])
                    mut_pos_name = f"{wt_aa}_{mutant_pos[-1]}_{mutant_pos[1]}"
                    mut_pos_wt_idx = wt_positions.index(mut_pos_name)
                    wt_adjacent_idxs = [mut_pos_wt_idx]
                    mt_adjacent_idxs = [mut_pos_wt_idx]
                    if mut_pos_wt_idx != len(wt_positions) - 1:
                        wt_adjacent_idxs.append(mut_pos_wt_idx + 1)
                        mt_adjacent_idxs.append(mut_pos_wt_idx + 2)
                    
                    # take mean over available adjacent residues
                    wt_feats = wt_feats[wt_adjacent_idxs].mean(0)
                    mt_feats = mt_feats[mt_adjacent_idxs].mean(0)
                elif mut_type == "ss":
                    # pick mutation position embeddings
                    mut_pos_name = f"{mutant_pos[0]}_{mutant_pos[-1]}_{mutant_pos[1]}"
                    mut_pos_wt_idx = wt_positions.index(mut_pos_name)
                    wt_feats = wt_feats[mut_pos_wt_idx]
                    mt_feats = mt_feats[mut_pos_wt_idx]
        elif protein_aggregation == "sum":
            wt_feats = wt_feats.sum(0)
            mt_feats = mt_feats.sum(0)
        else:
            wt_feats = wt_feats.mean(0)
            mt_feats = mt_feats.mean(0)

        return wt_feats, mt_feats

    def get_index_for_observable_position(self, obs_pos, protein_mt, mutations=None):
        new_index = {}
        resi_pos = obs_pos["resi"]
        if mutations is not None:
            pos_del = [m[1] for m in mutations if mutations[m] == '-']
            if resi_pos in pos_del:
                if resi_pos == 1:
                    resi_pos = 2
                else:
                    resi_pos = resi_pos - 1
        # resi_range = list(range(resi_pos,
        #                         resi_pos + obs_pos["shift_from_resi"] + 1))
        protein_mt_ = protein_mt[protein_mt["atom_name"] == "CA"].reset_index(drop=True)
        resi_obs_start = protein_mt_[(protein_mt_["residue_number_original"] == resi_pos) &
                               (protein_mt_["chain_id_original"] == obs_pos["chain_id"]) &
                               (protein_mt_['insertion'] == "")]\
            .index \
            .values

        if len(resi_obs_start) == 0:
            AssertionError(
                "Couldn't find observable residue in muatated protein", obs_pos
            )
        if len(resi_obs_start) != 1:
            AssertionError(
                "Several residues corresponds to the observable residue",
                len(resi_obs_start),
                obs_pos,
            )
            raise
        resi_obs_start = resi_obs_start[0]
        resi_obs = list(range(resi_obs_start,
                              resi_obs_start + obs_pos["shift_from_resi"] + 1))

        for idx in resi_obs:
            original_resi_num = protein_mt_.loc[idx, 'residue_number_original']
            original_resi_ins = protein_mt_.loc[idx, 'insertion']
            original_resi_chain = obs_pos["chain_id"]
            original_aa_wt = self.get_wt_aa(original_resi_num,
                                            original_resi_chain,
                                            original_resi_ins)

            # change position name if current residue was inserted
            is_indel = protein_mt_.loc[idx, "mask"]
            if is_indel:
                pos_id = f'{original_aa_wt}_{original_resi_chain}_{original_resi_num}i'
            else:
                pos_id = f'{original_aa_wt}_{original_resi_chain}_{original_resi_num}{original_resi_ins}'
            new_index[pos_id] = idx
        return new_index

    def evaluate_wildtype_protein(self, of_wrapper, store_of_protein=True):
        if self.get_wildtype_protein().empty:
            self.set_wildtype_protein()
        obs_positions_internal = {}
        if len(self.task["obs_positions"]) == 0:
            self.set_observable_positions()
        for obs_pos in self.task["obs_positions"]:
            obs_index = self.get_index_for_observable_position(obs_pos=obs_pos,
                                                               protein_mt=self.get_wildtype_protein(),
                                                               mutations=self.get_mutations())
            obs_positions_internal.update(obs_index)
        self.set_wildtype_observable_position(obs_positions_internal)

        of_wrapper.evaluate(protein_df=self.get_wildtype_protein())
        of_features_wt = of_wrapper.get_of_features_by_positions(
            self.get_wildtype_observable_positions())

        if store_of_protein:
            of_protein = of_wrapper.get_of_protein()
        else:
            of_protein = pd.DataFrame()
        if of_wrapper.of_output_per_cycle is not None:
            of_cycles = of_wrapper.get_of_features_by_positions_per_cycle(
                self.get_wildtype_observable_positions())
        else:
            of_cycles = {}
        self.set_wildtype_of_protein(of_protein=of_protein,
                                     of_features=of_features_wt,
                                     of_cycles=of_cycles)

    def evaluate_mutate_protein(self, of_wrapper, store_of_protein=True):
        if self.get_mutate_protein().empty:
            self.set_mutate_protein()
        obs_positions_internal = {}
        if len(self.task["obs_positions"]) == 0:
            self.set_observable_positions()
        for obs_pos in self.task["obs_positions"]:
            obs_index = self.get_index_for_observable_position(obs_pos=obs_pos,
                                                               protein_mt=self.get_mutate_protein(),
                                                               mutations=self.get_mutations())
            obs_positions_internal.update(obs_index)
        self.set_mutate_observable_position(obs_positions_internal)
        of_wrapper.evaluate(protein_df=self.get_mutate_protein())
        of_features_mt = of_wrapper.get_of_features_by_positions(
            self.get_mutate_observable_positions())
        if store_of_protein:
            of_protein = of_wrapper.get_of_protein()
        else:
            of_protein = pd.DataFrame()
        if of_wrapper.of_output_per_cycle is not None:
            of_cycles = of_wrapper.get_of_features_by_positions_per_cycle(
                self.get_wildtype_observable_positions())
        else:
            of_cycles = {}
        self.set_mutate_of_protein(of_protein=of_protein,
                                   of_features=of_features_mt,
                                   of_cycles=of_cycles)

    def evaluate(self, of_wrapper,
                 store_of_protein=True):
        self.evaluate_wildtype_protein(of_wrapper=of_wrapper,
                                       store_of_protein=store_of_protein)
        if len(self.get_mutations()) != 0:
            self.evaluate_mutate_protein(of_wrapper=of_wrapper,
                                         store_of_protein=store_of_protein)

    def load_task(self, path):
        # ToDo: check format of json dict
        self.task = json.load(open(path))

    def save_protein_job(self, path):
        pickle.dump(self.protein_job, open(path, 'wb'))

    def save_protein_of_w_features(self, path):
        pickle.dump(self.protein_of, open(path, 'wb'))

    def save(self, path):
        pickle.dump(self, open(path, 'wb'))
