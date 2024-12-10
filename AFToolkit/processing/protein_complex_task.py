import pickle

from AFToolKit.processing.protein_task import ProteinTask


class ProteinComplexTask:
    def __init__(self, task=None):
        """
        ProteinComplexTask class contains info about protein and set mutation
        task: json
        """
        self.protein = ProteinTask()
        self.complex = ProteinTask()
        if task is None:
            self.task = {
                "input_protein": {
                    "path": None,  # input protein path
                    "chains_for_protein": None,  # input protein chains
                    "chains_for_complex": None,  # input protein complex chains
                },
                "mutants": {},  # input mutants in format (aa_wt, residue_position, aa_mt, chain_id)
                "obs_positions": [],  # positions to calculate embeddings for in original numbering system
            }
        else:
            self.task = task

    def set_input_protein_task(self, protein_path=None, chains_for_protein=None,
                               chains_for_complex=None, task=None):
        if task is None:
            self.task["input_protein"] = {"path": protein_path,
                                          "chains_for_protein": chains_for_protein,
                                          "chains_for_complex": chains_for_complex}
        else:
            self.task = task

    def get_mutate_protein(self):
        return self.protein.get_mutate_protein()

    def get_wildtype_protein(self):
        return self.protein.get_wildtype_protein()

    def get_wildtype_protein_of(self):
        return self.protein.get_wildtype_protein_of()

    def get_mutate_protein_of(self):
        return self.protein.get_mutate_protein_of()

    def get_wildtype_protein_of_features(self):
        return self.protein.get_wildtype_protein_of_features()

    def get_mutate_protein_of_features(self):
        return self.protein.get_mutate_protein_of_features()

    def get_mutate_complex(self):
        return self.complex.get_mutate_protein()

    def get_wildtype_complex(self):
        return self.complex.get_wildtype_protein()

    def get_wildtype_complex_of(self):
        return self.complex.get_wildtype_protein_of()

    def get_mutate_complex_of(self):
        return self.complex.get_mutate_protein_of()

    def get_wildtype_complex_of_features(self):
        return self.complex.get_wildtype_protein_of_features()

    def get_mutate_complex_of_features(self):
        return self.complex.get_mutate_protein_of_features()

    def get_wt_aa(self, position, chain):
        return self.complex.get_wt_aa(position, chain)

    def set_task_mutants(self, mutants):
        self.task['mutants'] = mutants
        protein_mutations = {
            mut_pos: aa for mut_pos, aa in mutants.items()
            if mut_pos[2] in self.task['input_protein']['chains_for_protein']
        }
        self.protein.set_task_mutants(protein_mutations)
        self.complex.set_task_mutants(mutants)

    def add_mutation(self, chain_id, resi, aa_mt, aa_wt):
        self.task["mutants"][(aa_wt, resi, chain_id)] = aa_mt

        self.protein.add_mutation(chain_id, resi, aa_mt, aa_wt)
        self.complex.add_mutation(chain_id, resi, aa_mt, aa_wt)

    def add_observable_positions(self, resi=None, shift_from_resi=0, chain_id=None, name=None):
        if name is None:
            name = f"{resi}_{chain_id}"
        self.task["obs_positions"].append({"resi": resi,
                                           "shift_from_resi": shift_from_resi,
                                           "chain_id": chain_id,
                                           "name": name})
        if chain_id in self.task['input_protein']['chains_for_protein']:
            self.protein.add_observable_positions(resi, shift_from_resi, chain_id, name)
        self.complex.add_observable_positions(resi, shift_from_resi, chain_id, name)

    def set_wildtype(self):
        """Alias for code compatibility"""
        self.set_wildtype_protein()
    
    def set_wildtype_protein(self):
        # set wildtype for protein
        self.protein.set_input_protein_task(protein_path=self.task['input_protein']['path'],
                                            chains=self.task['input_protein']['chains_for_protein'])
        self.protein.set_task_mutants(self.task['mutants'])
        self.protein.set_wildtype_protein()

        # set wildtype for complex
        self.complex.set_input_protein_task(protein_path=self.task['input_protein']['path'],
                                            chains=self.task['input_protein']['chains_for_complex'])
        self.complex.set_task_mutants(self.task['mutants'])
        self.complex.set_wildtype_protein()

    def set_observable_positions(self, obs_positions=None):
        self.task["obs_positions"] = obs_positions
        if obs_positions is not None:
            protein_obs_positions = [
                pos for pos in obs_positions \
                if pos["chain_id"] in self.task['input_protein']['chains_for_protein']
            ]
        else:
            protein_obs_positions = None
        self.protein.set_observable_positions(protein_obs_positions)
        self.complex.set_observable_positions(obs_positions)

    def set_mutate_protein(self):
        # set mutants for protein
        self.protein.set_mutate_protein()

        # set mutants for complex
        self.complex.set_mutate_protein()

    def evaluate_wildtype_protein(self, of_wrapper, store_of_protein=True):
        self.protein.evaluate_wildtype_protein(
            of_wrapper,
            store_of_protein=store_of_protein
        )
        self.complex.evaluate_wildtype_protein(
            of_wrapper,
            store_of_protein=store_of_protein
        )

    def evaluate_mutate_protein(self, of_wrapper, store_of_protein=True):
        self.protein.evaluate_mutate_protein(
            of_wrapper,
            store_of_protein=store_of_protein
        )
        self.complex.evaluate_mutate_protein(
            of_wrapper,
            store_of_protein=store_of_protein
        )

    def evaluate(self, of_wrapper, store_of_protein=True):
        # evaluate complex and protein separately
        # self.complete_tasks(self.task)
        self.protein.evaluate(of_wrapper,
                              store_of_protein=store_of_protein)
        self.complex.evaluate(of_wrapper,
                              store_of_protein=store_of_protein)

    def save(self, path):
        pickle.dump(self, open(path, 'wb'))
