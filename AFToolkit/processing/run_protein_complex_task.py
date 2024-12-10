import logging
import os
import pickle
import argparse

import pandas as pd

from AFToolKit.processing.protein_complex_task import ProteinComplexTask
from AFToolKit.processing.openfold_wrapper import OpenFoldWrapper
from AFToolKit.processing.arg_parser import parser, parse_mutations, parse_positions

logger = logging.getLogger(__name__)
COLUMNS_INPUT = ['pdb_id', 'chains_for_protein', 'mutations', 'positions']


def protein_complex_task_parser():
    parent_parser = parser()
    parser_ = argparse.ArgumentParser(parents=[parent_parser])
    parser_.add_argument('--chains-for-complex', help='Chains of protein complex',
                         default=None)
    parser_.add_argument('--chains-for-protein', help='Chain of protein',
                         default=None)
    return parser_


def prepare_task(df):
    df.reset_index(drop=True)
    task_dict = {}
    if 'chains_for_complex' not in df.columns:
        df.loc[:, 'chains_for_complex'] = None
    else:
        df.loc[:, 'chains_for_complex'] = df['chains_for_complex'].apply(lambda s: s.split(','))
    if 'positions' not in df.columns:
        df.loc[:, 'positions'] = None
    if 'mutations' not in df.columns:
        df.loc[:, 'mutations'] = None
    for i in df.index:
        row = df.loc[i]
        task = {
            "input_protein": {
                "path": row['path'],
                "chains_for_protein": row['chains_for_protein'].split(','),
                "chains_for_complex": row['chains_for_complex'],
            },
            "mutants": parse_mutations(row['mutations']),
            "obs_positions": parse_positions(row['positions']),
        }
        task_dict[f"{row['pdb_id']}_{row['chains_for_protein']}_{row['mut_info']}"] = task

    return task_dict


def main():
    args = protein_complex_task_parser().parse_args()
    if args.input:
        task_table = pd.read_csv(args.input)
        tasks = prepare_task(task_table)
    elif args.pdb:
        if not os.path.exists(args.pdb):
            logger.exception("PDB file does not exists")
            raise
    else:
        logger.exception("Path to protein structure in pdb format or task in csv format is required")
        raise
    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)

    logger.info('Set OpenFold wrapper')
    of_wrapper = OpenFoldWrapper(device=args.device,
                                 inference_n_recycle=args.inference_n_recycle,
                                 always_use_template=args.always_use_template,
                                 side_chain_mask=args.side_chain_mask,
                                 return_all_cycles=args.return_all_cycles)
    of_wrapper.init_model()

    if args.pdb:
        protein_complex_task = ProteinComplexTask()
        if args.chains_for_protein:
            chains_for_protein = args.chains_for_protein.split(',')
        else:
            logging.exception('--chains-for-protein argument is required')
        if args.chains_for_complex:
            chains_for_complex = args.chains_for_complex.split(',')
        else:
            chains_for_complex = None
        protein_complex_task.set_input_protein_task(protein_path=args.pdb,
                                                    chains_for_protein=chains_for_protein,
                                                    chains_for_complex=chains_for_complex,
                                                    )
        protein_complex_task.set_task_mutants(parse_mutations(args.mutations))
        protein_complex_task.set_observable_positions(parse_positions(args.positions))
        protein_complex_task.set_wildtype()
        protein_complex_task.set_mutate_protein()
        logger.info('Calculate embeddings')
        protein_complex_task.evaluate(of_wrapper=of_wrapper,
                                      store_of_protein=args.store_protein)

        logger.info('Save results')
        protein_complex_task.save(f"{args.output_dir}/{args.pdb}_{args.chain}_{args.mutations}.pkl")

    else:
        # protein_complex_tasks = {}
        for task_id in tasks:
            protein_complex_task = ProteinComplexTask()
            try:
                print(task_id)
                protein_complex_task.set_input_protein_task(task=tasks[task_id])
                protein_complex_task.set_observable_positions(tasks[task_id]["obs_positions"])
                logger.info('Calculate embeddings')
                protein_complex_task.set_wildtype()
                protein_complex_task.set_mutate_protein()
                protein_complex_task.evaluate(of_wrapper=of_wrapper,
                                              store_of_protein=args.store_protein)
                pickle.dump(protein_complex_task,
                            open(f"{args.output_dir}/{task_id}.pkl", 'wb'))
                # protein_complex_tasks[task_id] = protein_complex_task
            except Exception as e:
                print(task_id, e)
                continue

            # pickle.dump(protein_complex_tasks, open(args.output, 'wb'))


if __name__ == "__main__":
    main()
