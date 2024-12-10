import logging
import os
import pickle
import warnings
import argparse

import pandas as pd

from AFToolKit.processing.protein_task import ProteinTask
from AFToolKit.processing.openfold_wrapper import OpenFoldWrapper
from AFToolKit.processing.arg_parser import parser, parse_mutations, parse_positions

warnings.filterwarnings("ignore")
logger = logging.getLogger(__name__)


def protein_task_parser():
    parent_parser = parser()
    parser_ = argparse.ArgumentParser(parents=[parent_parser])
    parser_.add_argument('--chain', help='Chain of protein',
                         default=None)
    # parser_.add_argument('--regions', help='Regions of protein in format {"A": [(1, 36)]}',
    #                      default=None)
    return parser_


def prepare_task(df):
    df.reset_index(drop=True)
    task_dict = {}
    if 'positions' not in df.columns:
        df.loc[:, 'positions'] = None
    if 'mutations' not in df.columns:
        df.loc[:, 'mutations'] = None
    for i in df.index:
        row = df.loc[i]
        task = {
            "input_protein": {
                "path": row['path'],
                "chains": row['chain'].split(','),
                "regions": None,
            },
            "mutants": parse_mutations(row['mutations']),
            "obs_positions": parse_positions(row['positions'])
        }
        task_dict[f"{row['pdb_id']}_{row['chain']}_{row['mut_info']}"] = task

    return task_dict


def main():
    args = protein_task_parser().parse_args()
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
        protein_task = ProteinTask()
        protein_task.set_input_protein_task(protein_path=args.pdb,
                                            chains=args.chain.split(','))
        protein_task.set_task_mutants(parse_mutations(args.mutations))
        protein_task.set_observable_positions(parse_positions(args.positions))
        logger.info('Calculate embeddings')
        protein_task.evaluate(of_wrapper=of_wrapper,
                              store_of_protein=args.store_protein
                              )

        logger.info('Save results')
        protein_task.save(f"{args.output_dir}/{args.pdb}_{args.chain}_{args.mutations}.pkl")

    else:
        protein_tasks = {}
        for task_id in tasks:
            protein_task = ProteinTask()
            try:
                print(task_id)
                protein_task.set_input_protein_task(task=tasks[task_id])
                logger.info('Calculate embeddings')
                protein_task.evaluate(of_wrapper=of_wrapper,
                                      store_of_protein=args.store_protein)
                # protein_tasks[task_id] = protein_task
                pickle.dump(protein_task,
                            open(f"{args.output_dir}/{task_id}.pkl", 'wb'))
            except Exception as e:
                print(task_id, e)
                continue

            # pickle.dump(protein_tasks, open(args.output, 'wb'))


if __name__ == "__main__":
    main()
