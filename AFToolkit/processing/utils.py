import subprocess

import numpy as np  
import pandas as pd
from functools import reduce
from scipy import stats
from sklearn import metrics

from biopandas.pdb import PandasPdb


AA_3_TO_1_DICT = {"CYS": "C", "ASP": "D", "SER": "S", "GLN": "Q", "LYS": "K", "ILE": "I",
                  "PRO": "P", "THR": "T", "PHE": "F", "ASN": "N", "GLY": "G", "HIS": "H",
                  "LEU": "L", "ARG": "R", "TRP": "W", "ALA": "A", "VAL": "V", "GLU": "E",
                  "TYR": "Y", "MET": "M", "UNK": "U", "-": "-"}

AA_1_TO_3_DICT = {v: k for k, v in AA_3_TO_1_DICT.items()}
DF_COLUMNS = ["record_name", "atom_number", "blank_1", "atom_name", "alt_loc",
               "residue_name", "blank_2", "chain_id", "residue_number", "insertion",
               "blank_3", "x_coord", "y_coord", "z_coord", "occupancy", "b_factor",
               "blank_4", "segment_id", "element_symbol", "charge", "line_idx",
               ]


def get_internal_residue_numbering(pdb_df):
    """
    In case of multichain structure change residue atom numeration.
    All chain id combine in one and between residues add gap of 25 positions.

    Args:
        pdb_df: pandas data frame for original protein structure

    Returns:
         residue_number_af: vector with residue new numbering
    """
    residue_number_af = []
    n_chain = 0
    bias = 0
    for chain, df_ in pdb_df.groupby(["chain_id_original"],
                                     sort=False):
        residue_number_af_ = []
        for ids, df__ in df_.groupby(["residue_number_original", "insertion"],
                                     sort=False):
            if ids[1] != '':
                bias += 1
            residue_number_af__ = df__["residue_number_original"].to_numpy() + bias
            residue_number_af_ += list(residue_number_af__)

        residue_number_af_ -= residue_number_af_[0] - 1 - n_chain
        residue_number_af += list(residue_number_af_)
        n_chain = 25 + residue_number_af[-1]  # add gap between chains
    return residue_number_af


def get_regions_mask(regions, protein):
    conditions = []
    for chain in regions:
        for r in regions[chain]:
            mask = (protein["chain_id_original"] == chain) & \
                   (protein["residue_number_original"].isin(range(r[0], r[1] + 1)))
            conditions.append(mask)
    return reduce(lambda x, y: x | y, conditions)


def load_protein(path, chains=None, regions=None):
    """load protein and change numbering from 1 to N_res
    add 25 residues gap between chains
    input: pdb_path
    output: pdb dataframe prepared for alphafold inference
    """

    pdb_df = PandasPdb().read_pdb(path).label_models().df["ATOM"]
    pdb_df = pdb_df[pdb_df.model_id == 1]
    pdb_df = pdb_df[pdb_df["alt_loc"].isin(["A", ""])]
    pdb_df = pdb_df[pdb_df["element_symbol"] != "H"]
    pdb_df = pdb_df[pdb_df["element_symbol"] != "D"]

    pdb_df["residue_number_original"] = pdb_df["residue_number"]
    pdb_df["chain_id_original"] = pdb_df["chain_id"]
    if chains is not None:
        pdb_df = pdb_df[pdb_df["chain_id_original"].isin(chains)]

    pdb_df["residue_number"] = get_internal_residue_numbering(pdb_df)
    pdb_df["chain_id"] = 'A'
    pdb_df["mask"] = False

    pdb_df = pdb_df[pdb_df["element_symbol"] != "H"]

    if regions is not None:
        combined_condition = get_regions_mask(regions, pdb_df)
        pdb_df = pdb_df[combined_condition]
        pdb_df['residue_number'] = get_internal_residue_numbering(pdb_df)

    return pdb_df


def remove_sidechain_for_mutate_position(protein_wt, mutant_codes):
    if len(mutant_codes) != 0:
        protein_mt = []
        protein_wt["residue_name_wt"] = protein_wt["residue_name"]

        mutant_codes_convert = {(AA_1_TO_3_DICT[wt[0]], wt[1], wt[2]): AA_1_TO_3_DICT[mutant_codes[wt]]
                                for wt in mutant_codes
                                if (wt[0] in AA_1_TO_3_DICT.keys()) and
                                (mutant_codes[wt] in AA_1_TO_3_DICT.keys())}

        for k, df in protein_wt.groupby(
                ["residue_name", "residue_number_original", "chain_id_original"], sort=False
        ):
            df_mutant = df.copy()

            if (k in mutant_codes_convert) and (mutant_codes_convert[k] != "-"):
                """
                AA substituion: keep only backbone atoms from the template and rename residue
                """
                df_mutant = df_mutant[df_mutant["atom_name"].isin(["N", "CA", "C", "O"])]
                assert df_mutant.shape[0] == 4

            protein_mt.append(df_mutant)

        protein_mt_df = pd.concat(protein_mt)
        return protein_mt_df
    else:
        return protein_wt


def insert_protein_region(pdb_df, insertion_residue_position, sequence):
    before_or_after = "after"
    """
    :param pdb_df: pdb dataframe
    :param insert_region: list in format [{"residue_number":resi, "chain_id":chain, "sequence": "GGGGRRRGG",
     "before_or_after":"before"}]
    :return: pdb_df with added dummy atoms
    """
    if isinstance(insertion_residue_position, str):
        ins_residue_number = int(insertion_residue_position[:-1])
        ins_residue_chain = insertion_residue_position[-1]
    else:
        (ins_residue_number, ins_residue_chain) = insertion_residue_position

    pdb_df = pdb_df.copy()

    pdb_resi_insert = pdb_df[
        (pdb_df["residue_number_original"] == ins_residue_number)
        & (pdb_df["chain_id_original"] == ins_residue_chain)
        ]
    if pdb_resi_insert.shape[0] == 0:
        ins_residue_number += 1
        pdb_resi_insert = pdb_df[
            (pdb_df["residue_number_original"] == ins_residue_number)
            & (pdb_df["chain_id_original"] == ins_residue_chain)
            ]
        before_or_after = "before"

    resi = pdb_resi_insert["residue_number"].iloc()[0]
    pdb_df_before = pdb_df[(pdb_df["residue_number"] < resi)]
    pdb_df_after = pdb_df[(pdb_df["residue_number"] > resi)]

    dummy_atoms = []
    for i, s in enumerate(list(sequence)):
        dummy_atom = pdb_resi_insert[pdb_resi_insert["atom_name"] == "CA"].copy()
        dummy_atom["residue_number_original"] = ins_residue_number
        dummy_atom["insertion"] = i
        dummy_atom["chain_id_original"] = ins_residue_chain
        dummy_atom.loc[:, "residue_number"] = i
        dummy_atom["mask"] = True
        dummy_atom["residue_name"] = AA_1_TO_3_DICT[s]
        dummy_atoms.append(dummy_atom)
    dummy_atoms = pd.concat(dummy_atoms)

    if before_or_after == "before":
        dummy_atoms.loc[:, "residue_number"] += resi
        pdb_resi_insert.loc[:, "residue_number"] += len(sequence)
        pdb_df_after.loc[:, "residue_number"] += len(sequence)
        return pd.concat([pdb_df_before, dummy_atoms, pdb_resi_insert, pdb_df_after])
    else:
        dummy_atoms.loc[:, "residue_number"] += resi + 1
        pdb_df_after.loc[:, "residue_number"] += len(sequence)
        return pd.concat([pdb_df_before, pdb_resi_insert, dummy_atoms, pdb_df_after])


def mutate_protein_subs_and_del(protein_wt, mutant_codes, ignore_not_found=False):
    """
    input:
     pdb_df - pandas dataframe
      mutan_codes in format {(aa_before, residue_number, chain_id): aa_after), ...}
    output:
      pdb dataframe for mutant and wt
      for mutant residues only backbone atoms are kept both in mutant and wt dataframes
    """

    """ convert single letter code to three letter code if it is the case """
    mutant_codes_convert = {(AA_1_TO_3_DICT[wt[0]], wt[1], wt[2]): AA_1_TO_3_DICT[mutant_codes[wt]]
                            for wt in mutant_codes
                            if (wt[0] in AA_1_TO_3_DICT.keys()) and
                            (mutant_codes[wt] in AA_1_TO_3_DICT.keys())}

    if ignore_not_found and (len(mutant_codes_convert) != len(mutant_codes)):
        miss = {wt: mutant_codes[wt]
                for wt in mutant_codes
                if (wt[0] not in AA_1_TO_3_DICT.keys()) or
                (mutant_codes[wt] not in AA_1_TO_3_DICT.keys())}
        raise ValueError(f'Mutations was missed or wrong specify: {miss}')

    protein_mt = []
    protein_wt["residue_name_wt"] = protein_wt["residue_name"]
    n_shift = 0

    for k, df in protein_wt.groupby(
            ["residue_name", "residue_number_original", "chain_id_original"], sort=False
    ):
        df_mutant = df.copy()
        df_mutant["residue_number"] += n_shift

        if (k in mutant_codes_convert) and (mutant_codes_convert[k] != "-"):
            """
            AA substituion: keep only backbone atoms from the template and rename residue
            """
            df_mutant = df_mutant[df_mutant["atom_name"].isin(["N", "CA", "C", "O"])]
            assert df_mutant.shape[0] == 4
            df_mutant.loc[:, "residue_name"] = mutant_codes_convert[k]

        if (k in mutant_codes_convert) and (mutant_codes_convert[k] == "-"):
            """
            deletion: ignore the residue and update the numbering
            """
            n_shift -= 1
            continue

        protein_mt.append(df_mutant)

    protein_mt_df = pd.concat(protein_mt)
    return protein_mt_df


def mutate_protein(pdb_df, mutant_codes, ignore_not_found=False):
    """
    function split mutant_codes to (1) single amino acid mutants & deletions (2) insertions
    input:
     pdb_df - pandas dataframe
      mutan_codes in format {(aa_before, residue_number, chain_id): aa_after), ...}
    output:
      pdb dataframe for mutant and wt
      for mutant residues only backbone atoms are kept both in mutant and wt dataframes
    """
    mutant_codes_subs = {k: mutant_codes[k] for k in mutant_codes if k[0] != "-"}
    mutant_codes_ins = {k: mutant_codes[k] for k in mutant_codes if k[0] == "-"}

    if len(mutant_codes_subs) != 0:
        pdb_df = mutate_protein_subs_and_del(pdb_df, mutant_codes_subs, ignore_not_found)

    if len(mutant_codes_ins) != 0:
        for wt in mutant_codes_ins:
            seq = mutant_codes_ins[wt]
            resi = wt[1]
            chain = wt[2]
            pdb_df = insert_protein_region(pdb_df, (resi, chain), seq)
    return pdb_df


def pdbline_to_dict(line):
    """
    :param line: string from PDB file (starts with ATOM ...) -> string
    """
    atom_name = line[13:15]
    if line[15] != " ":
        atom_name += line[15]
    residue_name = line[17:20]
    chain_name = line[21]
    residue_number = int(line[22:26])
    x, y, z = line[30:38], line[38:46], line[46:54]
    if x[0] != " ":
        x = x[1:]
    if y[0] != " ":
        y = y[1:]
    if z[0] != " ":
        z = z[1:]
    (x, y, z) = (float(x), float(y), float(z))
    if len(line) >= 77:
        atom_type = line[77]
    else:
        atom_type = atom_name[0]
    b = float(line[61:66])
    atom_number = int(line[4:11])
    df_ = {c: "" for c in DF_COLUMNS}
    df_["atom_number"] = atom_number
    df_["record_name"] = "ATOM"
    df_["atom_name"] = atom_name
    df_["residue_name"] = residue_name
    df_["aa"] = AA_3_TO_1_DICT[residue_name]
    df_["chain_id"] = chain_name
    df_["residue_number"] = residue_number
    df_["x_coord"] = x
    df_["y_coord"] = y
    df_["charge"] = 0
    df_["z_coord"] = z
    df_["occupancy"] = 1.0  # 1.0
    df_["b_factor"] = float(b)  # 105.55
    df_["element_symbol"] = atom_type
    return df_


def pdb_str_to_dataframe(pdb_lines, pdb_df_prev=None):
    """
    :param pdb_lines: alphafold predictions
    :param pdb_df_prev: dataframe that contains extra columns, e.g. original_numbering
    :return:
    """
    if isinstance(pdb_lines, str):
        pdb_lines = pdb_lines.split("\n")
    pdb_df = []
    for line in pdb_lines:
        if not line.startswith("ATOM"):
            continue
        pdb_df.append(pdbline_to_dict(line))
    pdb_df = pd.DataFrame(pdb_df)
    pdb_df = pdb_df.reindex(columns=DF_COLUMNS)
    pdb_df["line_idx"] = pdb_df.index
    if pdb_df_prev is None:
        return pdb_df

    original_numbering = {
        k: (
            r.iloc()[0]["residue_number_original"],
            r.iloc()[0]["chain_id_original"],
            r.iloc()[0]["insertion"],
        )
        for k, r in pdb_df_prev.groupby(["residue_number", "chain_id"], sort=False)
    }

    pdb_df["residue_number_original"] = [
        original_numbering[(r["residue_number"], r["chain_id"])][0]
        for r in pdb_df.iloc()
    ]
    pdb_df["chain_id_original"] = [
        original_numbering[(r["residue_number"], r["chain_id"])][1]
        for r in pdb_df.iloc()
    ]
    pdb_df["insertion"] = [
        original_numbering[(r["residue_number"], r["chain_id"])][2]
        for r in pdb_df.iloc()
    ]

    return pdb_df


def runcmd(cmd, verbose=False, *args, **kwargs):

    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        shell=True
    )
    std_out, std_err = process.communicate()
    if verbose:
        print(std_out.strip(), std_err)
    pass


def save_to_pdb(pdb_df, output_name, original_numbering=True):
    """
    :param pdb_df: pdb dataframe
    :param output_name: output pdb path
    :return:
    """
    prot = PandasPdb()
    pdb_df = pdb_df.copy()
    if original_numbering:
        pdb_df["residue_number"] = pdb_df["residue_number_original"]
        pdb_df["chain_id"] = pdb_df["chain_id_original"]
    prot.df["ATOM"] = pdb_df
    prot.to_pdb(output_name)
