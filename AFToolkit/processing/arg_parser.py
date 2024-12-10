import argparse


def parser():
    parser_ = argparse.ArgumentParser(add_help=False)
    parser_.add_argument('--pdb', help='Path to protein structure in PDB format')
    parser_.add_argument('--mutations', help='List of mutations in format <chain>:<AA_wt><pos><AA_mt>.'
                                             'Example A:D123C, A:R56Y',
                         default=None)
    parser_.add_argument('--positions', help='List of positions to calculate AlphaFold features '
                                             'and embeddings in format <chain>:<pos>.'
                                             'Example A:123, A:56',
                         default=None)
    parser_.add_argument('-o', '--output-dir', help='Output directory for files in pkl format',
                         default='./')
    parser_.add_argument('--input', '-i', help='Path to file with tasks in pkl format')
    parser_.add_argument('--store-protein', help='Flag to store AlphaFold relax protein structure',
                         action=argparse.BooleanOptionalAction)
    parser_.add_argument('--return-all-cycles', help='Flag to store AlphaFold relax protein structure',
                         action=argparse.BooleanOptionalAction, default=False)
    parser_.add_argument('-d', '--device', help='Device for code evaluating',
                         default='cuda:0')
    parser_.add_argument('-n', '--inference-n-recycle', help='Number of AF cycles',
                         default=1, type=int)
    parser_.add_argument('--always-use-template', help='Flag indicated usage of template for recylingn start with n=1',
                         action=argparse.BooleanOptionalAction, default=False)
    parser_.add_argument('--side-chain-mask', help='Flag indicated mask all side chain atoms ("gap" pipeline)',
                         action=argparse.BooleanOptionalAction, default=False)
    return parser_


def parse_mutations(mutations):
    mutations_converted = {}
    if mutations is not None:
        for mut in mutations.split(','):
            mut = mut.strip()
            chain = mut.split(':')[0]
            pos = int(mut.split(':')[1][1:-1])
            aa_wt = mut.split(':')[1][0]
            aa_mt = mut.split(':')[1][-1]
            mutations_converted[(aa_wt, pos, chain)] = aa_mt
    return mutations_converted


def parse_positions(pos):
    pos_list = []
    if pos is not None:
        for p in pos.split(','):
            p = p.strip()
            chain = p.split(':')[0]
            resi = int(p.split(':')[1])
            pos_list.append({"resi": resi, "shift_from_resi": 0, "chain_id": chain})
    return pos_list
