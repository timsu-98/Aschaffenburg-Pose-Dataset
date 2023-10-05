import argparse
from APD.APD import APD
import matplotlib.pyplot as plt
from tqdm import tqdm

from os import listdir
import json
import numpy as np

import pdb

def main():
    parser = argparse.ArgumentParser(description='Pipeline Arguments')
    parser.add_argument("path", type=str, help="path to json files")
    parser.add_argument('-v', '--vru_types', action='append', type=str,
                        help="select certain vru types for plotting ['ped', 'bike']", default=[])
    parser.add_argument('-s', '--sets', action='append', type=str,
                        help="select certain sets for plotting ['train', 'validation', 'test']", default=[])
    parser.add_argument('-d', '--data-fields',
                        action='append', type=str,
                        help="select data fields ['pose2d', 'pose3d', 'timestamps', 'head_smoothed', 'motion_primitives', 'vru_type', 'set'].",
                        default=['set', 'timestamps']
                        )

    args = parser.parse_args()

    apd = APD(data_path=args.path, vru_types=args.vru_types, sets=args.sets, data_fields=args.data_fields)

    pdb.set_trace()

if __name__ == '__main__':
    main()
