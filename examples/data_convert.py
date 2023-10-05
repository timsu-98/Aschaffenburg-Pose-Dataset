import argparse
from APD.APD import APD
import matplotlib.pyplot as plt
from tqdm import tqdm

from os import listdir
import json
import numpy as np

from mmaction.datasets import PoseDataset

import mmengine

import pdb

# Map for motion primitives
mp_map = {
    'wait':         0,
    'start':        1,
    'move':         2,
    'stop':         3,
    'turn_left':    4,
    'turn_right':   5
}

# Map for keypoint correspondence
kp_map = {
    0: 0,
    1: 1,
    2: 2,
    3: 3,
    4: 4,
    5: 5,
    6: 6,
    7: 7,
    8: 8,
    9: 9,
    10: 10,
    11: 11,
    12: 12,
    13: 13,
    14: 14,
    15: 15,
    16: 16,
}

#TODO: get image shape and make it a global variable
img_shape = (0, 0)

def convert(apd: APD):
    """Convert APD dataset to PoseDataset format"""
    dataset_length = int(len(apd.data.set))
    for i in tqdm(range(dataset_length)):
        pass
        #TODO: number frame_dir with number in the list
        #TODO: mark the splits
        #TODO: turn one-hot labels from ADP to number label in PoseDataset
        #TODO: turn 2d kyepoint to keypoint and keypoing score
    return None

def main():
    parser = argparse.ArgumentParser(description='Pipeline Arguments')
    parser.add_argument("path", type=str, help="path to json files")

    parser.add_argument('-d', '--data-fields',
                        action='append', type=str,
                        help="select data fields ['pose2d', 'pose3d', 'timestamps', 'head_smoothed, 'motion_primitives'].",
                        default=['set', 'timestamps', 'pose2d', 'motion_primitives']
                        )
    parser.add_argument(
        'output',
        type=str,
        nargs='?',
        default='data/new_dataset.pkl',
        help='The output file for converted dataset.'
    )

    args = parser.parse_args()

    apd = APD(
        data_path=args.path, 
        vru_types=['bike'], 
        sets=['train', 'validation', 'test'],
        data_fields=args.data_fields
    )

    new_data_list = convert(apd)

    mmengine.dump(new_data_list, args.output)

if __name__ == '__main__':
    main()
