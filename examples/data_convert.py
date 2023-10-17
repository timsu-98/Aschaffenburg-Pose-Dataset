import argparse
from APD.APD import APD
import matplotlib.pyplot as plt
from tqdm import tqdm

from os import listdir
import json
import numpy as np

# from mmaction.datasets import PoseDataset

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
# TODO: find out the actual correspondence
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

actions = ['move', 'start', 'stop', 'turn_left', 'turn_right', 'wait']

joints = [
    'lankle', 'lear', 'lelbow', 'leye', 'lhip', 'lknee', 
    'lshoulder', 'lwrist', 'neck', 'nose', 'rankle', 'rear', 
    'relbow', 'reye', 'rhip', 'rknee', 'rshoulder', 'rwrist', 0, 12
]

#TODO: get image shape and make it a global variable
img_shape = (0, 0)

def convert(apd: APD):
    """Convert APD dataset to PoseDataset format"""
    dataset_size = int(len(apd.data.set))

    # Print out warning for missing data
    for required_element in ('motion_primitives', 'pose2d', 
                             'set', 'timestamps'):
        if required_element not in apd.data.keys():
            print(f'Warning: required element missing: {required_element}')
    
    motion_primitives = apd.data['motion_primitives']
    pose2d = apd.data['pose2d']
    set_of_data = apd.data['set']
    timestamps = apd.data['timestamps']

    # Initialize fields for PoseDataset
    split = dict(
        train = [],
        validation = [],
        test = []
    )
    annotations = []
    
    # Data conversion
    print('Converting data...')
    for i in tqdm(range(dataset_size)):
        # Initialize annotation dict
        annotation = dict(
            frame_dir = i,  # the order of sequences in dataset
            total_frames = None,
            img_shape = img_shape,  # same for all images
            original_shape = img_shape,  # same as img_shape
            label = None,
            keypoint = None,
            keypoint_score = None
        )

        # Length of the sequence
        annotation['total_frames'] = int(len(timestamps[i]))

        # mark the splits
        split[set_of_data[i]].append(i)
        
        # turn one-hot labels from ADP to number label in PoseDataset
        action_scores = []
        for action in actions:
            action_scores.append(np.sum(motion_primitives[action][i]) / annotation['total_frames'])
        annotation['label'] = np.argmax(np.array(action_scores)) # Get the action with the most one-hot labels
        #TODO: turn 2d kyepoint to keypoint and keypoing score

        annotations.append(annotation)

    return dict(
        split = split,
        annotations = annotations
    )

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
