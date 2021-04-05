import pandas as pd
import numpy as np
import os
import random
import glob

from misc import data_io

DATA_DIR = 'data/YMJA/'

""" Folder structure
action/
    clip0_positions.json
    clip1_positions.json
    clip2_positions.json
        
Ex: DATA_DIR + 'Tripping/_2017-11-06-det-van-home15.json'
"""

FOLD_MECH = 'Uniform' # Can either be Random or Uniform
NUM_FOLDS = 5
ACTIONS = ['No_penalty', 'Holding', 'Hooking', 'Slashing', 'Tripping'] # ['No_penalty', 'Cross_Checking', 'Hi_sticking', 'Holding', 'Hooking', 'Interference', 'Roughing', 'Slashing', 'Tripping']

# Determine folds once for entire directory
FOLDS = []
FILES = []
nextint = 0
for subdir, dirs, files in os.walk(DATA_DIR):
    for file in files:
        if file.endswith(".json"):
            FILES.append(os.path.join(subdir, file))
            if FOLD_MECH == 'Random':
                FOLDS.append(random.randint(0, NUM_FOLDS))
            else: # Uniform distribution
                FOLDS.append(nextint)
                nextint = (nextint + 1) % NUM_FOLDS

def get_ground_truth(data_dir=DATA_DIR):
    action_lst = []
    for file in FILES:
        penalty_class = file.split("/")[2]
        action_lst.append(ACTIONS.index(penalty_class))
    
    dataframe_dict = {'fold': FOLDS,
                     'path': FILES,
                     'action': action_lst}

    ground_truth = pd.DataFrame(dataframe_dict)
    return ground_truth

def get_folds():
    folds = np.arange(NUM_FOLDS)
    
    return folds

def get_train_gt(fold_num):
    if fold_num < 0 or fold_num > NUM_FOLDS:
        raise ValueError("fold_num must be within 0 and " + NUM_FOLDS + ", value entered: "+str(fold_num))
    
    ground_truth = get_ground_truth()
    gt_split = ground_truth[ground_truth.fold != fold_num]
    
    return gt_split

def get_val_gt(fold_num):
    if fold_num < 0 or fold_num > NUM_FOLDS:
        raise ValueError("fold_num must be within 0 and " + NUM_FOLDS + ", value entered: "+str(fold_num))
    
    ground_truth = get_ground_truth()
    gt_split = ground_truth[ground_truth.fold == fold_num]
    
    return gt_split

def get_train(fold_num, **kwargs):
    if fold_num < 0 or fold_num > NUM_FOLDS:
        raise ValueError("fold_num must be within 0 and " + NUM_FOLDS + ", value entered: "+str(fold_num))
    
    ground_truth = get_ground_truth()
    gt_split = ground_truth[ground_truth.fold != fold_num]
    
    X, Y = data_io.get_data(gt_split, pose_style='YMJA', **kwargs)
    
    return X, Y
    
def get_val(fold_num, **kwargs):
    if fold_num < 0 or fold_num > NUM_FOLDS:
        raise ValueError("fold_num must be within 0 and " + NUM_FOLDS + ", value entered: "+str(fold_num))
    
    ground_truth = get_ground_truth()
    gt_split = ground_truth[ground_truth.fold == fold_num]
    
    X, Y = data_io.get_data(gt_split, pose_style='YMJA', **kwargs)
    
    return X, Y

