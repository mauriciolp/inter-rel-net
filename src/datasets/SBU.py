import pandas as pd
import numpy as np
import glob

from misc import data_io

DATA_DIR = 'data/sbu/'

""" Folder structure
<set>/
    <action_id>/
        001/
        [002] # not always
        [003] # not always
            depth_...
            rgb_...
            skeleton_pos.txt

Ex: DATA_DIR + '/s01s02/02/001/skeleton_pos.txt'
"""

SETS = ['s01s02','s01s03','s01s07','s02s01','s02s03','s02s06','s02s07','s03s02',
        's03s04','s03s05','s03s06','s04s02','s04s03','s04s06','s05s02','s05s03',
        's06s02','s06s03','s06s04','s07s01','s07s03']

FOLDS = [
    [ 1,  9, 15, 19],
    [ 5,  7, 10, 16],
    [ 2,  3, 20, 21],
    [ 4,  6,  8, 11],
    [12, 13, 14, 17, 18]]

ACTIONS = ['Approaching','Departing','Kicking','Punching','Pushing','Hugging',
           'ShakingHands','Exchanging']

def get_ground_truth(data_dir=DATA_DIR):
    setname_lst, fold_lst, seq_lst, action_lst, path_lst = [], [], [], [], []
    for set_id, set_name in enumerate(SETS):
        for action_id in range(len(ACTIONS)):
            search_exp = '{}/{}/{:02}/*'.format(data_dir, set_name, action_id+1)
            paths = glob.glob(search_exp)
            paths.sort()
            for path in paths:
                seq = path.split('/')[-1]
                fold = np.argwhere([ set_id+1 in lst for lst in FOLDS ])[0,0]
                
                setname_lst.append(set_name)
                fold_lst.append(fold)
                seq_lst.append(seq)
                action_lst.append(action_id)
                path_lst.append(path)
    
    dataframe_dict = {'set_name': setname_lst,
                     'fold': fold_lst,
                     'seq': seq_lst,
                     'path': path_lst,
                     'action': action_lst}
    ground_truth = pd.DataFrame(dataframe_dict)
    return ground_truth

def get_folds():
    folds = np.arange(len(FOLDS))
    
    return folds

def get_train_gt(fold_num):
    if fold_num < 0 or fold_num > 5:
        raise ValueError("fold_num must be within 0 and 5, value entered: "+str(fold_num))
    
    ground_truth = get_ground_truth()
    gt_split = ground_truth[ground_truth.fold != fold_num]
    
    return gt_split

def get_val_gt(fold_num):
    if fold_num < 0 or fold_num > 5:
        raise ValueError("fold_num must be within 0 and 5, value entered: "+str(fold_num))
    
    ground_truth = get_ground_truth()
    gt_split = ground_truth[ground_truth.fold == fold_num]
    
    return gt_split

def get_train(fold_num, **kwargs):
    if fold_num < 0 or fold_num > 5:
        raise ValueError("fold_num must be within 0 and 5, value entered: "+str(fold_num))
    
    ground_truth = get_ground_truth()
    gt_split = ground_truth[ground_truth.fold != fold_num]
    
    X, Y = data_io.get_data(gt_split, pose_style='SBU', **kwargs)
    
    return X, Y
    
def get_val(fold_num, **kwargs):
    if fold_num < 0 or fold_num > 5:
        raise ValueError("fold_num must be within 0 and 5, value entered: "+str(fold_num))
    
    ground_truth = get_ground_truth()
    gt_split = ground_truth[ground_truth.fold == fold_num]
    
    X, Y = data_io.get_data(gt_split, pose_style='SBU', **kwargs)
    
    return X, Y

