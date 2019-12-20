import pandas as pd
import numpy as np

from misc import data_io

DATA_DIR = 'data/ut-interaction/'

""" Folder structure
<'set1' or 'set2'>/keypoints
    <video_name>/
        <video_name>_<frame_num>_keypoints.json
        ...

Ex: DATA_DIR + 'set1/keypoints/0_1_4/0_1_4_000000000042_keypoints.json'
"""

VIDEOS = [
    ['0_1_4','1_1_2','2_1_1','3_1_3','4_1_0','5_1_5','6_2_4','7_2_5','8_2_0',
    '9_2_2','10_2_1','11_2_3','12_3_4','13_3_2','14_3_1','15_3_3','16_3_5',
    '17_3_0','18_4_4','19_4_1','20_4_2','21_4_0','22_4_3','23_4_5','24_5_0',
    '25_5_4','26_5_2','27_5_1','28_5_3','29_5_5','30_6_2','31_6_5','32_6_1',
    '33_6_3','34_6_0','35_7_0','36_7_5','37_7_4','38_7_2','39_7_3','40_7_1',
    '41_8_0','42_8_2','43_8_4','44_8_4','45_8_5','46_8_3','47_8_1','48_9_3',
    '49_9_5','50_9_2','51_9_4','52_9_0','53_9_1','54_10_0','55_10_4','56_10_5',
    '57_10_3','58_10_1','59_10_2'], #set1
    ['0_11_4','1_11_2','2_11_5','3_11_0','4_11_3','5_11_1','6_12_0','7_12_3',
    '8_12_5','9_12_1','10_12_4','11_12_2','12_13_4','13_13_2','14_13_1',
    '15_13_3','16_13_5','17_13_0','18_14_0','19_14_1','20_14_5','21_14_3',
    '22_14_4','23_14_2','24_15_1','25_15_0','26_15_4','27_15_2','28_15_3',
    '29_15_5','30_16_3','31_16_0','32_16_1','33_16_4','34_16_2','35_16_5',
    '36_17_1','37_17_0','38_17_3','39_17_5','40_17_4','41_17_2','42_18_2',
    '43_18_4','44_18_1','45_18_3','46_18_5','47_18_0','48_19_0','49_19_1',
    '50_19_4','51_19_3','52_19_5','53_19_2','54_20_1','55_20_0','56_20_5',
    '57_20_3','58_20_4','59_20_2']  #set2
]

ACTIONS = ['Hand Shaking','Hugging','Kicking','Pointing','Punching','Pushing']

def get_ground_truth(data_dir=DATA_DIR):
    video_lst, setid_lst, seq_lst, path_lst, action_lst = [], [], [], [], []
    for set_id, set_videos in enumerate(VIDEOS):
        video_lst = video_lst + set_videos
        setid_lst = setid_lst + len(set_videos)*[set_id+1]
        for video in set_videos:
            num, seq, action = video.split('_')
            seq_lst.append(int(seq))
            action_lst.append(int(action))
            path = '{}set{}/keypoints/{}/'.format(data_dir, set_id+1, video)
            path_lst.append(path)

    dataframe_dict = {'video_id': video_lst,
                     'setid': setid_lst,
                     'seq': seq_lst,
                     'path': path_lst,
                     'action': action_lst}
    ground_truth = pd.DataFrame(dataframe_dict).set_index('video_id')
    
    return ground_truth

def get_folds(setid):
    if setid == 1:
        folds = np.arange(10)
    elif setid == 2:
        folds = np.arange(10, 20)
    else:
        raise ValueError("setid must be 1 or 2, value entered: "+str(setid))
    
    return folds

def get_train_gt(fold_num):
    if fold_num < 0 or fold_num > 19:
        raise ValueError("fold_num must be within 0 and 19, value entered: "+str(fold_num))
    if fold_num < 10:
        setid = 1
        sequences = np.arange(10)
        fold_sequences = sequences[sequences != fold_num] + 1
    else:
        setid = 2
        sequences = np.arange(10, 20)
        fold_sequences = sequences[sequences != fold_num] + 1
    
    ground_truth = get_ground_truth()
    gt_split = ground_truth[ground_truth.setid == setid]
    gt_split = gt_split[gt_split.seq.isin(fold_sequences)]
    
    return gt_split

def get_val_gt(fold_num):
    if fold_num < 0 or fold_num > 19:
        raise ValueError("fold_num must be within 0 and 19, value entered: "+str(fold_num))
    if fold_num < 10:
        setid = 1
        sequences = np.arange(10)
        fold_sequences = sequences[sequences == fold_num] + 1
    else:
        setid = 2
        sequences = np.arange(10, 20)
        fold_sequences = sequences[sequences == fold_num] + 1
    
    ground_truth = get_ground_truth()
    gt_split = ground_truth[ground_truth.setid == setid]
    gt_split = gt_split[gt_split.seq.isin(fold_sequences)]
    
    return gt_split

def get_train(fold_num, **kwargs):
    if fold_num < 0 or fold_num > 19:
        raise ValueError("fold_num must be within 0 and 19, value entered: "+str(fold_num))
    if fold_num < 10:
        setid = 1
        sequences = np.arange(10)
        fold_sequences = sequences[sequences != fold_num] + 1
    else:
        setid = 2
        sequences = np.arange(10, 20)
        fold_sequences = sequences[sequences != fold_num] + 1
    
    return get_seqs(setid, fold_sequences, **kwargs)

def get_val(fold_num, **kwargs):
    if fold_num < 0 or fold_num > 19:
        raise ValueError("fold_num must be within 0 and 19, value entered: "+str(fold_num))
    if fold_num < 10:
        setid = 1
        sequences = np.arange(10)
        fold_sequences = sequences[sequences == fold_num] + 1
    else:
        setid = 2
        sequences = np.arange(10, 20)
        fold_sequences = sequences[sequences == fold_num] + 1
    
    return get_seqs(setid, fold_sequences, **kwargs)

def get_seqs(setid, selected_sequences, **kwargs):
    if setid < 1 or setid > 2:
        raise ValueError("setid must be 1 or 2, value entered: "+str(setid))

    ground_truth = get_ground_truth()
    gt_split = ground_truth[ground_truth.setid == setid]
    
    gt_split = gt_split[gt_split.seq.isin(selected_sequences)]
    
    X, Y = data_io.get_data(gt_split, pose_style='OpenPose', **kwargs)
    
    return X, Y
