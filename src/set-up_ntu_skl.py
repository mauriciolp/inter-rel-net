import numpy as np
import pandas as pd
import os, argparse
import progressbar
from zipfile import ZipFile
from shutil import copyfile

from datasets import NTU, NTU_V2

np.seterr(divide='ignore', invalid='ignore')

def load_args():
    ap = argparse.ArgumentParser(
        description=('Set-up NTU datasets. '+
            'Read skeletons from zip file and save all info to a csv file.'),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    
    ap.add_argument('dataset_version',
        help='Version of the dataset to set-up. V2 requires V1 first.',
        choices=['1','2'])
    
    ap.add_argument('-o','--overwrite',
        help="Overwrite existing files",
        action='store_true')
    
    ap.add_argument('-c','--convert',
        help="Convert csv file to numpy format (faster reading).",
        action='store_true')
    
    args = ap.parse_args()
    
    return args

def parse_ntu_skeleton(skl_file):
    frame_num = int(skl_file.readline().strip())
    video_poses = []
    for frame_id in range(frame_num):
        num_people = int(skl_file.readline().strip())
        persons = []
        for person_id in range(num_people):
            values = skl_file.readline().strip().split()
            keys = ['bodyID','clipedEdges','handLeftConfidence',
                'handLeftState','handRightConfidence','handRightState',
                'isResticted','leanX','leanY','trackingState','jointCount']
            person_info = dict(zip(keys, values))
            
            num_joints = int(skl_file.readline().strip())
            joints = []
            for joint_id in range(num_joints):
                values = skl_file.readline().strip().split()
                joint_info = {
                    'x': float(values[0]),
                    'y': float(values[1]),
                    'z': float(values[2]),
                    'depthX': float(values[3]),
                    'depthY': float(values[4]),
                    'colorX': float(values[5]),
                    'colorY': float(values[6]),
                    'orientationW': float(values[7]),
                    'orientationX': float(values[8]),
                    'orientationY': float(values[9]),
                    'orientationZ': float(values[10]),
                    'trackingState': values[11],
                }
                joints.append(joint_info)
            person_info['joints'] = joints
            persons.append(person_info)
        video_poses.append(persons)
    return video_poses

def prune_video_poses(video_poses):
    
    frame_pose = video_poses[0]
    
    all_mid_coords = []
    central_dists = []
    for person_pose in frame_pose:
        mid_joint = person_pose['joints'][1] # Middle_of_the_spine
        mid_coords = [mid_joint['x'], mid_joint['y']] # only use x and y
        all_mid_coords.append(mid_coords)
        central_dists.append(np.linalg.norm(mid_coords))
    
    to_keep = sorted(np.argsort(central_dists)[:2])
    
    # pruned_video_poses = [ [x[i] for i in to_keep] for x in video_poses ]
    pruned_video_poses = []
    for frame_pose in video_poses:
        if len(frame_pose) > 2:
            pruned_video_poses.append([frame_pose[i] for i in to_keep])
        else:
            pruned_video_poses.append(frame_pose)
    
    return pruned_video_poses

def flatten_video_poses(video_poses):
    flat_video_poses = []
    if np.any([len(frame) > 2 for frame in video_poses]):
        video_poses = prune_video_poses(video_poses)
    for frame_pose in video_poses:
        persons_coords = []
        for person_pose in frame_pose:
            joints_coords = []
            for joint in person_pose['joints']:
                joint_coords = [joint['x'], joint['y'], joint['z']]
                joints_coords += joint_coords
            persons_coords += joints_coords
        if len(frame_pose) == 0: # 0 persons, need to pad with zeroes
            persons_coords += 150*[0]
        elif len(frame_pose) < 2: # only one person, need to pad with zeroes
            persons_coords += 75*[0]
        flat_video_poses.append(persons_coords)
    return flat_video_poses

def apply_NTU_normalization(video_poses):
    joint_indexing = ['Base_of_the_spine', 'Middle_of_the_spine', 'Neck', 
      'Head', 'Left_shoulder', 'Left_elbow', 'Left_wrist', 'Left_hand', 
      'Right_shoulder', 'Right_elbow', 'Right_wrist', 'Right_hand', 'Left_hip', 
      'Left_knee', 'Left_ankle', 'Left_foot', 'Right_hip', 'Right_knee', 
      'Right_ankle', 'Right_foot', 'Spine', 'Tip_of_the_left_hand', 
      'Left_thumb', 'Tip_of_the_right_hand', 'Right_thumb']
    
    middle_joint_idx = joint_indexing.index('Middle_of_the_spine')
    spine_base_joint_idx = joint_indexing.index('Base_of_the_spine')
    spine_joint_idx = joint_indexing.index('Spine')
    left_shoulder_joint_idx = joint_indexing.index('Left_shoulder')
    right_shoulder_joint_idx = joint_indexing.index('Right_shoulder')
    
    reshaped_video_poses = np.reshape(video_poses, (-1,2,25,3))
    
    normalized_video_poses = []
    for frame_idx, frame_poses in enumerate(reshaped_video_poses):
        p1_coords = frame_poses[0]
        p1_middle_joint = p1_coords[middle_joint_idx]
        p1_spine_base_joint = p1_coords[spine_base_joint_idx]
        p1_spine_joint = p1_coords[spine_joint_idx]
        p1_left_shoulder_joint = p1_coords[left_shoulder_joint_idx]
        p1_right_shoulder_joint = p1_coords[right_shoulder_joint_idx]
        
        new_origin = p1_middle_joint
        # scale_val = .5 / np.linalg.norm(p1_spine_base_joint - p1_spine_joint)
        
        y = p1_spine_joint - p1_spine_base_joint
        y = y/np.linalg.norm(y)
        
        x = p1_left_shoulder_joint - p1_right_shoulder_joint
        x = x/np.linalg.norm(x)
        
        z = np.cross(x, y)
        z = z/np.linalg.norm(z)
        
        x = np.cross(y,z)
        x = x/np.linalg.norm(x)
        
        rotation_matrix = np.array([x,y,z])
        
        normalized_frame_poses = []
        for person in frame_poses:
            if np.count_nonzero(person) > 0: # Checking if it is not a dummy
                translated = person - new_origin
                rotated = np.dot(rotation_matrix, translated.T).T
                scaled = rotated# * scale_val
            else:
                scaled = person
            
            normalized_frame_poses.append(scaled)
        
        if np.isnan(normalized_frame_poses).any():
            if normalized_video_poses == []:
                normalized_frame_poses = np.zeros_like(normalized_frame_poses)
            else:
                normalized_frame_poses = normalized_video_poses[-1]
        normalized_video_poses.append(normalized_frame_poses)
    
    normalized_video_poses = np.reshape(normalized_video_poses, (-1,150))
    return normalized_video_poses

def parse_videoname(videoname):
    setup = int(videoname[1:4])
    cam = int(videoname[5:8])
    subj = int(videoname[9:12])
    dup = int(videoname[13:16])
    act = int(videoname[17:20])
    return setup, cam, subj, dup, act

if __name__ == '__main__':
    args = load_args()
    
    print('> Starting Set-up NTU Dataset')
    
    dataset_version = args.dataset_version
    overwrite = args.overwrite
    convert = args.convert
    
    print('\tDataset Version:', dataset_version)
    print('\tOverwrite skl.csv and descs.csv:', overwrite)
    print('\tConvert skl.csv to skl.npy:', convert)
    
    if not convert:
        if dataset_version == '1':
            zip_filepath = os.path.join(NTU.DATA_DIR, 'nturgbd_skeletons.zip')
            out_filepath = os.path.join(NTU.DATA_DIR, 'skl.csv')
            
            if os.path.exists(out_filepath):
                if overwrite:
                    os.remove(out_filepath)
                else:
                    raise Exception("Output file already exists, use --overwrite option. File: "+out_filepath)
            
            descs_data = []
            line_pt = 1
            
            print("Reading skeletons from zipfile:", zip_filepath)
            print("Saving normalized skeletons at:", out_filepath)
            print("Reading, normalizing and saving...")
            with ZipFile(zip_filepath) as zfile, open(out_filepath, 'ab') as out_file:
                filenames_list = sorted([file_info.filename for file_info in zfile.infolist()])
                if filenames_list[0] == 'nturgb+d_skeletons/':
                    filenames_list.pop(0)
                
                progbar = progressbar.ProgressBar(max_value=len(filenames_list))
                for idx, filename in enumerate(filenames_list):
                    progbar.update(idx)
                    videoname = filename.split('/')[-1].split('.')[0]
                    if videoname in NTU.IGNORE_LIST:
                        continue
                    
                    skl_file = zfile.open(filename)
                    video_poses = parse_ntu_skeleton(skl_file)
                    # if len(video_poses[0]) > 2:
                        # print("\nWarning: More than two skls at video =", filename)
                    flat_video_poses = flatten_video_poses(video_poses)
                    norm_video_poses = apply_NTU_normalization(flat_video_poses)
                    np.savetxt(out_file, norm_video_poses, fmt='%g', delimiter=',')
                    if idx%25:
                        out_file.flush()
                    
                    num_frames = len(video_poses)
                    srt_fr_pt = line_pt
                    end_fr_pt = line_pt+num_frames-1
                    line_pt += num_frames
                    setup, cam, subj, dup, act = parse_videoname(videoname)
                    descs_data.append([setup,cam,subj,dup,act,srt_fr_pt,end_fr_pt])
                progbar.finish()
            descs_filepath = os.path.join(NTU.DATA_DIR, 'descs.csv')
            descs = pd.DataFrame(descs_data, columns=['setup', 'camera', 'subject', 
                        'duplicate', 'action', 'start_frame_pt','end_frame_pt'])
            print("Saving dataset description at:", descs_filepath)
            descs.T.to_csv(descs_filepath, index=False, header=None)
        elif dataset_version == '2':
            descs_v1_filepath = os.path.join(NTU.DATA_DIR, 'descs.csv')
            v1_filepath = os.path.join(NTU.DATA_DIR, 'skl.csv')
            out_filepath = os.path.join(NTU_V2.DATA_DIR, 'skl.csv')
            zip_filepath = os.path.join(NTU_V2.DATA_DIR, 'nturgbd_skeletons_s018_to_s032.zip')
            
            if os.path.exists(out_filepath) and not overwrite:
                raise Exception("Output file already exists, use --overwrite option. File: "+out_filepath)
            
            print("Copying NTU V1 file to NTU V2 folder:", v1_filepath, out_filepath)
            copyfile(v1_filepath, out_filepath)
            
            descs_v1 = pd.read_csv(descs_v1_filepath, index_col=False,header=None).T
            descs_v1.columns = ['setup','camera','subject','duplicate','action',
                'start_frame_pt','end_frame_pt']
            
            descs_data = []
            line_pt = descs_v1.iloc[-1].end_frame_pt + 1
            
            print("Reading skeletons from zipfile:", zip_filepath)
            print("Saving normalized skeletons at:", out_filepath)
            print("Reading, normalizing and saving...")
            with ZipFile(zip_filepath) as zfile, open(out_filepath, 'ab') as out_file:
                filenames_list = sorted([file_info.filename for file_info in zfile.infolist()])
                
                progbar = progressbar.ProgressBar(max_value=len(filenames_list))
                for idx, filename in enumerate(filenames_list):
                    progbar.update(idx)
                    videoname = filename.split('.')[0]
                    if videoname in NTU_V2.IGNORE_LIST:
                        continue
                    
                    skl_file = zfile.open(filename)
                    video_poses = parse_ntu_skeleton(skl_file)
                    # if len(video_poses[0]) > 2:
                        # print("\nWarning: More than two skls at:", filename)
                    flat_video_poses = flatten_video_poses(video_poses)
                    norm_video_poses = apply_NTU_normalization(flat_video_poses)
                    np.savetxt(out_file, norm_video_poses, fmt='%g', delimiter=',')
                    if idx%25:
                        out_file.flush()
                    
                    num_frames = len(video_poses)
                    srt_fr_pt = line_pt
                    end_fr_pt = line_pt+num_frames-1
                    line_pt += num_frames
                    setup, cam, subj, dup, act = parse_videoname(videoname)
                    descs_data.append([setup,cam,subj,dup,act,srt_fr_pt,end_fr_pt])
                progbar.finish()
            descs_v2 = pd.DataFrame(descs_data, columns=['setup','camera','subject', 
                        'duplicate', 'action', 'start_frame_pt','end_frame_pt'])
            descs = pd.concat([descs_v1, descs_v2], ignore_index=True)
            descs_filepath = os.path.join(NTU_V2.DATA_DIR, 'descs.csv')
            print("Saving dataset description at:", descs_filepath)
            descs.T.to_csv(descs_filepath, index=False, header=None)
    else:
        if dataset_version == '1':
            data_dir = NTU.DATA_DIR
        elif dataset_version == '2':
            data_dir = NTU_V2.DATA_DIR
        
        csv_filepath = os.path.join(data_dir, 'skl.csv')
        npy_filepath = os.path.join(data_dir, 'skl.npy')
        print("Reading skeletons from csv:", csv_filepath)
        print("This can take several minutes...")
        skls = np.loadtxt(csv_filepath, delimiter=',', dtype='float32')
        print("Saving skeletons from csv:", npy_filepath)
        np.save(npy_filepath, skls)
    
    print("> Finished successfully!")
