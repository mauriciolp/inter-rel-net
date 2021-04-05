import numpy as np
import pandas as pd
import json, glob
import linecache
import h5py

from keras.utils import to_categorical

TORSO, LEFT_HAND, RIGHT_HAND, LEFT_LEG, RIGHT_LEG = 0,1,2,3,4

# OpenPose body parts

# POSE_BODY_25_BODY_PARTS = ["Nose","Neck","RShoulder","RElbow","RWrist","LShoulder","LElbow","LWrist","MidHip","RHip","RKnee","RAnkle","LHip","LKnee","LAnkle","REye","LEye","REar","LEar","LBigToe","LSmallToe","LHeel","RBigToe","RSmallToe","RHeel","Background"]
# POSE_BODY_25_BODY_PARTS_COARSE = [TORSO, TORSO, RIGHT_HAND, RIGHT_HAND, RIGHT_HAND, LEFT_HAND, LEFT_HAND, LEFT_HAND, TORSO, RIGHT_LEG, RIGHT_LEG,RIGHT_LEG,LEFT_LEG,LEFT_LEG,LEFT_LEG,TORSO,TORSO,TORSO,TORSO,LEFT_LEG,LEFT_LEG,LEFT_LEG,RIGHT_LEG,RIGHT_LEG,RIGHT_LEG, 5]
# POSE_BODY_25_BODY_PARTS_COARSE_TEXT = ["TORSO", "TORSO", "RIGHT_HAND", "RIGHT_HAND", "RIGHT_HAND", "LEFT_HAND", "LEFT_HAND", "LEFT_HAND", "TORSO", "RIGHT_LEG", "RIGHT_LEG","RIGHT_LEG","LEFT_LEG","LEFT_LEG","LEFT_LEG","TORSO","TORSO","TORSO","TORSO","LEFT_LEG","LEFT_LEG","LEFT_LEG","RIGHT_LEG","RIGHT_LEG","RIGHT_LEG", "Background"]

POSE_BODY_25_BODY_PARTS = ["Nose","Neck","RShoulder","RElbow","RWrist","LShoulder","LElbow","LWrist","MidHip","RHip","RKnee","RAnkle","LHip","LKnee","LAnkle","REye","LEye","REar","LEar","LBigToe","LSmallToe","LHeel","RBigToe","RSmallToe","RHeel"]

POSE_BODY_25_BODY_PARTS_COARSE = [TORSO, TORSO, RIGHT_HAND, RIGHT_HAND, RIGHT_HAND, LEFT_HAND, LEFT_HAND, LEFT_HAND, TORSO, RIGHT_LEG, RIGHT_LEG,RIGHT_LEG,LEFT_LEG,LEFT_LEG,LEFT_LEG,TORSO,TORSO,TORSO,TORSO,LEFT_LEG,LEFT_LEG,LEFT_LEG,RIGHT_LEG,RIGHT_LEG,RIGHT_LEG]
POSE_BODY_25_BODY_PARTS_COARSE_TEXT = ["TORSO", "TORSO", "RIGHT_HAND", "RIGHT_HAND", "RIGHT_HAND", "LEFT_HAND", "LEFT_HAND", "LEFT_HAND", "TORSO", "RIGHT_LEG", "RIGHT_LEG","RIGHT_LEG","LEFT_LEG","LEFT_LEG","LEFT_LEG","TORSO","TORSO","TORSO","TORSO","LEFT_LEG","LEFT_LEG","LEFT_LEG","RIGHT_LEG","RIGHT_LEG","RIGHT_LEG"]

### SBU body parts

# SBU_15_BODY_PARTS = ["HEAD","NECK","TORSO","LEFT_SHOULDER","LEFT_ELBOW","LEFT_HAND","RIGHT_SHOULDER","RIGHT_ELBOW","RIGHT_HAND","LEFT_HIP","LEFT_KNEE","LEFT_FOOT","RIGHT_HIP","RIGHT_KNEE","RIGHT_FOOT"]
SBU_15_BODY_PARTS = ["Nose","Neck","MidHip","LShoulder","LElbow","LWrist","RShoulder","RElbow","RWrist","LHip","LKnee","LAnkle","RHip","RKnee","RAnkle"] # OpenPose equivalent?
SBU_15_BODY_PARTS_COARSE = [TORSO,TORSO,TORSO,LEFT_HAND,LEFT_HAND,LEFT_HAND,RIGHT_HAND,RIGHT_HAND,RIGHT_HAND,LEFT_LEG,LEFT_LEG,LEFT_LEG,RIGHT_LEG,RIGHT_LEG,RIGHT_LEG]
SBU_15_BODY_PARTS_COARSE_TEXT = ["TORSO","TORSO","TORSO","LEFT_HAND","LEFT_HAND","LEFT_HAND","RIGHT_HAND","RIGHT_HAND","RIGHT_HAND","LEFT_LEG","LEFT_LEG","LEFT_LEG","RIGHT_LEG","RIGHT_LEG","RIGHT_LEG"]

### NTU body parts
# NTU_25_BODY_PARTS = ['Base_of_the_spine', 'Middle_of_the_spine', 'Neck', 'Head', 'Left_shoulder', 'Left_elbow', 'Left_wrist', 'Left_hand', 'Right_shoulder', 'Right_elbow', 'Right_wrist', 'Right_hand', 'Left_hip', 'Left_knee', 'Left_ankle', 'Left_foot', 'Right_hip', 'Right_knee', 'Right_ankle', 'Right_foot', 'Spine', 'Tip_of_the_left_hand', 'Left_thumb', 'Tip_of_the_right_hand', 'Right_thumb']

## OpenPose equivalent
NTU_25_BODY_PARTS = ['MidHip','Middle_of_the_spine','Neck','Nose','LShoulder','LElbow','LWrist','Left_hand','RShoulder','RElbow','RWrist','Right_hand','LHip','LKnee','LAnkle','LBigToe','RHip','RKnee','RAnkle','RBigToe','Spine','Tip_of_the_left_hand','Left_thumb','Tip_of_the_right_hand','Right_thumb']
NTU_25_BODY_PARTS_COARSE = [TORSO, TORSO, TORSO, TORSO, LEFT_HAND, LEFT_HAND, LEFT_HAND, LEFT_HAND, RIGHT_HAND, RIGHT_HAND, RIGHT_HAND, RIGHT_HAND, LEFT_LEG, LEFT_LEG, LEFT_LEG, LEFT_LEG, RIGHT_LEG, RIGHT_LEG, RIGHT_LEG, RIGHT_LEG, TORSO, LEFT_HAND, LEFT_HAND, RIGHT_HAND, RIGHT_HAND]
NTU_25_BODY_PARTS_COARSE_TEXT = ["TORSO", "TORSO", "TORSO", "TORSO", "LEFT_HAND", "LEFT_HAND", "LEFT_HAND", "LEFT_HAND", "RIGHT_HAND", "RIGHT_HAND", "RIGHT_HAND", "RIGHT_HAND", "LEFT_LEG", "LEFT_LEG", "LEFT_LEG", "LEFT_LEG", "RIGHT_LEG", "RIGHT_LEG", "RIGHT_LEG", "RIGHT_LEG", "TORSO", "LEFT_HAND", "LEFT_HAND", "RIGHT_HAND", "RIGHT_HAND"]


def filter_joints(person, selected_joints, joint_indexing=POSE_BODY_25_BODY_PARTS):
    joints_mask = np.isin(joint_indexing, selected_joints)
    selected_parts = np.array(joint_indexing)[joints_mask]
    selected_coords = np.array(person['coords'])[joints_mask]
    
    person['coords'] = selected_coords
    
    return person

def prune_bodies(video_poses, metric_name = 'central'):
    max_num_ppl = np.max([ len(frame_poses) for frame_poses in video_poses])
    bodies_coords = [ [] for _ in range(max_num_ppl) ]
    
    central_points = []
    for frame_poses in video_poses:
        for pose_idx, pose in enumerate(frame_poses):
            bodies_coords[pose_idx].append(pose['coords'].tolist())
        ### Dynamic central_point based on the min and max coords
        curr_coords = [ person['coords'] for person in frame_poses ]
        max_coords = np.array(curr_coords).max(axis=1).max(axis=0)
        non_zero_curr_coords = np.array(curr_coords)
        non_zero_curr_coords[non_zero_curr_coords==0] = 9999
        min_coords = np.array(non_zero_curr_coords).min(axis=1).min(axis=0)
        frame_central_point = np.mean([min_coords, max_coords], axis=0)
        central_points.append(frame_central_point)
    central_point = np.mean(central_points, axis=0)
    
    ### Compute Metric
    metric = []
    for body_coords in bodies_coords:
        body = np.array(body_coords)
        if metric_name == 'motion':
            t1 = body[:-1]
            t2 = body[1:]
            frames_motion = np.linalg.norm(t2 - t1, axis=2).mean(axis=1)
            total_motion = frames_motion.sum()
            metric.append(total_motion)
        elif metric_name == 'central':
            ### Distance to body center
            central_idx = 8 # "MidHip"
            dist_central = np.linalg.norm(body[:,central_idx] - central_point, axis=1).mean()
            
            ### Distance to all joints
            # joints_distance = np.linalg.norm(body - central_point, axis=2)
            # dist_central = joints_distance.mean()
            # dist_central = np.median(joints_distance)
            
            metric.append(dist_central)
        
    ### Prune based on metric
    metric_cut = sorted(metric)[:2][-1]
    pruned_video_poses = []
    for frame_poses in video_poses:
        pruned_frame_poses = []
        for pose_idx, pose in enumerate(frame_poses):
            if metric[pose_idx] <= metric_cut:
                # print(pose_idx)
                pruned_frame_poses.append(pose)
        pruned_video_poses.append(pruned_frame_poses)
    
    return pruned_video_poses

def prune_people(people):
    prunned_people = []
    for person in people:
        if np.mean(person['confs']) < 0.15: # Used at UT
            continue
        prunned_people.append(person)
    
    return prunned_people

def parse_json(json_filepath, prune=True, pose_style='OpenPose', ymja_frame_data = None):
    
    people = []
    if pose_style == 'OpenPose':
        with open(json_filepath) as json_file:
            frame_data = json.load(json_file)
        iter_list = frame_data['people']
    elif pose_style == 'YMJA':
        frame_data = ymja_frame_data
        iter_list = [frame_data['perp'], frame_data['victim']]

    for person in iter_list:
        per = {}
        if pose_style == 'OpenPose':
            pose_keypoints_2d = person['pose_keypoints_2d']
        elif pose_style == 'YMJA':
            pose_keypoints_2d = person

        coords_x = pose_keypoints_2d[0::3]
        coords_y = pose_keypoints_2d[1::3]
        confidences = pose_keypoints_2d[2::3]
        coords = np.array([coords_x, coords_y]).T
        per['coords'] = coords
        per['confs'] = confidences
        people.append(per)

    if prune:
        people = prune_people(people)
    
    return people

def denormalize(norm_coords):
    """ SBU denormalization
        original_X = 1280 - (normalized_X .* 2560);
        original_Y = 960 - (normalized_Y .* 1920);
        original_Z = normalized_Z .* 10000 ./ 7.8125;
    """
    denorm_coords = np.empty(norm_coords.shape)
    denorm_coords[:,0] = 1280 - (norm_coords[:,0] * 2560)
    denorm_coords[:,1] = 960 - (norm_coords[:,1] * 1920)
    denorm_coords[:,2] = norm_coords[:,1] * 10000 / 7.8125
    
    return denorm_coords

def parse_sbu_txt(pose_filepath, normalized=False):
    video_poses_mat = np.loadtxt(pose_filepath, delimiter=',', usecols=range(1,91))
    
    video_poses = []
    for frame_pose in video_poses_mat:
        people = []
        # 2 persons * 15 joints * 3 dimensions
        people_poses = frame_pose.reshape(2,45)
        for person in people_poses:
            per = {}
            if normalized:
                per['coords'] = person.reshape(15,3)
            else:
                per['coords'] = denormalize(person.reshape(15,3))
            per['confs'] = 15*[1]
            people.append(per)
        video_poses.append(people)
    
    return video_poses

def denormalize_ntu(norm_coords):
    # This is not possible because there is loss of information after normalization
    raise NotImplementedError("denormalize_ntu")

def apply_NTU_normalization(video_poses, pose_style):
    """ From "NTU RGB+D: A Large Scale Dataset for 3D Human Activity Analysis"
        > We translate them to the body coordinate system with its origin on the “middle of the spine” joint (number 2 in Figure 1), followed by a 3D rotation to fix the X axis parallel to the 3D vector from “right shoulder” to “left shoulder”, and Y axis towards the 3D vector from “spine base” to “spine”. The Z axis is fixed as the new X × Y. In the last step of normalization, we scale all the 3D points based on the distance between “spine base” and “spine” joints.
        > In the cases of having more than one body in the scene, we transform all of them with regard to the main actor’s skeleton.
    
        - "spine base" was translated to "MidHip" at NTU_25_BODY_PARTS
        
        Obs: Since OpenPose and SBU do not have the joints “middle of the spine”
        and “spine”, 'MidHip' and 'Neck' are respectively used instead.
    """
    
    if pose_style == 'OpenPose' or pose_style == 'YMJA':
        joint_indexing = POSE_BODY_25_BODY_PARTS
    elif pose_style == 'SBU':
        joint_indexing = SBU_15_BODY_PARTS
    else:
        raise NotImplementedError("Invalid pose_style: "+pose_style)
    
    middle_joint_idx = joint_indexing.index('MidHip')
    neck_joint_idx = joint_indexing.index('Neck')
    left_shoulder_joint_idx = joint_indexing.index('LShoulder')
    right_shoulder_joint_idx = joint_indexing.index('RShoulder')
    
    normalized_video_poses = []
    for frame_idx, frame_poses in enumerate(video_poses):
        p1_coords = frame_poses[0]['coords']
        p1_middle_joint = p1_coords[middle_joint_idx]
        p1_neck_joint = p1_coords[neck_joint_idx]
        p1_left_shoulder_joint = p1_coords[left_shoulder_joint_idx]
        p1_right_shoulder_joint = p1_coords[right_shoulder_joint_idx]
        
        new_origin = p1_middle_joint
        scale_val = .5 / np.linalg.norm(p1_middle_joint - p1_neck_joint)
        
        y = p1_neck_joint - p1_middle_joint
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
            translated = person['coords'] - new_origin
            rotated = np.dot(rotation_matrix, translated.T).T
            scaled = rotated# * scale_val
            
            normalized_person = {'coords': scaled, 'confs': person['confs'].copy()}
            normalized_frame_poses.append(normalized_person)
        
        normalized_video_poses.append(normalized_frame_poses)
        
    return normalized_video_poses

def parse_ntu_skl(row_start, row_end, normalized=False, 
        ntu_skl_filepath='data/ntu-rgbd/skl.csv'):
    
    if ntu_skl_filepath.endswith('.csv'):
        ### Not so fast
        video_poses_mat = []
        for row_idx in range(row_start, row_end+1):
            selected_row = linecache.getline(ntu_skl_filepath, row_idx)
            video_poses_mat.append(np.fromstring(selected_row, sep=','))
        video_poses_mat = np.array(video_poses_mat)
    elif ntu_skl_filepath.endswith('.npy'):
        ### Fast
        video_poses_mat = np.load(ntu_skl_filepath, mmap_mode='r')[row_start-1:row_end]
    elif ntu_skl_filepath.endswith('.mat'):
        ### Fast, but makes training hangs sometimes
        with h5py.File(ntu_skl_filepath,'r') as f:
            video_poses_mat = f['skl'][:,row_start-1:row_end].T
    
    video_poses = []
    for frame_pose in video_poses_mat:
        people = []
        # 2 persons * 25 joints * 3 dimensions
        people_poses = frame_pose.reshape(2,75)
        for person in people_poses:
            per = {}
            if normalized:
                per['coords'] = person.reshape(25,3)
            else:
                per['coords'] = denormalize_ntu(person.reshape(25,3))
            per['confs'] = 25*[1]
            people.append(per)
        video_poses.append(people)
    
    return video_poses

def parse_ntu_skeleton(skl_filepath):
    """ Based on ReadBodyFile.m from Liu Jun """
    with open(skl_filepath) as skl_file:
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

def track_bodies(video_poses):
    # null_joints = np.array([ [-1, -1] for _ in range(25) ])
    null_joints = np.array([ [0, 0] for _ in range(25) ])
    max_num_ppl = np.max([ len(frame_poses) for frame_poses in video_poses])
    bodies_coords = [[] for _ in range(max_num_ppl)]
    print(len(video_poses[0]))
    for pose_idx, pose in enumerate(video_poses[0]):
        bodies_coords[pose_idx].append(pose['coords'].tolist())
    for missing_idx in range(pose_idx+1,max_num_ppl):
        bodies_coords[missing_idx].append(null_joints)
    
    prvs_coords = [ np.array(body[0]) for body in bodies_coords ]
    prvs_coords_log = []
    for prvs_idx, frame_poses in enumerate(video_poses[1:]):
        if frame_poses == []: # Skipping frame because there is no pose extracted
            for body_coords in bodies_coords:
                body_coords.append(null_joints)
            continue
        curr_coords = [ person['coords'] for person in frame_poses ]
        
        joints_distances = [ np.linalg.norm(curr_coords - coord, axis=2) for coord in prvs_coords ]
        distances = np.array([ np.median(dists, axis=1).tolist() 
            for dists in joints_distances ])
        
        ### Average current distance with prvs distance from prvs_coords_log
        last_k = 3
        dists_list = [distances]
        for older_prvs_coords in prvs_coords_log[-last_k:]:
            older_joints_distances = [ np.linalg.norm(curr_coords - coord, axis=2) 
                for coord in older_prvs_coords ]
            older_dists = np.array([ np.median(dists, axis=1).tolist() 
                for dists in older_joints_distances ])
            dists_list.append(older_dists)
        distances = np.average(dists_list, axis=0)
        
        ### Attribute pose to min distance body, if body still have not been appended yet
        sorted_idx = np.dstack(np.unravel_index(np.argsort(
            distances.ravel()), distances.shape))[0]
        poses_used = []
        bodies_appended = []
        for dist_idx in sorted_idx:
            body_idx, pose_idx = dist_idx
            if body_idx in bodies_appended or pose_idx in poses_used:
                continue
            min_dist = distances[tuple(dist_idx)]
            if min_dist < 50: # Hand-picked threshold
                bodies_coords[body_idx].append(curr_coords[pose_idx])
                poses_used.append(pose_idx)
                bodies_appended.append(body_idx)
            if len(poses_used) == len(curr_coords):
                break
        
        # Making sure all bodies are appended
        for body_coords in bodies_coords:
            if len(body_coords) == (prvs_idx+1):
                body_coords.append(null_joints)
        
        prvs_coords_log.append([ body_prvs_coords.copy() for body_prvs_coords in prvs_coords])
        
        # Updating prvs_coords for the non-zero values
        for body_coords, body_prvs_coords in zip(bodies_coords, prvs_coords):
            last_body_coords = body_coords[-1]
            body_prvs_coords[last_body_coords > 0] = last_body_coords[last_body_coords > 0]
        
    tracked_video_poses = []
    for frame_idx in range(len(video_poses)):
        tracked_video_poses.append([ 
            {'coords': np.array(body_coords[frame_idx])} 
            for body_coords in bodies_coords ])
    
    return tracked_video_poses

def read_video_poses(video_gt, pose_style='OpenPose', normalization=None, prune=True):
    if pose_style == 'OpenPose' or pose_style == 'YMJA':

        video_poses = []
        if pose_style == 'OpenPose':
            video_keypoints_dir = video_gt.path
            json_list = glob.glob(video_keypoints_dir+'/*.json')
            json_list.sort()
            
            if json_list == []:
                raise FileNotFoundError("Error reading keypoints at: "+video_keypoints_dir)

            for json_file in json_list:
                people = parse_json(json_file, prune)
                video_poses.append(people)
        # YMJA
        else:
            with open(video_gt.path) as json_file:
                frames_data = json.load(json_file)
                for frame_data in frames_data:
                    people = parse_json(video_gt.path, prune, 'YMJA', frame_data)
                    video_poses.append(people)
        if prune:
            tracked_video_poses = track_bodies(video_poses)
            pruned_video_poses = prune_bodies(tracked_video_poses)
            video_poses = pruned_video_poses
        
        if normalization == 'NTU':
            video_poses = apply_NTU_normalization(video_poses, pose_style)
    elif pose_style == 'SBU':
        video_keypoints_dir = video_gt.path
        pose_filepath = video_keypoints_dir + '/skeleton_pos.txt'
        normalized = (normalization == 'SBU')
        video_poses = parse_sbu_txt(pose_filepath, normalized=normalized)
        
        if normalization == 'NTU':
            video_poses = apply_NTU_normalization(video_poses, pose_style)
    elif pose_style == 'NTU':
        normalized = (normalization == 'NTU')
        video_poses = parse_ntu_skl(
            video_gt.start_frame_pt, video_gt.end_frame_pt, 
            normalized=normalized,
            ntu_skl_filepath=video_gt.DATA_DIR+'/skl.npy')
    elif pose_style == 'NTU-V2':
        normalized = (normalization == 'NTU')
        video_poses = parse_ntu_skl(
            video_gt.start_frame_pt, video_gt.end_frame_pt, 
            normalized=normalized,
            ntu_skl_filepath=video_gt.DATA_DIR+'/skl.npy')
    
    # Add 'zeros' person if there is a single person for the majority of the poses
    num_people_med = np.median([ len(frame_pose) for frame_pose in video_poses ])
    if num_people_med == 1:
        zero_coords = np.zeros_like(video_poses[0][0]['coords'])
        for frame_pose in video_poses:
            zero_person = {'coords': zero_coords}
            frame_pose.append(zero_person)
    
    return video_poses

def insert_joint_idx(p1_and_p2, num_joints, scale):
    for idx in range(num_joints):
        p1_and_p2[idx].append(idx/scale)
        p1_and_p2[idx+num_joints].append(idx/scale)
        # p1_and_p2[idx].append(idx/(num_joints-1)) # div_len_idx
        # p1_and_p2[idx+num_joints].append(idx/(num_joints-1)) # div_len_idx
        
        # one_hot = np.zeros(num_joints)
        # one_hot[idx] = 1
        # p1_and_p2[idx] += one_hot.tolist()
        # p1_and_p2[idx+num_joints] += one_hot.tolist()
    pass

def insert_body_part(p1_and_p2, num_joints, scale, body_parts_mapping):
    num_body_parts = len(np.unique(body_parts_mapping))
    for idx in range(num_joints):
        body_part_idx = body_parts_mapping[idx]/scale
        p1_and_p2[idx].append(body_part_idx)
        p1_and_p2[idx+num_joints].append(body_part_idx)

# Called from each dataset
def get_data(gt_split, pose_style, timesteps=16, skip_timesteps=None,
        add_joint_idx=True, add_body_part=True, normalization=None, 
        selected_joints=None, num_classes=None, prune=False, 
        sample_method = 'central', seq_step=None, flat_seqs=False, arch=None):
    if pose_style == 'OpenPose' or pose_style == 'YMJA':
        joint_indexing = POSE_BODY_25_BODY_PARTS
        body_parts_mapping = POSE_BODY_25_BODY_PARTS_COARSE
    elif pose_style == 'SBU':
        joint_indexing = SBU_15_BODY_PARTS
        body_parts_mapping = SBU_15_BODY_PARTS_COARSE
    elif pose_style == 'NTU' or pose_style == 'NTU-V2':
        joint_indexing = NTU_25_BODY_PARTS
        body_parts_mapping = NTU_25_BODY_PARTS_COARSE
        
    all_video_poses = []
    for video_id, video_gt in gt_split.iterrows():
        video_poses = read_video_poses(video_gt, pose_style, 
            normalization=normalization, prune=prune)
        if selected_joints is not None:
            for frame_pose in video_poses:
                for person in frame_pose:
                    filter_joints(person, selected_joints, joint_indexing)
        all_video_poses.append(video_poses)
    
    scale = (1 if normalization is None else 10) # unscaled or div_10
    
    X = []
    Y = []
    num_joints = len(all_video_poses[0][0][0]['coords'])
    num_dim = len(all_video_poses[0][0][0]['coords'][0])
    for i, video_poses in enumerate(all_video_poses):
        action = gt_split.action.values[i]
        p1_all_joint_coords = [ [] for _ in range(num_joints) ]
        p2_all_joint_coords = [ [] for _ in range(num_joints) ]
        
        if skip_timesteps is not None:
            video_poses = video_poses[::skip_timesteps]
        
        for frame_pose in video_poses:
            if len(frame_pose) < 2: # Skipping frames with insufficient people
                continue
            p1 = frame_pose[0]
            for idx, coord in enumerate(p1['coords']):
                p1_all_joint_coords[idx] += coord.tolist()
            p2 = frame_pose[1]
            for idx, coord in enumerate(p2['coords']):
                p2_all_joint_coords[idx] += coord.tolist()
        p1_and_p2 = np.array(p1_all_joint_coords + p2_all_joint_coords)

        ### 1) Keeping only the central timesteps
        if sample_method == 'central':
            p1_and_p2.resize((num_joints*2, p1_and_p2.shape[1]//num_dim, num_dim))
            if p1_and_p2.shape[1] < timesteps: # Need to pad sequence
                pad_val = int( np.ceil( (timesteps - p1_and_p2.shape[1])/2 ) )
                pad_width = ((0,0), (pad_val,pad_val), (0,0))
                p1_and_p2 = np.pad(p1_and_p2, pad_width=pad_width, mode='constant')
            center = p1_and_p2.shape[1]//2
            central_window = slice(center - timesteps//2, center + timesteps//2)
            p1_and_p2 = p1_and_p2[:,central_window].reshape(
                (num_joints*2, timesteps*num_dim))
            p1_and_p2 = p1_and_p2.tolist()
            
            if add_joint_idx:
                insert_joint_idx(p1_and_p2, num_joints, scale)
            if add_body_part:
                insert_body_part(p1_and_p2, num_joints, scale, body_parts_mapping)
            
            X.append(p1_and_p2)
            Y.append(action)
        
        ### 2) Breaking the video into multiple inputs of length 'timesteps'
        if sample_method == 'all':
            p1_and_p2.resize((num_joints*2, p1_and_p2.shape[1]//num_dim, num_dim))
            if p1_and_p2.shape[1] < timesteps: # Need to pad sequence
                pad_val = int( np.ceil( (timesteps - p1_and_p2.shape[1])/2 ) )
                pad_width = ((0,0), (pad_val,pad_val), (0,0))
                p1_and_p2 = np.pad(p1_and_p2, pad_width=pad_width, mode='constant')
            
            num_frames = p1_and_p2.shape[1]
            range_end = (num_frames - timesteps + 1)
            if seq_step is None:
                seq_step = timesteps//2
            p1_and_p2 = np.array([ p1_and_p2[:,i:i+timesteps].reshape(-1,timesteps*num_dim).tolist()
                for i in range(0, range_end, seq_step)])
            
            p1_and_p2 = p1_and_p2.tolist()
            
            if add_joint_idx:
                for p1_and_p2_seq in p1_and_p2:
                    insert_joint_idx(p1_and_p2_seq, num_joints, scale)
            if add_body_part:
                for p1_and_p2_seq in p1_and_p2:
                    insert_body_part(p1_and_p2_seq, num_joints, scale, body_parts_mapping)
            
            if not flat_seqs:
                X.append(p1_and_p2)
                Y.append(action)
            else:
                X += p1_and_p2
                Y += [action] * len(p1_and_p2)
    
    if num_classes is None:
        num_classes = gt_split.action.max()+1
    Y = to_categorical(Y, num_classes)
    
    # Input for the network must be (n_joints * 2 (for each person), n_samples, timesteps*num_dim + bodypart + joint_index)
    # Dim 1 in format (p0_j0, p0_j1, p0_j2, ..., p1_j0, p1_j1,...)
    # Dimension 3 in format (x_0, y_0, z_0, x_1, y_1, z_1)
    if sample_method == 'central' or flat_seqs:
        X = np.array(X).transpose((1,0,2)).tolist()

    # If using joint stream, want to convert to form (n_joints = 15, n_samples, timesteps * num_people (2) * num_dimension)
    # Dimension 3 in format (x_p0_t0, y_p0_t0, z_p0_t0, x_p1_t0, y_p1_t0, z_p1_t0, x_p0_t1, y_po_t1, ...)
    if arch == 'joint' or arch == 'joint_temp_fused':

        new_x = np.array(X)
        
        # Separate timesteps and dimension axis
        new_x = new_x.reshape((new_x.shape[0], new_x.shape[1], timesteps, num_dim))

        # Separate joints of each person into two arrays
        p1_joints = new_x[0:new_x.shape[0]//2]
        p2_joints = new_x[new_x.shape[0]//2:new_x.shape[0]]

        # Concatenate along dimension axis and collapse into single object
        joint_stream = np.concatenate((p1_joints, p2_joints), axis=3)
        joint_stream = joint_stream.reshape(joint_stream.shape[0], joint_stream.shape[1], joint_stream.shape[2] * joint_stream.shape[3])

        if arch != 'joint_temp_fused':
            X = joint_stream

    # If using temp stream, want to convert to form (n_joints = timesteps, n_samples, num_joints * num_people(2) * num_dimenson)
    # Dimension 3 in format (x_p0_j0, y_p0_j0, z_p0_j0, x_p1_j0, y_p1_j0, z_p1_j0, x_p0_j1, ...)
    
    if arch == 'temp' or arch == 'joint_temp_fused':
        new_x = np.array(X)
        
        # Separate timesteps and dimension axis
        new_x = new_x.reshape((new_x.shape[0], new_x.shape[1], timesteps, num_dim))

        # Separate joints of each person into two arrays
        p1_joints = new_x[0:new_x.shape[0]//2]
        p2_joints = new_x[new_x.shape[0]//2:new_x.shape[0]]

        
        # Concatenate along dimension axis
        # New form is (joints, num_samples, timesteps, person*num_dimension)
        temp_stream = np.concatenate((p1_joints, p2_joints), axis=3)
        
        # New form is (timesteps, num_samples, joints, person*num_dimensions)
        temp_stream = np.transpose(temp_stream, axes=(2, 1, 0, 3))

        # Collapse into final form
        temp_stream = temp_stream.reshape(temp_stream.shape[0], temp_stream.shape[1], temp_stream.shape[2] * temp_stream.shape[3])

        if arch != 'joint_temp_fused':
            X = temp_stream

    if arch == 'joint_temp_fused':
        return ([joint_stream, temp_stream], Y)

    return X, Y
