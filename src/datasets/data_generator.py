import os
import numpy as np

from keras.utils import Sequence

from datasets import UT, SBU, NTU, NTU_V2, YMJA
from misc.data_io import get_data

class DataGenerator(Sequence):
    def __init__(self, dataset_name, dataset_fold, subset,
                batch_size=32, reshuffle=False, shuffle_indiv_order=False, 
                sample_method = 'central', 
                **data_kwargs):
        if dataset_name == 'UT':
            dataset = UT
            self.pose_style = 'OpenPose'
        elif dataset_name == 'SBU':
            dataset = SBU
            self.pose_style = 'SBU'
        elif dataset_name == 'YMJA':
            dataset = YMJA
            self.pose_style = 'YMJA'
        elif dataset_name == 'NTU':
            dataset = NTU
            self.pose_style = 'NTU'
        elif dataset_name == 'NTU-V2':
            dataset = NTU_V2
            self.pose_style = 'NTU-V2'
        
        self.batch_size = batch_size
        self.reshuffle = reshuffle
        self.data_kwargs = data_kwargs
        self.shuffle_indiv_order = shuffle_indiv_order
        self.subset = subset
        self.sample_method = sample_method
        
        if subset == 'train':
            self.ground_truth = dataset.get_train_gt(dataset_fold)
        elif subset == 'validation':
            self.ground_truth = dataset.get_val_gt(dataset_fold)
        
        self.num_classes = self.ground_truth.action.max()+1
        
        if sample_method == 'central':
            num_videos = self.ground_truth.shape[0]
            if subset == 'train':
                self.shuffled_idx = np.random.choice(self.ground_truth.index.values, 
                    num_videos, replace=False)
            elif subset == 'validation':
                self.shuffled_idx = self.ground_truth.index.values
            self.num_batches = int(np.ceil(num_videos/batch_size))
        elif sample_method == 'all':
            self.data_kwargs['flat_seqs'] = True
            if  self.data_kwargs.get('seq_step') is None:
                self.data_kwargs['seq_step'] = self.data_kwargs['timesteps']//2
                
            seqs_mapping_file = 'seqs_mapping-subset_{}-fold_{}-timesteps_{}-{}-seq_step_{}.csv'.format(
                subset, dataset_fold, data_kwargs['timesteps'],
                data_kwargs['skip_timesteps'], data_kwargs['seq_step'])
            seqs_mapping_filepath = dataset.DATA_DIR + '/seqs_mapping/' + seqs_mapping_file
                
            if os.path.exists(seqs_mapping_filepath):
                type_index = type(self.ground_truth.index[0])
                seqs_mapping = np.genfromtxt(seqs_mapping_filepath, dtype='str')
                seqs_mapping = [ [ type_index(r[0]), int(r[1])] for r in seqs_mapping ]
            else:
                seqs_mapping = [] # Array with mapping for each sequence: (video_idx, seq_idx)
                for video_idx in self.ground_truth.index.values:
                    _, Y = get_data(self.ground_truth.loc[[video_idx]], 
                        pose_style=self.pose_style, 
                        sample_method=sample_method, **self.data_kwargs)
                    num_seqs = len(Y)
                    seqs_mapping += [ [video_idx, seq_idx] for seq_idx in range(num_seqs) ]
                
                np.savetxt(seqs_mapping_filepath, seqs_mapping, fmt='%s')
                
            self.seqs_mapping = seqs_mapping
            
            if subset == 'train':
                self.shuffled_idx = np.random.choice(range(len(seqs_mapping)), 
                    len(seqs_mapping), replace=False)
            elif subset == 'validation':
                self.shuffled_idx = list(range(len(seqs_mapping)))
            self.num_batches = int(np.ceil(len(seqs_mapping)/batch_size))
        
        # Validating data_kwargs
        _ = get_data(self.ground_truth.sample(), pose_style=self.pose_style, 
            sample_method=sample_method, **self.data_kwargs)
    
    def __len__(self):
        return self.num_batches
    
    def __getitem__(self, idx):
        return self.getSampleData(idx, False)

    def getSampleData(self, idx, new_arch_sample):
        batch_idxs = self.shuffled_idx[
            self.batch_size*idx:self.batch_size*(idx+1)]
        
        if self.sample_method == 'central':
            batch_gt = self.ground_truth.loc[batch_idxs]
            batch_x, batch_y = get_data(batch_gt, pose_style=self.pose_style, 
                num_classes=self.num_classes, **self.data_kwargs)
        elif self.sample_method == 'all':
            batch_x, batch_y = [], []
            for mapping_idx in batch_idxs:
                video_idx, seq_idx = self.seqs_mapping[mapping_idx]
                video_x, video_y = get_data(self.ground_truth.loc[[video_idx]], 
                    pose_style=self.pose_style,  num_classes=self.num_classes,
                    sample_method=self.sample_method, **self.data_kwargs)
                seq_data = [ j[seq_idx] for j in video_x]
                batch_x.append(seq_data)
                batch_y.append(video_y[seq_idx])
            batch_x = np.array(batch_x).transpose((1,0,2))
            batch_y = np.array(batch_y)
        
        if self.shuffle_indiv_order:
            NUM_PEOPLE = 2
            NUM_DIM = 3
            if self.pose_style == 'YMJA':
                NUM_DIM = 2

            if 'arch' in self.data_kwargs and (self.data_kwargs['arch'] == 'joint' or self.data_kwargs['arch'] == 'temp'):
                # Convert to form (num joints, num samples, timesteps, num people, num dimensions) for joint stream
                # Convert to form (timesteps, num_samples, num_joints, num_people, num_dimensions) for temporal stream
                batch_x = batch_x.reshape((batch_x.shape[0], batch_x.shape[1], batch_x.shape[2]//(NUM_PEOPLE*NUM_DIM), NUM_PEOPLE, NUM_DIM))
                
                # For half of the samples in the batch, we want to flip the people axis
                swap_index = np.random.choice(batch_x.shape[1], batch_x.shape[1]//2, replace=False)
                batch_x[:,swap_index] = np.flip(batch_x[:,swap_index], axis=3)

                # Reshape it to old form
                batch_x = batch_x.reshape((batch_x.shape[0], batch_x.shape[1], batch_x.shape[2] * batch_x.shape[3] * batch_x.shape[4]))

            elif 'arch' in self.data_kwargs and self.data_kwargs['arch'] == 'joint_temp_fused':

                new_batch_x = []
                # Iterate over both streams
                for batch_x_comp in batch_x:
                    # Convert to form (num joints, num samples, timesteps, num people, num dimensions) for joint stream
                    # Convert to form (timesteps, num_samples, num_joints, num_people, num_dimensions) for temporal stream
                    batch_x_comp = batch_x_comp.reshape((batch_x_comp.shape[0], batch_x_comp.shape[1], batch_x_comp.shape[2]//(NUM_PEOPLE*NUM_DIM), NUM_PEOPLE, NUM_DIM))
                    
                    # For half of the samples in the batch, we want to flip the people axis
                    swap_index = np.random.choice(batch_x_comp.shape[1], batch_x_comp.shape[1]//2, replace=False)
                    batch_x_comp[:,swap_index] = np.flip(batch_x_comp[:,swap_index], axis=3)

                    # Reshape it to old form
                    batch_x_comp = batch_x_comp.reshape((batch_x_comp.shape[0], batch_x_comp.shape[1], batch_x_comp.shape[2] * batch_x_comp.shape[3] * batch_x_comp.shape[4]))
                    
                    batch_x_comp = [ np.array(input) for input in batch_x_comp]

                    # For fused model during training, must append to single list of 25 instead of multiple lists
                    if new_arch_sample:
                        new_batch_x.append(batch_x_comp)
                    else:
                        new_batch_x += batch_x_comp

                return new_batch_x, batch_y

            else:
                ### Always swap half of the batch
                ### Swaps between the P1 joints and the P2 joints
                # 
                # In half of samples in batch, switch between who is P1 and P2
                #
                # For instance, if 4 joints, dimension batch_x = (4,8)
                # Dimension batch_y = (8) (num samples)
                #
                # >>> batch_x original
                # array([[ 0,  1,  2,  3,  4,  5,  6,  7],
                #        [ 8,  9, 10, 11, 12, 13, 14, 15],
                #        [16, 17, 18, 19, 20, 21, 22, 23],
                #        [24, 25, 26, 27, 28, 29, 30, 31]])
                # >>> batch_x new
                # array([[16,  1, 18,  3, 20, 21,  6,  7],
                #        [24,  9, 26, 11, 28, 29, 14, 15],
                #        [ 0, 17,  2, 19,  4,  5, 22, 23],
                #        [ 8, 25, 10, 27, 12, 13, 30, 31]])

                num_joints = len(batch_x)//2
                p1_joints = list(range(num_joints))
                p2_joints = list(range(num_joints, num_joints*2))
                swap_index = sorted(np.random.choice(list(range(batch_y.shape[0])), 
                    batch_y.shape[0]//2, replace=False))
                batch_x = np.array(batch_x)
                batch_x[:,swap_index] = np.squeeze(
                    batch_x[p2_joints + p1_joints][:,[swap_index]], axis=1)
                
        batch_x = [ np.array(input) for input in batch_x]
        
        # Must append to single list of 25 for training and validation
        if 'arch' in self.data_kwargs and self.data_kwargs['arch'] == 'joint_temp_fused' and not new_arch_sample:
            batch_x[0] = [ np.array(input) for input in batch_x[0]]
            batch_x[1] = [ np.array(input) for input in batch_x[1]]

            batch_x = batch_x[0] + batch_x[1]

        return batch_x, batch_y
    
    def on_epoch_end(self):
        if self.reshuffle:
            if self.sample_method == 'central':
                self.shuffled_idx = np.random.choice(self.ground_truth.index.values, 
                    self.ground_truth.shape[0], replace=False)
            elif self.sample_method == 'all':
                self.shuffled_idx = np.random.choice(range(len(self.seqs_mapping)), 
                    len(self.seqs_mapping), replace=False)

class DataGeneratorSeq(Sequence):
    def __init__(self, dataset_name, dataset_fold, subset,
                batch_size=32, reshuffle=False, shuffle_indiv_order=False, 
                pad_sequences = False, maxlen=None, padding='pre',
                buffer_data = False,
                **data_kwargs):
        if dataset_name == 'UT':
            dataset = UT
            max_framenum = 183
            self.pose_style = 'OpenPose'
        elif dataset_name == 'SBU':
            dataset = SBU
            max_framenum = 46
            self.pose_style = 'SBU'
        elif dataset_name == 'YMJA':
            dataset = YMJA
            max_framenum = 64
            self.pose_style = 'YMJA'
        elif dataset_name == 'NTU':
            dataset = NTU
            max_framenum = 300 # for all videos
            # max_framenum = 205 # for mutual videos
            self.pose_style = 'NTU'
        elif dataset_name == 'NTU-V2':
            dataset = NTU_V2
            # max_framenum = 300 # 214 for mutual videos
            max_framenum = 214 # 300 for all videos
            self.pose_style = 'NTU-V2'
        
        self.batch_size = batch_size
        self.reshuffle = reshuffle
        self.data_kwargs = data_kwargs
        self.shuffle_indiv_order = shuffle_indiv_order
        self.subset = subset
        self.pad_sequences = pad_sequences
        self.padding = padding
        self.buffer_data = buffer_data
        
        self.data_kwargs['flat_seqs'] = False
        self.data_kwargs['sample_method'] = 'all'
        
        if  self.data_kwargs.get('seq_step') is None:
            self.data_kwargs['seq_step'] = self.data_kwargs['timesteps']//2
        
        if maxlen is not None:
            self.maxlen = maxlen
        else: # Estimate maxlen
            self.maxlen = max_framenum//data_kwargs['seq_step']
            if data_kwargs['skip_timesteps'] is not None:
                self.maxlen = self.maxlen//data_kwargs['skip_timesteps']
            
        if subset == 'train':
            self.ground_truth = dataset.get_train_gt(dataset_fold)
        elif subset == 'validation':
            self.ground_truth = dataset.get_val_gt(dataset_fold)
        
        self.num_classes = self.ground_truth.action.max()+1
        num_videos = self.ground_truth.shape[0]
        self.num_batches = int(np.ceil(num_videos/batch_size))
        
        if not self.buffer_data:
            if subset == 'train':
                self.shuffled_idx = np.random.choice(self.ground_truth.index.values, 
                    num_videos, replace=False)
            elif subset == 'validation':
                self.shuffled_idx = self.ground_truth.index.values
            
            # Validating data_kwargs
            _ = get_data(self.ground_truth.sample(), pose_style=self.pose_style, 
                num_classes=self.num_classes, **self.data_kwargs)
        else:
            if subset == 'train':
                self.shuffled_idx = np.random.choice(range(num_videos), 
                    num_videos, replace=False)
            elif subset == 'validation':
                self.shuffled_idx = list(range(num_videos))
            
            self.buffer = get_data(self.ground_truth,pose_style=self.pose_style, 
                num_classes=self.num_classes, **self.data_kwargs)
    
    def __len__(self):
        return self.num_batches
    
    def __getitem__(self, idx):
        batch_idxs = self.shuffled_idx[
            self.batch_size*idx:self.batch_size*(idx+1)]
        
        if not self.buffer_data:
            batch_gt = self.ground_truth.loc[batch_idxs]
            
            batch_x, batch_y = get_data(batch_gt, pose_style=self.pose_style, 
                num_classes=self.num_classes, **self.data_kwargs)
        else:
            batch_x = [ self.buffer[0][idx] for idx in batch_idxs ]
            batch_y = self.buffer[1][batch_idxs]
        
        if self.shuffle_indiv_order:
            ### Always swap half of the batch
            num_joints = len(batch_x[0][0])//2
            p1_joints = list(range(num_joints))
            p2_joints = list(range(num_joints, num_joints*2))
            swap_index = sorted(np.random.choice(list(range(batch_y.shape[0])), 
                batch_y.shape[0]//2, replace=False))
                
            for video_swap_id in swap_index:
                video_seqs = np.array(batch_x[video_swap_id])
                batch_x[video_swap_id] = video_seqs[:, p2_joints + p1_joints, :]
        
        # Padding sequences
        if self.pad_sequences:
            maxlen = self.maxlen
            for idx, video_seqs in enumerate(batch_x):
                video_seqs = np.array(video_seqs)
                padded_video_seqs = np.zeros((maxlen,)+np.array(batch_x[0][0]).shape)
                if self.padding == 'pre':
                    padded_video_seqs[-len(video_seqs[:maxlen]):,...] = video_seqs[:maxlen,...]
                elif self.padding == 'post':
                    padded_video_seqs[:len(video_seqs[:maxlen]),...] = video_seqs[:maxlen,...]
                batch_x[idx] = padded_video_seqs
        
        batch_x = np.array(batch_x)
        return batch_x, batch_y
    
    def on_epoch_end(self):
        if self.reshuffle:
            num_videos = self.ground_truth.shape[0]
            if not self.buffer_data:
                self.shuffled_idx = np.random.choice(self.ground_truth.index.values, 
                    num_videos, replace=False)
            else:
                self.shuffled_idx = np.random.choice(range(num_videos), 
                    num_videos, replace=False)
                
