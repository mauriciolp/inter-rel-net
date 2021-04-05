import numpy as np
import argparse, sys, os, time
import progressbar

import tensorflow as tf
if int(tf.__version__.split('.')[1]) >= 14:
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

from keras.metrics import categorical_accuracy
import keras.backend as K
    
from datasets import UT, SBU, NTU, NTU_V2, YMJA
from datasets.data_generator import DataGenerator
from models.rn import get_model, fuse_rn
from misc.utils import read_config

from math import pi, sqrt, exp


#%% Functions
def load_args():
    ap = argparse.ArgumentParser(
        description='Predict using Relational Network.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    
    # Positional arguments
    ap.add_argument('weights_path',
        help='path to weights to be loaded by the relational network',
        type=str)
    
    # Data arguments
    ap.add_argument('-d','--dataset-name',
        help="dataset to be used for predicting",
        default='UT',
        choices=['UT', 'SBU', 'NTU', 'NTU_V2', 'YMJA'])
    ap.add_argument('-f','--dataset-fold',
        help="dataset fold to be used for predicting",
        default=9,
        type=int)
    ap.add_argument('-t', '--timesteps',
        type=int,
        default=16,
        help='how many timesteps to use')
    
    # Predicting arguments
    ap.add_argument('-b', '--batch-size',
        type=int,
        default=32,
        help='batch size used to extract the features')
    
    ap.add_argument('--print_args',
        help="whether to print current arguments' values",
        action='store_true')
    
    args = ap.parse_args()
    
    return args

def gauss(n, sigma=1):
    """ Gaussian function to generate weights prioritizing the central seqs """
    
    r = range(-int(n/2),int(n/2)+1)
    if n%2 == 0: # Even n
        r = [ x + .5 for x in r][:-1]
        g = [1 / (sigma * sqrt(2*pi)) * exp(-float(x)**2/(2*sigma**2)) for x in r]
    else: # Odd n
        g = [1 / (sigma * sqrt(2*pi)) * exp(-float(x)**2/(2*sigma**2)) for x in r]
    
    return np.array(g)/np.sum(g) # Normalizing to sum up to 1

""" predict_rn is outdated. If update is necessary, base on predict_fused_rn. """
def predict_rn(weights_path, dataset_name, model_kwargs, data_kwargs,
        dataset_fold=None, batch_size=32, verbose=2):
    if verbose > 0:
        print("***** Predicting parameters *****")
        print("\t weights_path:", weights_path)
        print("\t Dataset:", dataset_name)
        print("\t Dataset fold:", dataset_fold)
        print("\t Skeleton info")
        for key, value in data_kwargs.items():
            print("\t > {}: {}".format(key, value))
        print("\t Model info")
        for key, value in model_kwargs.items():
            print("\t > {}: {}".format(key, value))
        # print("\t Relation info")
        # print("\t > relationship type:", rel_type)
        # print("\t > fusion type:", fuse_type)
        print("\t Predicting options")
        print("\t > Batch Size:", batch_size)
    
    ####
    if dataset_name == 'UT':
        dataset = UT
    elif dataset_name == 'SBU':
        dataset = SBU
    elif dataset_name == 'NTU':
        dataset = NTU
    elif dataset_name == 'YMJA':
        dataset = YMJA
    
    if verbose > 0:
        print("Reading data...")
    X_val, Y_val = dataset.get_val(dataset_fold, **data_kwargs)
    
    timesteps = data_kwargs['timesteps']
    add_joint_idx = data_kwargs['add_joint_idx']
    add_body_part = data_kwargs['add_body_part']
    
    num_joints = len(X_val)//2
    object_shape = (len(X_val[0][0]),)
    output_size = len(Y_val[0])
    overhead = add_joint_idx + add_body_part # True/False = 1/0
    num_dim = (object_shape[0]-overhead)//timesteps
    
    if verbose > 0:
        print("Creating model...")
    model = get_model(num_objs=num_joints, object_shape=object_shape, 
        output_size=output_size, num_dim=num_dim, overhead=overhead,
        **model_kwargs)
    
    model.load_weights(weights_path)
    
    if verbose > 0:
        print("Starting predicting...")
    
    Y_pred = model.predict(X_val, batch_size=batch_size, verbose=verbose)
    
    acc_tensor = categorical_accuracy(Y_val, Y_pred)
    acc = K.eval(acc_tensor).mean()
    
    if verbose > 0:
        print("Validation acc: {:.2%}".format(acc))
    
    return Y_pred, Y_val

def predict_rn_seq(weights_path, dataset_name, model_kwargs, data_kwargs,
        dataset_fold=None, batch_size=32, verbose=2, return_acc = False, 
        use_data_gen=True):
    
    sample_method = 'all'
    flat_seqs = data_kwargs.get('flat_seqs', False)
    if verbose > 0:
        print("***** Predicting parameters *****")
        print("\t weights_path:", weights_path)
        print("\t Dataset:", dataset_name)
        print("\t Dataset fold:", dataset_fold)
        print("\t Skeleton info")
        for key, value in data_kwargs.items():
            print("\t > {}: {}".format(key, value))
        print("\t Model info")
        for key, value in model_kwargs.items():
            print("\t > {}: {}".format(key, value))
        print("\t Predicting options")
        print("\t > Batch Size:", batch_size)
        print("\t > Use Data Generator:", use_data_gen)
    
    ####
    if dataset_name == 'UT':
        dataset = UT
    elif dataset_name == 'SBU':
        dataset = SBU
    elif dataset_name == 'NTU':
        dataset = NTU
    elif dataset_name == 'YMJA':
        dataset = YMJA
    
    if verbose > 0:
        print("Reading data...")
    
    timesteps = data_kwargs['timesteps']
    add_joint_idx = data_kwargs.get('add_joint_idx', False)
    add_body_part = data_kwargs.get('add_body_part', False)
    
    if use_data_gen:
        if verbose > 0:
            print("> Using DataGenerator")
        val_generator = DataGenerator(dataset_name, dataset_fold, 'validation',
                batch_size=batch_size, reshuffle=False, shuffle_indiv_order=False,
                **data_kwargs)
        X_val, Y_val = val_generator[0]
        num_joints = len(X_val)//2
        object_shape = (len(X_val[0][0]),)
        
        videos_address = []
        prvs_video_idx = val_generator.seqs_mapping[0][0]
        pointer, num_seqs = 0, 0
        for video_idx, seq_idx in val_generator.seqs_mapping:
            if prvs_video_idx == video_idx:
                num_seqs += 1
            else:
                videos_address.append(slice(pointer, pointer+num_seqs))
                pointer += num_seqs
                num_seqs = 1
                prvs_video_idx = video_idx
        videos_address.append(slice(pointer, pointer+num_seqs))
    else:
        if verbose > 0:
            print("> Reading all data at once")
        X_val, Y_val = dataset.get_val(dataset_fold,selected_joints=selected_joints, 
            timesteps=timesteps, skip_timesteps=skip_timesteps, 
            normalization=normalization,
            add_joint_idx=add_joint_idx, add_body_part=add_body_part,
            sample_method=sample_method, seq_step=seq_step, flat_seqs=flat_seqs,
            )
    
        if flat_seqs: # Accuracy in this case will be per sequence and not per video
            num_joints = len(X_val)//2
            object_shape = (len(X_val[0][0]),)
        else:  # Accuracy in this case will be per video, after averaging the seqs
            num_joints = len(X_val[0][0])//2
            object_shape = (len(X_val[0][0][0]),)
            ## Flatten X_val at axis = 1 (num_seqs), and swap axis (1,0,2)
            ## Num_videos replaced by -> SUM num_seqs
            ## Keep "address" of each input, so unflatten from Y_pred can take place
            reshaped_X_val = [] # reshaped X_val dropping axis which represents the video seqs
            videos_address = [] # video address in reshaped_X_val
            pointer = 0
            for video_seqs in X_val:
                num_seqs = len(video_seqs)
                videos_address.append(slice(pointer, pointer+num_seqs))
                pointer += num_seqs
                reshaped_X_val += video_seqs
            X_val = np.array(reshaped_X_val).transpose((1,0,2)).tolist()
            
    overhead = add_joint_idx + add_body_part # True/False = 1/0
    num_dim = (object_shape[0]-overhead)//timesteps
    output_size = len(Y_val[0])
    
    if verbose > 0:
        print("Creating model...")
    model = get_model(num_objs=num_joints, object_shape=object_shape, 
        output_size=output_size, num_dim=num_dim, overhead=overhead,
        **model_kwargs)
    
    if verbose > 0:
        print("Loading weights...")
    model.load_weights(weights_path)
    
    if verbose > 0:
        print("Starting predicting...")
    
    if use_data_gen:
        ### Not using predict_generator because it might mixes the batches order
        ### Also, I have to read all Y_val anyway
        
        reshaped_Y_pred = []
        Y_val_flat = []
        if verbose > 0: progbar = progressbar.ProgressBar(max_value=len(val_generator))
        for batch_idx in range(len(val_generator)):
            if verbose > 0: progbar.update(batch_idx)
            batch_x, batch_y = val_generator[batch_idx]
            reshaped_Y_pred += list(model.predict_on_batch(batch_x))
            Y_val_flat += batch_y.tolist()
        if verbose > 0: progbar.finish()
        
        Y_val = []
        for video_address in videos_address:
            Y_val.append(Y_val_flat[video_address][0])
    else:
        reshaped_Y_pred = model.predict(X_val, batch_size=batch_size, 
            verbose=verbose)
    
    use_gauss_weight = True
    if use_gauss_weight:
        print("Averaging scores with Gaussian weights...")
    else:
        print("Averaging scores evenly, without weights...")
        
    if not flat_seqs: # Undo and avg reshaped_Y_pred (SUM num_seqs, ...) -> (Num_videos, ...)
        Y_pred = []
        for video_address in videos_address:
            if use_gauss_weight:
                avg_scores = np.average(reshaped_Y_pred[video_address], axis=0, 
                    weights=gauss(len(reshaped_Y_pred[video_address])) )
            else:
                avg_scores = np.average(reshaped_Y_pred[video_address], axis=0)
            Y_pred.append(avg_scores.tolist())
    else:
        Y_pred = reshaped_Y_pred
    
    acc_tensor = categorical_accuracy(Y_val, Y_pred)
    acc = K.eval(acc_tensor).mean()
    
    if verbose > 0:
        print("Validation acc: {:.2%}".format(acc))
    
    if return_acc:
        return acc
    else:
        return Y_pred, Y_val

def predict_fused_rn(fusion_weights_path, dataset_name, dataset_fold,
        config_filepaths, freeze_g_theta=False, fuse_at_fc1=False,
        batch_size=32, verbose=2, gpus=1):
        
    if verbose > 0:
        print("***** Predicting parameters *****")
        print("\t fusion_weights_path:", fusion_weights_path)
        print("\t Dataset:", dataset_name)
        print("\t Dataset fold:", dataset_fold)
        print("\t Fusion info")
        print("\t > config_filepaths:", config_filepaths)
        print("\t > freeze_g_theta:", freeze_g_theta)
        print("\t > fuse_at_fc1:", fuse_at_fc1)
        print("\t Predicting options")
        print("\t > Batch Size:", batch_size)
    
    data_kwargs, _, _ = read_config(config_filepaths[0])
    
    val_generator = DataGenerator(dataset_name, dataset_fold, 'validation',
            batch_size=batch_size, reshuffle=False, shuffle_indiv_order=False,
            **data_kwargs)
    X_val, Y_val = val_generator[0]
    
    num_joints = len(X_val)//2
    object_shape = (len(X_val[0][0]),)
    output_size = len(Y_val[0])
    
    if verbose > 0:
        print("Reading Y_val...")
    Y_val = []
    for batch_idx in range(len(val_generator)):
        _, y_val = val_generator[batch_idx]
        Y_val += y_val.tolist()
    
    models_kwargs = []
    for config_filepath in config_filepaths:
        data_kwargs, model_kwargs, train_kwargs = read_config(config_filepath)
        timesteps = data_kwargs['timesteps']
        add_joint_idx = data_kwargs['add_joint_idx']
        add_body_part = data_kwargs['add_body_part']
        overhead = add_joint_idx + add_body_part # True/False = 1/0
        num_dim = (object_shape[0]-overhead)//timesteps
        model_kwargs['num_dim'] = num_dim
        model_kwargs['overhead'] = overhead
        models_kwargs.append(model_kwargs)
    
    train_kwargs['drop_rate'] = 0
    weights_filepaths = [ [] for _ in config_filepaths ]
    
    if verbose > 0:
        print("Creating model...")
    model = fuse_rn(num_joints, object_shape, output_size, train_kwargs,
        models_kwargs, weights_filepaths, freeze_g_theta=freeze_g_theta, 
        fuse_at_fc1=fuse_at_fc1)

    if verbose > 0:
        print("Loading weights...")
    model.load_weights(fusion_weights_path)
    
    if verbose > 0:
        print("Starting predicting...")
    
    Y_pred = model.predict_generator(val_generator, max_queue_size=10, workers=5, 
        use_multiprocessing=True, verbose=verbose)
    
    acc_tensor = categorical_accuracy(Y_val, Y_pred)
    acc = K.eval(acc_tensor).mean()
    
    if verbose > 0:
        print("Validation acc: {:.2%}".format(acc))
        
    # Convert back from to_categorical
    Y_pred = np.argmax(Y_pred, axis=1, out=None).tolist()
    Y_val = np.argmax(Y_val, axis=1, out=None).tolist()
    
    return Y_pred, Y_val

def predict_fused_rn_seq(fusion_weights_path, dataset_name, dataset_fold, 
        config_filepaths, freeze_g_theta=False, fuse_at_fc1=False, flat_seqs=False,
        batch_size=32, verbose=2, gpus=1, return_acc = False, use_data_gen=True):
        
    if verbose > 0:
        print("***** Predicting parameters *****")
        print("\t fusion_weights_path:", fusion_weights_path)
        print("\t Dataset:", dataset_name)
        print("\t Dataset fold:", dataset_fold)
        print("\t Fusion info")
        print("\t > config_filepaths:", config_filepaths)
        print("\t > freeze_g_theta:", freeze_g_theta)
        print("\t > fuse_at_fc1:", fuse_at_fc1)
        print("\t Predicting options")
        print("\t > Batch Size:", batch_size)
        print("\t > flat_seqs:", flat_seqs)
        print("\t > Use Data Generator:", use_data_gen)
    
    ####
    if dataset_name == 'UT':
        dataset = UT
    elif dataset_name == 'SBU':
        dataset = SBU
    elif dataset_name == 'NTU':
        dataset = NTU
    elif dataset_name == 'YMJA':
        dataset = YMJA
    
    data_kwargs, _, _ = read_config(config_filepaths[0])
    
    data_kwargs['sample_method'] = 'all'
    data_kwargs['seq_step'] =data_kwargs.get('seq_step', data_kwargs['timesteps']//2)
    
    if verbose > 0:
        print("Reading data...")
    
    if use_data_gen:
        if verbose > 0:
            print("> Using DataGenerator")
        val_generator = DataGenerator(dataset_name, dataset_fold, 'validation',
                batch_size=batch_size, reshuffle=False, shuffle_indiv_order=False, **data_kwargs)
        X_val, Y_val = val_generator[0]
        num_joints = len(X_val)//2
        object_shape = (len(X_val[0][0]),)
        
        if verbose > 0:
            print("> Reading Y_val...")
        Y_val_flat = []
        for batch_idx in range(len(val_generator)):
            _, y_val = val_generator[batch_idx]
            Y_val_flat += y_val.tolist()
        
        videos_address = []
        prvs_video_idx = val_generator.seqs_mapping[0][0]
        pointer, num_seqs = 0, 0
        for video_idx, seq_idx in val_generator.seqs_mapping:
            if prvs_video_idx == video_idx:
                num_seqs += 1
            else:
                videos_address.append(slice(pointer, pointer+num_seqs))
                pointer += num_seqs
                num_seqs = 1
                prvs_video_idx = video_idx
        videos_address.append(slice(pointer, pointer+num_seqs))
        
        Y_val = []
        for video_address in videos_address:
            Y_val.append(Y_val_flat[video_address][0])
    else:
        if verbose > 0:
            print("> Reading all data at once")
        
        X_val, Y_val = dataset.get_val(dataset_fold, **data_kwargs)
    
        if flat_seqs: # Accuracy in this case will be per sequence and not per video
            num_joints = len(X_val)//2
            object_shape = (len(X_val[0][0]),)
        else:  # Accuracy in this case will be per video, after averaging the seqs
            num_joints = len(X_val[0][0])//2
            object_shape = (len(X_val[0][0][0]),)
            ## Flatten X_val at axis = 1 (num_seqs), and swap axis (1,0,2)
            ## Num_videos replaced by -> SUM num_seqs
            ## Keep "address" of each input, so unflatten from Y_pred can take place
            reshaped_X_val = [] # reshaped X_val dropping axis which represents the video seqs
            videos_address = [] # video address in reshaped_X_val
            pointer = 0
            for video_seqs in X_val:
                num_seqs = len(video_seqs)
                videos_address.append(slice(pointer, pointer+num_seqs))
                pointer += num_seqs
                reshaped_X_val += video_seqs
            X_val = np.array(reshaped_X_val).transpose((1,0,2)).tolist()
    
    output_size = len(Y_val[0])
   
    if verbose > 0:
        print("Creating model...")
    
    models_kwargs = []
    for config_filepath in config_filepaths:
        data_kwargs, model_kwargs, train_kwargs = read_config(config_filepath)
        timesteps = data_kwargs['timesteps']
        add_joint_idx = data_kwargs['add_joint_idx']
        add_body_part = data_kwargs['add_body_part']
        overhead = add_joint_idx + add_body_part # True/False = 1/0
        num_dim = (object_shape[0]-overhead)//timesteps
        model_kwargs['num_dim'] = num_dim
        model_kwargs['overhead'] = overhead
        models_kwargs.append(model_kwargs)
    
    train_kwargs['drop_rate'] = 0
    weights_filepaths = [ [] for _ in config_filepaths ]
    
    model = fuse_rn(num_joints, object_shape, output_size, train_kwargs,
        models_kwargs, weights_filepaths, freeze_g_theta=freeze_g_theta, 
        fuse_at_fc1=fuse_at_fc1)
    
    if verbose > 0:
        print("Loading weights...")
    model.load_weights(fusion_weights_path)
    
    if verbose > 0:
        print("Starting predicting...")
    
    if use_data_gen:
        reshaped_Y_pred = model.predict_generator(val_generator, max_queue_size=10,
            workers=5, use_multiprocessing=True, verbose=verbose)
    else:
        reshaped_Y_pred = model.predict(X_val, batch_size=batch_size, 
            verbose=verbose)
    
    use_gauss_weight = True
    if not flat_seqs: # Undo and avg reshaped_Y_pred (SUM num_seqs, ...) -> (Num_videos, ...)
        Y_pred = []
        for video_address in videos_address:
            if use_gauss_weight:
                avg_scores = np.average(reshaped_Y_pred[video_address], axis=0, 
                    weights=gauss(len(reshaped_Y_pred[video_address])) )
            else:
                avg_scores = np.average(reshaped_Y_pred[video_address], axis=0)
            Y_pred.append(avg_scores.tolist())
    
    acc_tensor = categorical_accuracy(Y_val, Y_pred)
    acc = K.eval(acc_tensor).mean()
    
    if verbose > 0:
        print("Validation acc: {:.2%}".format(acc))
        
    # Convert back from to_categorical
    Y_pred = np.argmax(Y_pred, axis=1, out=None).tolist()
    Y_val = np.argmax(Y_val, axis=1, out=None).tolist()
    
    if return_acc:
        return acc
    else:
        return Y_pred, Y_val

    

    
#%% Main
if __name__ == '__main__':
    args = vars(load_args())

    print('> Starting Predict RN - ', time.asctime( time.localtime(time.time()) ))

    print_args = args.pop('print_args')
    if print_args:
        print("Program arguments and values:")
        for argument, value in args.items():
            print('\t', argument, ":", value)
    
    predict_rn(**args)

    print('\n> Finished Predict RN -', time.asctime( time.localtime(time.time()) ))
