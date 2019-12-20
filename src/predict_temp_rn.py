import numpy as np
import argparse, sys, os, time
import progressbar

import tensorflow as tf
if int(tf.__version__.split('.')[1]) >= 14:
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

from keras.metrics import categorical_accuracy
import keras.backend as K
    
from datasets import UT, SBU, NTU
from datasets.data_generator import DataGeneratorSeq
from misc.utils import read_config
from models.temporal_rn import get_model, get_fusion_model

def run_predict(model, verbose, val_generator):
    if verbose > 0:
        print("Starting predicting...")
    
    Y_pred = []
    Y_val = []
    if verbose > 0: progbar = progressbar.ProgressBar(max_value=len(val_generator))
    for batch_idx in range(len(val_generator)):
        if verbose > 0: progbar.update(batch_idx)
        batch_x, batch_y = val_generator[batch_idx]
        Y_pred += list(model.predict_on_batch(batch_x))
        Y_val += batch_y.tolist()
    if verbose > 0: progbar.finish()
    
    y_true = np.array(Y_val)
    y_pred = np.array(Y_pred)

    n_hits = np.sum(y_true.argmax(axis=-1) == y_pred.argmax(axis=-1))
    acc = n_hits/y_true.shape[0]
    
    if verbose > 0:
        print("Validation acc: {:.2%}".format(acc))
        
    # Convert back from to_categorical
    Y_pred = np.argmax(Y_pred, axis=1).tolist()
    Y_val = np.argmax(Y_val, axis=1).tolist()
    
    return Y_pred, Y_val

#%% Functions
def predict_temp_rn(weights_path, dataset_name, model_kwargs, data_kwargs,
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
        print("\t Predicting options")
        print("\t > Batch Size:", batch_size)
    
    if verbose > 0:
        print("Initializing Data Generator...")
    val_generator = DataGeneratorSeq(dataset_name, dataset_fold, 'validation',
            batch_size=batch_size, reshuffle=False, shuffle_indiv_order=False,
            pad_sequences=True, **data_kwargs)
    X_val, Y_val = val_generator[0]
    
    timesteps = data_kwargs['timesteps']
    add_joint_idx = data_kwargs['add_joint_idx']
    add_body_part = data_kwargs['add_body_part']
    
    _, seq_len, num_joints, *object_shape = np.array(X_val).shape
    num_joints = num_joints//2
    object_shape = tuple(object_shape)
    output_size = len(Y_val[0])
    overhead = add_joint_idx + add_body_part # True/False = 1/0
    num_dim = (object_shape[0]-overhead)//timesteps
    
    if verbose > 0:
        print("Creating model...")
    model = get_model(num_objs=num_joints, object_shape=object_shape, 
        output_size=output_size, num_dim=num_dim, overhead=overhead,
        seq_len=seq_len, **model_kwargs)
    
    if verbose > 0:
        print("Loading weights...")
    model.load_weights(weights_path)
    
    Y_pred, Y_val = run_predict(model, verbose, val_generator)
    
    return Y_pred, Y_val


def predict_fused_temp_rn(fusion_weights_path, dataset_name, dataset_fold,
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
    
    ####    
    data_kwargs, _, _ = read_config(config_filepaths[0])
    
    if verbose > 0:
        print("Initializing Data Generator...")
    val_generator = DataGeneratorSeq(dataset_name, dataset_fold, 'validation',
            batch_size=batch_size, reshuffle=False, shuffle_indiv_order=False,
            pad_sequences=True, **data_kwargs)
    X_val, Y_val = val_generator[0]
    
    _, seq_len, num_joints, *object_shape = np.array(X_val).shape
    num_joints = num_joints//2
    object_shape = tuple(object_shape)
    output_size = len(Y_val[0])
    
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
    model = get_fusion_model(num_joints, object_shape, output_size, seq_len, 
        train_kwargs, models_kwargs, weights_filepaths, 
        freeze_g_theta=freeze_g_theta, fuse_at_fc1=fuse_at_fc1)

    if verbose > 0:
        print("Loading weights...")
    model.load_weights(fusion_weights_path)
    
    if verbose > 0:
        print("Starting predicting...")
    
    Y_pred = []
    Y_val = []
    if verbose > 0: progbar = progressbar.ProgressBar(max_value=len(val_generator))
    for batch_idx in range(len(val_generator)):
        if verbose > 0: progbar.update(batch_idx)
        batch_x, batch_y = val_generator[batch_idx]
        Y_pred += list(model.predict_on_batch(batch_x))
        Y_val += batch_y.tolist()
    if verbose > 0: progbar.finish()
    
    y_true = np.array(Y_val)
    y_pred = np.array(Y_pred)

    n_hits = np.sum(y_true.argmax(axis=-1) == y_pred.argmax(axis=-1))
    acc = n_hits/y_true.shape[0]
    
    if verbose > 0:
        print("Validation acc: {:.2%}".format(acc))
        
    # Convert back from to_categorical
    Y_pred = np.argmax(Y_pred, axis=1).tolist()
    Y_val = np.argmax(Y_val, axis=1).tolist()
    
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
    
    predict_fused_rn(**args)

    print('\n> Finished Predict RN -', time.asctime( time.localtime(time.time()) ))
