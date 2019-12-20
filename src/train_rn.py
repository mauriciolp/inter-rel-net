import numpy as np
import argparse, sys, os, time

import tensorflow as tf
if int(tf.__version__.split('.')[1]) >= 14:
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

from keras.optimizers import SGD
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, TensorBoard, EarlyStopping, CSVLogger, Callback
    
from datasets import UT, SBU, NTU, NTU_V2
from datasets.data_generator import DataGenerator
from models.rn import get_model, fuse_rn
from misc.utils import read_config

#%% Functions
def load_args():
    ap = argparse.ArgumentParser(
        description='Train Relational Network.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    
    # Positional arguments
    ap.add_argument('output_path',
        help='path to output the model snapshots and log',
        type=str)
    
    # Data arguments
    ap.add_argument('-d','--dataset-name',
        help="dataset to be used for training",
        default='UT',
        choices=['UT', 'SBU', 'NTU_V2', 'NTU'])
    ap.add_argument('-f','--dataset-fold',
        help="dataset fold to be used for training",
        default=9,
        type=int)
    ap.add_argument('-t', '--timesteps',
        type=int,
        default=16,
        help='how many timesteps to use')
    
    # Training arguments
    ap.add_argument('-l', '--learning-rate',
        type=float,
        default=1e-4,
        help='learning rate for training')
    ap.add_argument('-r', '--drop-rate',
        type=float,
        default=0.3,
        help='dropout rate for training')
    ap.add_argument('-b', '--batch-size',
        type=int,
        default=32,
        help='batch size used to train')
    ap.add_argument('-G', '--gpus',
        type=int,
        default=1,
        help='number of gpus to use')
    ap.add_argument('-e', '--epochs',
        type=int,
        default=20,
        help='number of epochs to train')
    ap.add_argument('-c', '--checkpoint-period',
        type=int,
        default=100,
        help='interval (number of epochs) between checkpoints')
    
    ap.add_argument('--print_args',
        help="whether to print current arguments' values",
        action='store_true')
    
    args = ap.parse_args()
    
    return args

class AuxModelCheckpoint(Callback):
    """ Workaround for renaming checkpoint file and avoiding OSError
    
    Assumes the filename contains a '-temp' substring, that will be removed when 
    renaming the file, this way avoiding the lock conflict.
    """
    def __init__(self, filepath):
        super(AuxModelCheckpoint, self).__init__()
        self.filepath = filepath
        
        path, filename = os.path.split(self.filepath)
        new_filename = filename.replace('-temp','')
        new_filepath = os.path.join(path, new_filename)
        
        self.new_filepath = new_filepath
    
    def on_epoch_end(self, epoch, logs={}):
        if os.path.exists(self.filepath):
            os.rename(self.filepath, self.new_filepath)

def set_callbacks(output_path, checkpoint_period, batch_size, use_earlyStopping=True):
    callbacks_list = []
    
    checkpoint_filename = ("relnet_weights-temp.hdf5")
    filepath = os.path.join(output_path, checkpoint_filename)
    modelCheckpoint = ModelCheckpoint(filepath, verbose=0,
                    save_best_only=True, monitor='val_loss',
                    save_weights_only=True, period=checkpoint_period)
    callbacks_list.append(modelCheckpoint)

    auxModelCheckpoint = AuxModelCheckpoint(filepath)
    callbacks_list.append(auxModelCheckpoint)
    
    log_dir = output_path + '/logs'
    tensorBoard = TensorBoard(log_dir, batch_size=batch_size, write_graph=False)
    callbacks_list.append(tensorBoard)
    
    filepath = os.path.join(output_path, "relnet_weights-val_acc-temp.hdf5")
    modelCheckpoint_val_acc = ModelCheckpoint(filepath, verbose=0,
                    save_best_only=True, monitor='val_acc',
                    save_weights_only=True, period=checkpoint_period)
    callbacks_list.append(modelCheckpoint_val_acc)
    
    auxModelCheckpoint_val_acc = AuxModelCheckpoint(filepath)
    callbacks_list.append(auxModelCheckpoint_val_acc)
    
    if use_earlyStopping:
        earlyStopping = EarlyStopping(monitor='val_acc', min_delta=0, patience=100, 
            verbose=0)
        callbacks_list.append(earlyStopping)
    
    csvLogger = CSVLogger(output_path + '/training.log', append=True)
    callbacks_list.append(csvLogger)
    
    return callbacks_list

def train_model(model, verbose, learning_rate, output_path, checkpoint_period, 
        batch_size, epochs, use_data_gen, train_data, val_data, subsample_ratio,
        use_earlyStopping=True):
    if verbose > 0:
        print("Compiling model...")
    model.compile(loss='categorical_crossentropy',
                optimizer=Adam(lr=learning_rate),
                metrics=['accuracy'],
                )
    
    # Setting up Callbacks
    callbacks_list = set_callbacks(output_path, checkpoint_period, batch_size,
        use_earlyStopping=use_earlyStopping)
    
    if verbose > 0:
        print("Starting training...")
        
    if use_data_gen:
        train_generator = train_data
        val_generator = val_data
        
        steps_per_epoch = (None if subsample_ratio is None else
            int(subsample_ratio*len(train_generator)))
        validation_steps = None
        
        if subsample_ratio is not None and verbose > 0:
            print("Train num batches:", len(train_generator))
            print("Train steps:", steps_per_epoch)
        
        fit_history = model.fit_generator(train_generator,
            epochs=epochs,
            steps_per_epoch=steps_per_epoch,
            callbacks=callbacks_list,
            validation_data=val_generator,
            validation_steps=validation_steps,
            workers=4, max_queue_size=8, # Default is 10
            use_multiprocessing=True,
            verbose=verbose)
    else:
        X_train, Y_train = train_data
        X_val, Y_val = val_data
        fit_history = model.fit(X_train, Y_train,
            batch_size=batch_size,
            validation_data=(X_val, Y_val),
            verbose=verbose,
            epochs=epochs,
            steps_per_epoch=steps_per_epoch,
            validation_steps=validation_steps,
            callbacks=callbacks_list,
            shuffle=True)
    
    max_acc = np.max(fit_history.history['acc'])
    min_loss = np.min(fit_history.history['loss'])
    val_max_acc = np.max(fit_history.history['val_acc'])
    val_min_loss = np.min(fit_history.history['val_loss'])
    
    if verbose > 0:
        print("Train - Max ACC: {:.2%} Min loss: {:.4f}".format(max_acc, min_loss))
        print("Valid - Max ACC: {:.2%} Min loss: {:.4f}".format(val_max_acc, val_min_loss))
    
    return fit_history

def train_rn(output_path, dataset_name, model_kwargs, data_kwargs,
        dataset_fold=None, drop_rate=0.1, 
        batch_size=32, epochs=100, checkpoint_period=5, learning_rate=1e-4, 
        kernel_init_type='TruncatedNormal', kernel_init_param=0.045, kernel_init_seed=None,
        subsample_ratio=None, gpus=1, verbose=2, use_data_gen=True):
    
    if verbose > 0:
        print("***** Training parameters *****")
        print("\t Output path:", output_path)
        print("\t Dataset:", dataset_name)
        print("\t Dataset fold:", dataset_fold)
        print("\t Skeleton info")
        for key, value in data_kwargs.items():
            print("\t > {}: {}".format(key, value))
            
        print("\t Model info")
        for key, value in model_kwargs.items():
            print("\t > {}: {}".format(key, value))
        print("\t Kernel Init info")
        print("\t > kernel_init_type:", kernel_init_type)
        print("\t > kernel_init_param:", kernel_init_param)
        print("\t > kernel_init_seed:", kernel_init_seed)
        print("\t Training options")
        print("\t > Batch Size:", batch_size)
        print("\t > Epochs:", epochs)
        print("\t > Checkpoint Period:", checkpoint_period)
        print("\t > Learning Rate:", learning_rate)
        print("\t > Dropout rate:", drop_rate)
        print("\t > Training Subsample Ratio:", subsample_ratio)
    
    if dataset_name == 'UT':
        dataset = UT
    elif dataset_name == 'SBU':
        dataset = SBU
    elif dataset_name == 'NTU':
        dataset = NTU
        use_data_gen = True # Unable to read all data at once, dataset too big.
    
    if verbose > 0:
        print("Reading data...")
    
    timesteps = data_kwargs['timesteps']
    add_joint_idx = data_kwargs['add_joint_idx']
    add_body_part = data_kwargs['add_body_part']
    
    if use_data_gen:
        if verbose > 0:
            print("> Using DataGenerator")
        train_generator = DataGenerator(dataset_name, dataset_fold, 'train',
                batch_size=batch_size, reshuffle=True, shuffle_indiv_order=True, 
                **data_kwargs)
        val_generator = DataGenerator(dataset_name, dataset_fold, 'validation',
                batch_size=batch_size, reshuffle=False, shuffle_indiv_order=False,
                **data_kwargs)
        X_train, Y_train = train_generator[0]
        X_val, Y_val = val_generator[0]
        
        train_data = train_generator
        val_data = val_generator
    else:
        if verbose > 0:
            print("> Reading all data at once")
        X_train, Y_train = dataset.get_train(dataset_fold, **data_kwargs)
        X_val, Y_val = dataset.get_val(dataset_fold, **data_kwargs)
        
        train_data = [X_train, Y_train]
        val_data = [X_val, Y_val]
    
    num_joints = len(X_train)//2
    object_shape = (len(X_train[0][0]),)
    output_size = len(Y_train[0])
    overhead = add_joint_idx + add_body_part # True/False = 1/0
    num_dim = (object_shape[0]-overhead)//timesteps
    
    if verbose > 0:
        print("Creating model...")
    
    model = get_model(num_objs=num_joints, object_shape=object_shape, 
        output_size=output_size, num_dim=num_dim, overhead=overhead,
        kernel_init_type=kernel_init_type, kernel_init_param=kernel_init_param, 
        kernel_init_seed=kernel_init_seed, drop_rate=drop_rate,
        **model_kwargs)
    
    fit_history = train_model(model=model, verbose=verbose, learning_rate=learning_rate, 
        output_path=output_path, checkpoint_period=checkpoint_period, 
        batch_size=batch_size, epochs=epochs, use_data_gen=use_data_gen, 
        train_data=train_data, val_data=val_data, subsample_ratio=subsample_ratio)
    
    return fit_history

def train_fused_rn(output_path, dataset_name, dataset_fold,
        config_filepaths, weights_filepaths,
        batch_size=32, epochs=100, checkpoint_period=5, learning_rate=1e-4, 
        drop_rate=0.1, freeze_g_theta=False, fuse_at_fc1=False,
        initial_epoch=0, initial_weights=None, use_data_gen = True,
        subsample_ratio=None,
        gpus=1,verbose=2):
    
    data_kwargs, _, _ = read_config(config_filepaths[0])
    
    if verbose > 0:
        print("***** Training parameters *****")
        print("\t Output path:", output_path)
        print("\t Dataset:", dataset_name)
        print("\t Dataset fold:", dataset_fold)
        print("\t Fusion info")
        print("\t > config_filepaths:", config_filepaths)
        print("\t > weights_filepaths:", weights_filepaths)
        print("\t > freeze_g_theta:", freeze_g_theta)
        print("\t > fuse_at_fc1:", fuse_at_fc1)
        print("\t Training options")
        print("\t > Batch Size:", batch_size)
        print("\t > Epochs:", epochs)
        print("\t > Checkpoint Period:", checkpoint_period)
        print("\t > Learning Rate:", learning_rate)
        print("\t > Dropout rate:", drop_rate)
        print("\t > Training Subsample Ratio:", subsample_ratio)
        print("\t GPUs:", gpus)
        print("\t Skeleton info")
        for key, value in data_kwargs.items():
            print("\t > {}: {}".format(key, value))
    
    if dataset_name == 'UT':
        dataset = UT
    elif dataset_name == 'SBU':
        dataset = SBU
    
    if verbose > 0:
        print("Reading data...")
    
    if use_data_gen:
        train_generator = DataGenerator(dataset_name, dataset_fold, 'train',
                batch_size=batch_size, reshuffle=True, shuffle_indiv_order=True, 
                **data_kwargs)
        val_generator = DataGenerator(dataset_name, dataset_fold, 'validation',
                batch_size=batch_size, reshuffle=False, shuffle_indiv_order=False,
                **data_kwargs)
        X_train, Y_train = train_generator[0]
        X_val, Y_val = val_generator[0]
        train_data = train_generator
        val_data = val_generator
    else:
        X_train, Y_train = dataset.get_train(dataset_fold, **data_kwargs)
        X_val, Y_val = dataset.get_val(dataset_fold, **data_kwargs)
        train_data = [X_train, Y_train]
        val_data = [X_val, Y_val]
    
    num_joints = len(X_train)//2
    object_shape = (len(X_train[0][0]),)
    output_size = len(Y_train[0])
    
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
    
    train_kwargs['drop_rate'] = drop_rate
    if verbose > 0:
        print("Creating model...")
    model = fuse_rn(num_joints, object_shape, output_size, train_kwargs,
        models_kwargs, weights_filepaths, freeze_g_theta=freeze_g_theta, 
        fuse_at_fc1=fuse_at_fc1)
    
    if initial_weights is not None:
        model.load_weights(initial_weights)
    
    fit_history = train_model(model=model, verbose=verbose, learning_rate=learning_rate, 
        output_path=output_path, checkpoint_period=checkpoint_period, 
        batch_size=batch_size, epochs=epochs, use_data_gen=use_data_gen, 
        train_data=train_data, val_data=val_data, subsample_ratio=subsample_ratio)
    
    return fit_history


    
#%% Main
if __name__ == '__main__':
    args = vars(load_args())

    print('> Starting Train RN - ', time.asctime( time.localtime(time.time()) ))

    print_args = args.pop('print_args')
    if print_args:
        print("Program arguments and values:")
        for argument, value in args.items():
            print('\t', argument, ":", value)
    
    train_rn(**args)

    print('\n> Finished Train RN -', time.asctime( time.localtime(time.time()) ))
