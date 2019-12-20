from keras.layers import Dense, Input, Dropout
from keras.layers import LSTM, TimeDistributed, Lambda, Concatenate, Average
from keras.models import Model
from keras import backend as K

from . import rn

def get_irn(num_objs, object_shape, prune_at_layer=None, 
        **irn_kwargs):
    
    irn = rn.f_phi(num_objs, object_shape, **irn_kwargs)
    
    if prune_at_layer is not None:
        for layer in irn.layers[::-1]: # reverse looking for desired f_phi_fc* layer
            if layer.name.startswith(prune_at_layer) or layer.name.endswith(prune_at_layer):
                top_layer_out = layer.output
                break
        irn = Model(inputs=irn.input, outputs=top_layer_out)
    
    return irn

def g_theta_lstm(seq_len, object_shape, kernel_init, drop_rate, 
        prune_at_layer=None, **g_theta_kwargs):
    g_theta_model = rn.g_theta(object_shape, kernel_init, **g_theta_kwargs)
    
    if prune_at_layer is not None:
        # Reverse looking for desired g_theta_fc* layer
        for layer in g_theta_model.layers[::-1]: 
            if layer.name.endswith(prune_at_layer):
                top_layer_out = layer.output
                break
        g_theta_model = Model(inputs=g_theta_model.input, outputs=top_layer_out)
    
    # Wrapping with timedist
    input_g_theta = Input(shape=((2,)+object_shape))
    slice = Lambda(lambda x: [x[:,0], x[:,1]] )(input_g_theta)
    g_theta_model_out = g_theta_model(slice)
    merged_g_theta_model = Model(inputs=input_g_theta, outputs=g_theta_model_out)
    
    temporal_input = Input(shape=((seq_len, 2,) + object_shape))
    x = TimeDistributed(merged_g_theta_model)(temporal_input)
    
    x = LSTM(500, dropout=drop_rate, return_sequences=True)(x)
    g_theta_lstm_model = Model(inputs=temporal_input, outputs=x, name='g_theta_lstm')
    
    return g_theta_lstm_model

def average_per_sequence(tensors_list):
    expanded_dims = [ K.expand_dims(t, axis=1) for t in tensors_list ]
    single_tensor = K.concatenate(expanded_dims, axis=1)
    
    averages_per_seq = K.mean(single_tensor, axis=1)
    return averages_per_seq

def create_relationships(rel_type, g_theta_model, temp_input):
    num_objs = int(temp_input.shape[2])//2
    
    if rel_type == 'inter':
        # All joints from person1 connected to all joints of person2, and back
        g_theta_outs = []
        
        for idx_i in range(num_objs): # Indexes Person 1
            for idx_j in range(num_objs, num_objs*2): # Indexes Person 2
                pair_name = 'pair_p0-j{}_p1-j{}'.format(idx_i, idx_j-num_objs)
                slice = Lambda(lambda x: K.concatenate([x[:,:,idx_i:idx_i+1], 
                    x[:,:,idx_j:idx_j+1]], axis=2), name=pair_name)(temp_input)
                g_theta_outs.append(g_theta_model(slice))
        for idx_j in range(num_objs, num_objs*2):
            for idx_i in range(num_objs):
                pair_name = 'pair_p1-j{}_p0-j{}'.format(idx_j-num_objs, idx_i)
                slice = Lambda(lambda x: K.concatenate([x[:,:,idx_j:idx_j+1], 
                    x[:,:,idx_i:idx_i+1]], axis=2), name=pair_name )(temp_input)
                g_theta_outs.append(g_theta_model(slice))
        
        g_theta_merged_out = Lambda(average_per_sequence, name='avg_seqs')(g_theta_outs)
    elif rel_type == 'indivs' or rel_type == 'intra':
        # All joints from person connected to all other joints of itself
        g_theta_indiv1_outs = []
        for idx_i in range(num_objs): # Indexes Person 1
            for idx_j in range(idx_i+1, num_objs):
                pair_name = 'pair_p0-j{}_p0-j{}'.format(idx_i, idx_j)
                slice = Lambda(lambda x: K.concatenate([x[:,:,idx_i:idx_i+1], 
                    x[:,:,idx_j:idx_j+1]], axis=2), 
                    name=pair_name )(temp_input)
                g_theta_indiv1_outs.append(g_theta_model(slice))
                
        g_theta_indiv2_outs = []
        for idx_i in range(num_objs, num_objs*2): # Indexes Person 2
            for idx_j in range(idx_i+1, num_objs*2):
                pair_name = 'pair_p1-j{}_p1-j{}'.format(idx_i-num_objs, idx_j-num_objs)
                slice = Lambda(lambda x: K.concatenate([x[:,:,idx_i:idx_i+1], 
                    x[:,:,idx_j:idx_j+1]], axis=2), 
                    name=pair_name )(temp_input)
                g_theta_indiv2_outs.append(g_theta_model(slice))
        
        indiv1_avg = Lambda(average_per_sequence, name='avg_seqs_p0')(g_theta_indiv1_outs)
        indiv2_avg = Lambda(average_per_sequence, name='avg_seqs_p1')(g_theta_indiv2_outs)
        g_theta_merged_out = Concatenate()([indiv1_avg, indiv2_avg])
    else:
        raise ValueError("Invalid rel_type:"+rel_type)
        
    return g_theta_merged_out

def create_timedist_top(input_top, kernel_init, drop_rate=0, fc_units=[500,100,100],
        fc_drop=False):
    x = TimeDistributed(
        Dropout(drop_rate), name='timedist_dropout')(input_top)
    x = TimeDistributed(
        Dense(fc_units[0], activation='relu', kernel_initializer=kernel_init), 
        name='timedist_fc1')(x)
    if fc_drop:
        x = TimeDistributed(Dropout(drop_rate), name='timedist_dropout_1')(x)
    x = TimeDistributed(
        Dense(fc_units[1], activation='relu', kernel_initializer=kernel_init), 
        name='timedist_fc2')(x)
    if fc_drop:
        x = TimeDistributed(Dropout(drop_rate), name='timedist_dropout_2')(x)
    x = TimeDistributed(
        Dense(fc_units[2], activation='relu', kernel_initializer=kernel_init), 
        name='timedist_fc3')(x)
    
    return x

def get_model(num_objs, object_shape, output_size, seq_len=4, 
        num_lstms=1, prune_at_layer=None, lstm_location='top',
        kernel_init_type='TruncatedNormal', kernel_init_param=0.045, kernel_init_seed=None,
        **irn_kwargs):
    
    drop_rate = irn_kwargs.get('drop_rate', 0)
    
    kernel_init = rn.get_kernel_init(kernel_init_type, param=kernel_init_param, 
        seed=kernel_init_seed)
    
    temp_input = Input(shape=((seq_len, num_objs*2,) + object_shape))
    
    if lstm_location == 'top': # After f_phi
        irn_model = get_irn(num_objs, object_shape, prune_at_layer=prune_at_layer, 
                kernel_init=kernel_init, **irn_kwargs)
        
        # Creating model with merged input then slice, to apply TimeDistributed
        input_irn = Input(shape=((num_objs*2,)+object_shape))
        slice = Lambda(lambda x: [ x[:,i] for i in range(num_objs*2) ])(input_irn)
        irn_model_out = irn_model(slice)
        merged_irn_model = Model(inputs=input_irn, outputs=irn_model_out)
        
        # Wrapping merged model with TimeDistributed
        x = TimeDistributed(merged_irn_model)(temp_input)
    elif lstm_location == 'middle': # Between g_theta and f_phi
        irn_model = get_irn(num_objs, object_shape, 
            prune_at_layer=('average','concatenate'), 
            kernel_init=kernel_init, **irn_kwargs)
        
        # Creating model with merged input then slice, to apply TimeDistributed
        input_irn = Input(shape=((num_objs*2,)+object_shape))
        slice = Lambda(lambda x: [ x[:,i] for i in range(num_objs*2) ])(input_irn)
        irn_model_out = irn_model(slice)
        merged_irn_model = Model(inputs=input_irn, outputs=irn_model_out)
        
        # Wrapping merged model with TimeDistributed
        irn_out = TimeDistributed(merged_irn_model)(temp_input)
        lstm_out = LSTM(256, dropout=drop_rate, return_sequences=True)(irn_out)
        x = Concatenate()([irn_out, lstm_out])
        
        ### Replacing g_theta output - not promising results
        # lstm_out = LSTM(int(irn_out.shape[-1]), return_sequences=True)(irn_out)
        # x = lstm_out
        
        top_kwargs = rn.get_relevant_kwargs(irn_kwargs, create_timedist_top) 
        x = create_timedist_top(x, kernel_init, **top_kwargs)
    elif lstm_location == 'bottom': # At g_theta, after fc layers
        g_theta_kwargs = rn.get_relevant_kwargs(irn_kwargs, rn.g_theta)
        g_theta_kwargs['fc_drop'] = irn_kwargs.get('g_fc_drop', False)
        g_theta_lstm_model = g_theta_lstm(seq_len, object_shape, kernel_init, 
            drop_rate, **g_theta_kwargs)
        
        ### Ensuring back compatibility
        rel_type = irn_kwargs.get('rel_type')
        fuse_type = irn_kwargs.get('fuse_type')
        if rel_type not in ['inter', 'indivs']: # need to translate
            if fuse_type == 'indiv1_indiv2':
                rel_type = 'indivs'
            elif rel_type == 'p1_p2_all_bidirectional':
                rel_type = 'inter'
        
        g_theta_merged_out = create_relationships(rel_type, g_theta_lstm_model, 
            temp_input)
        
        top_kwargs = rn.get_relevant_kwargs(irn_kwargs, create_timedist_top) 
        x = create_timedist_top(g_theta_merged_out, kernel_init, **top_kwargs)
    
    if num_lstms == 2:
        x = LSTM(256, dropout=drop_rate, return_sequences=True)(x)
    x = LSTM(256, dropout=drop_rate)(x)
    
    out_softmax = Dense(output_size, activation='softmax', 
        kernel_initializer=kernel_init, name='softmax')(x)
    
    model = Model(inputs=temp_input, outputs=out_softmax, name="temp_rel_net")
    
    return model

def get_fusion_model(num_objs, object_shape, output_size, seq_len, train_kwargs,
        models_kwargs, weights_filepaths, freeze_g_theta=False, fuse_at_fc1=False):
    
    prunned_models = []
    for model_kwargs, weights_filepath in zip(models_kwargs, weights_filepaths):
        temp_model = get_model(num_objs=num_objs, object_shape=object_shape, 
            output_size=output_size, seq_len=seq_len, **model_kwargs)
        
        if weights_filepath != []:
            temp_model.load_weights(weights_filepath)
        
        for layer in temp_model.layers: # Looking for time_distributed layer
            if layer.name.startswith('time_distributed'):
                time_distributed_layer = layer
                break
        
        model = time_distributed_layer.layer.get_layer('f_phi')
        
        model_inputs = []
        for layer in model.layers:
            if layer.name.startswith('person'):
                model_inputs.append(layer.input)
        
        if not fuse_at_fc1:
            for layer in model.layers[::-1]: # reverse looking for last pool layer
                if layer.name.startswith(('average','concatenate')):
                    out_pool = layer.output
                    break
            prunned_model = Model(inputs=model_inputs, outputs=out_pool)
        else: # Prune keeping dropout + f_phi_fc1
            for layer in model.layers[::-1]: # reverse looking for last f_phi_fc1 layer
                if layer.name.startswith(('f_phi_fc1')):
                    out_f_phi_fc1 = layer.output
                    break
            prunned_model = Model(inputs=model_inputs, outputs=out_f_phi_fc1)
        
        if freeze_g_theta:
            for layer in prunned_model.layers: # Freezing model
                layer.trainable = False
        prunned_models.append(prunned_model)
    
    # Train params
    drop_rate = train_kwargs.get('drop_rate', 0.1)
    kernel_init_type = train_kwargs.get('kernel_init_type', 'TruncatedNormal')
    kernel_init_param = train_kwargs.get('kernel_init_param', 0.045)
    kernel_init_seed = train_kwargs.get('kernel_init_seed')
    
    kernel_init = rn.get_kernel_init(kernel_init_type, param=kernel_init_param, 
        seed=kernel_init_seed)
    
    # Building bottom
    person1_joints = []
    person2_joints = []
    for i in range(num_objs):
        object_i = Input(shape=object_shape, name="person1_object"+str(i))
        object_j = Input(shape=object_shape, name="person2_object"+str(i))
        person1_joints.append(object_i)
        person2_joints.append(object_j)
    inputs = person1_joints + person2_joints
    
    models_outs = [ m(inputs) for m in prunned_models ]
    x = Concatenate()(models_outs)
    
    # Building top and Model
    top_kwargs = rn.get_relevant_kwargs(model_kwargs, rn.create_top) 
    out_rn = rn.create_top(x, kernel_init, **top_kwargs)
    
    irn_model = Model(inputs=inputs, outputs=out_rn)
    
    # Wrapping with TimeDistributed
    
    input_irn = Input(shape=((num_objs*2,)+object_shape))
    slice = Lambda(lambda x: [ x[:,i] for i in range(num_objs*2) ])(input_irn)
    
    irn_model_out = irn_model(slice)
    merged_irn_model = Model(inputs=input_irn, outputs=irn_model_out)
    
    temp_input = Input(shape=((seq_len, num_objs*2,) + object_shape))
    x = TimeDistributed(merged_irn_model)(temp_input)
    
    lstm = LSTM(256, dropout=drop_rate)(x)
    
    out_softmax = Dense(output_size, activation='softmax', 
        kernel_initializer=kernel_init, name='softmax')(lstm)
    model = Model(inputs=temp_input, outputs=out_softmax, name="fused_temp_rel_net")
    
    return model

