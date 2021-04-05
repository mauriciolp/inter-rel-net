from keras.layers import Dense, Dropout, Concatenate, Input
from keras.layers import Add, Maximum, Average, Subtract, Lambda
from keras.models import Model
from models.attention import IRNAttention

from keras import initializers

from keras import backend as K

def get_relevant_kwargs(kwargs, func):
    from inspect import signature
    keywords = signature(func).parameters.keys()
    new_kwargs = {}
    for key in keywords:
        if key in kwargs.keys():
            new_kwargs[key] = kwargs[key]
    
    return new_kwargs

def get_kernel_init(type, param=None, seed=None):
    kernel_init = None
    if type == 'glorot_uniform':
        kernel_init= initializers.glorot_uniform(seed=seed)
    elif type == 'VarianceScaling':
        kernel_init= initializers.VarianceScaling(seed=seed)
    elif type == 'RandomNormal':
        if param is None:
            param = 0.04
        kernel_init= initializers.RandomNormal(mean=0.0, stddev=param, seed=seed)
    elif type == 'TruncatedNormal':
        if param is None:
            param = 0.045 # Best for non-normalized coordinates
            # param = 0.09 # "Best" for normalized coordinates
        kernel_init= initializers.TruncatedNormal(mean=0.0, stddev=param, seed=seed)
    elif type == 'RandomUniform':
        if param is None:
            param = 0.055 # Best for non-normalized coordinates
            # param = ?? # "Best" for normalized coordinates
        kernel_init= initializers.RandomUniform(minval=-param, maxval=param, seed=seed)
        
    return kernel_init

def get_model(num_objs, object_shape, rel_type, output_size, 
        kernel_init_type='TruncatedNormal', kernel_init_param=0.045, kernel_init_seed=None,
        **f_and_g_kwargs):
    kernel_init = get_kernel_init(kernel_init_type, param=kernel_init_param, 
        seed=kernel_init_seed)
    
    f_phi_model = f_phi(num_objs, object_shape, rel_type, 
        kernel_init=kernel_init, **f_and_g_kwargs)
    
    if 'return_attention' in f_and_g_kwargs and f_and_g_kwargs['return_attention']:
        out_rn = Dense(output_size, activation='softmax', 
            kernel_initializer=kernel_init, name='model')(f_phi_model.output[0])
        model = Model(inputs=f_phi_model.input, outputs=[out_rn, f_phi_model.output[1]], name="rel_net")
    else:

        out_rn = Dense(output_size, activation='softmax', 
            kernel_initializer=kernel_init, name='softmax')(f_phi_model.output)
        model = Model(inputs=f_phi_model.input, outputs=out_rn, name="rel_net")
    
    return model

def fuse_rel_models(fuse_type, person1_joints, person2_joints, **g_theta_kwargs):
    if fuse_type == 'indiv_and_inter':
        g_theta_indiv = g_theta(model_name="g_theta_indivs", **g_theta_kwargs)
        indiv1_avg = create_relationships('p1_p1_all', g_theta_indiv, 
            person1_joints, person2_joints)
        
        indiv2_avg = create_relationships('p2_p2_all', g_theta_indiv, 
            person1_joints, person2_joints)
        
        g_theta_inter = g_theta(model_name="g_theta_inter", **g_theta_kwargs)
        inter_avg = create_relationships('p1_p2_all_bidirectional', 
            g_theta_inter, person1_joints, person2_joints)
        
        x = Concatenate()([indiv1_avg, indiv2_avg, inter_avg])
    elif fuse_type == 'indiv1_indiv2_inter':
        g_theta_indiv1 = g_theta(model_name="g_theta_indiv1", **g_theta_kwargs)
        indiv1_avg = create_relationships('p1_p1_all', g_theta_indiv1, 
            person1_joints, person2_joints)
        
        g_theta_indiv2 = g_theta(model_name="g_theta_indiv2", **g_theta_kwargs)
        indiv2_avg = create_relationships('p2_p2_all', g_theta_indiv2, 
            person1_joints, person2_joints)
        
        g_theta_inter = g_theta(model_name="g_theta_inter", **g_theta_kwargs)
        inter_avg = create_relationships('p1_p2_all_bidirectional', 
            g_theta_inter, person1_joints, person2_joints)
        
        x = Concatenate()([indiv1_avg, indiv2_avg, inter_avg])
    elif fuse_type == 'indiv1_inter':
        g_theta_indiv = g_theta(model_name="g_theta_indiv", **g_theta_kwargs)
        indiv1_avg = create_relationships('p1_p1_all', g_theta_indiv, 
            person1_joints, person2_joints)
        
        g_theta_inter = g_theta(model_name="g_theta_inter", **g_theta_kwargs)
        inter_avg = create_relationships('p1_p2_all_bidirectional', 
            g_theta_inter, person1_joints, person2_joints)
        
        x = Concatenate()([indiv1_avg, inter_avg])
    elif fuse_type == 'indiv1_indiv2':
        g_theta_indiv = g_theta(model_name="g_theta_indiv", **g_theta_kwargs)
        indiv1_avg = create_relationships('p1_p1_all', g_theta_indiv, 
            person1_joints, person2_joints)
        
        indiv2_avg = create_relationships('p2_p2_all', g_theta_indiv, 
            person1_joints, person2_joints)
        
        x = Concatenate()([indiv1_avg, indiv2_avg])
    elif fuse_type == 'indiv1_indiv2_bidirectional':
        g_theta_indiv = g_theta(model_name="g_theta_indiv", **g_theta_kwargs)
        indiv1_avg = create_relationships('p1_p1_all_bidirectional', g_theta_indiv, 
            person1_joints, person2_joints)
        
        indiv2_avg = create_relationships('p2_p2_all_bidirectional', g_theta_indiv, 
            person1_joints, person2_joints)
        
        x = Concatenate()([indiv1_avg, indiv2_avg])
    elif fuse_type == 'indiv1_indiv2_unshared':
        g_theta_indiv1 = g_theta(model_name="g_theta_indiv1", **g_theta_kwargs)
        indiv1_avg = create_relationships('p1_p1_all', g_theta_indiv1, 
            person1_joints, person2_joints)
        
        g_theta_indiv2 = g_theta(model_name="g_theta_indiv2", **g_theta_kwargs)
        indiv2_avg = create_relationships('p2_p2_all', g_theta_indiv2, 
            person1_joints, person2_joints)
        
        x = Concatenate()([indiv1_avg, indiv2_avg])
    elif fuse_type == 'inter1_inter2':
        g_theta_inter = g_theta(model_name="g_theta_inter", **g_theta_kwargs)
        inter1_avg = create_relationships('p1_p2_all', g_theta_inter, 
            person1_joints, person2_joints)
        
        inter2_avg = create_relationships('p1_p2_all', g_theta_inter, 
            person2_joints, person1_joints)
        
        x = Concatenate()([inter1_avg, inter2_avg])
    else:
        raise ValueError("Invalid fuse_type:", fuse_type)
    return x

def create_relationships(rel_type, g_theta_model, p1_joints, p2_joints, use_attention=False, use_relations=True, attention_proj_size=None, return_attention=False):
    g_theta_outs = []

    if not use_relations:
        for object_i in p1_joints:
            g_theta_outs.append(g_theta_model([object_i]))

        if use_attention:
            # Output may be tuple if return_attention is true, second element is attention vector
            return IRNAttention(projection_size=attention_proj_size, return_attention=return_attention)(g_theta_outs)
        else:
            rel_out = Average()(g_theta_outs)

        return rel_out
    
    if rel_type == 'inter' or rel_type == 'p1_p2_all_bidirectional':
        # All joints from person1 connected to all joints of person2, and back
        for object_i in p1_joints:
            for object_j in p2_joints:
                g_theta_outs.append(g_theta_model([object_i, object_j]))
        for object_i in p2_joints:
            for object_j in p1_joints:
                g_theta_outs.append(g_theta_model([object_i, object_j]))
        rel_out = Average()(g_theta_outs)
    elif rel_type == 'intra' or rel_type == 'indivs':

        indiv1_avg = create_relationships('p1_p1_all', g_theta_model, 
            p1_joints, p2_joints)
        
        indiv2_avg = create_relationships('p2_p2_all', g_theta_model, 
            p1_joints, p2_joints)
        
        rel_out = Concatenate()([indiv1_avg, indiv2_avg])
    elif rel_type == 'inter_and_indivs':
        # All joints from person1 connected to all joints of person2, and back
        for object_i in p1_joints:
            for object_j in p2_joints:
                g_theta_outs.append(g_theta_model([object_i, object_j]))
        for object_i in p2_joints:
            for object_j in p1_joints:
                g_theta_outs.append(g_theta_model([object_i, object_j]))
        
        # All joints from person1 connected to all other joints of itself
        for idx, object_i in enumerate(p1_joints):
            for object_j in p1_joints[idx+1:]:
            # for object_j in p1_joints[idx:]:
                g_theta_outs.append(g_theta_model([object_i, object_j]))
                
        # All joints from person2 connected to all other joints of itself
        for idx, object_i in enumerate(p2_joints):
            for object_j in p2_joints[idx+1:]:
            # for object_j in p2_joints[idx:]:
                g_theta_outs.append(g_theta_model([object_i, object_j]))
        
        rel_out = Average()(g_theta_outs)
    elif rel_type == 'p1_p2_all':
        # All joints from person1 connected to all joints of person2
        for object_i in p1_joints:
            for object_j in p2_joints:
                g_theta_outs.append(g_theta_model([object_i, object_j]))
        rel_out = Average()(g_theta_outs)
    elif rel_type == 'p1_p1_all':
        # All joints from person1 connected to all other joints of itself
        for idx, object_i in enumerate(p1_joints):
            for object_j in p1_joints[idx+1:]:
            # for object_j in p1_joints[idx:]:
                g_theta_outs.append(g_theta_model([object_i, object_j]))
        if use_attention:
            return IRNAttention(projection_size=attention_proj_size, return_attention=return_attention)(g_theta_outs)
        else:
            rel_out = Average()(g_theta_outs)
    elif rel_type == 'p2_p2_all':
        # All joints from person2 connected to all other joints of itself
        rel_out = create_relationships(
            'p1_p1_all', g_theta_model, p2_joints, p1_joints)
    elif rel_type == 'p1_p1_all_bidirectional':
        # All joints from person1 connected to all other joints of itself, and back
        rel_out = create_relationships(
            'p1_p2_all_bidirectional', g_theta_model, p1_joints, p1_joints)
    elif rel_type == 'p2_p2_all_bidirectional':
        # All joints from person2 connected to all other joints of itself, and back
        rel_out = create_relationships(
            'p1_p2_all_bidirectional', g_theta_model, p2_joints, p2_joints)
    elif rel_type == 'p1_p1_all-p2_p2_all':
        # All joints from person1 connected to all other joints of itself
        for idx, object_i in enumerate(p1_joints):
            for object_j in p1_joints[idx+1:]:
            # for object_j in p1_joints[idx:]:
                g_theta_outs.append(g_theta_model([object_i, object_j]))
        for idx, object_i in enumerate(p2_joints):
            for object_j in p2_joints[idx+1:]:
            # for object_j in p1_joints[idx:]:
                g_theta_outs.append(g_theta_model([object_i, object_j]))
        rel_out = Average()(g_theta_outs)
    else:
        raise ValueError("Invalid rel_type:", rel_type)
    
    return rel_out

def create_top(input_top, kernel_init, drop_rate=0, fc_units=[500,100,100], 
        fc_drop=False):

    x = Dropout(drop_rate)(input_top)
    
    x = Dense(fc_units[0], activation='relu', kernel_initializer=kernel_init, 
        name="f_phi_fc1")(x)
    if fc_drop: x = Dropout(drop_rate)(x)
    x = Dense(fc_units[1], activation='relu', kernel_initializer=kernel_init, 
        name="f_phi_fc2")(x)
    if fc_drop: x = Dropout(drop_rate)(x)
    x = Dense(fc_units[2], activation='relu', kernel_initializer=kernel_init, 
        name="f_phi_fc3")(x)
    
    return x

def f_phi(num_objs, object_shape, rel_type, kernel_init, fc_units=[500,100,100],
        drop_rate=0, fuse_type=None, fc_drop=False, use_attention=False, use_relations=True, projection_size=None, return_attention=False, **g_theta_kwargs):

    # For Joint Stream, similar structure except have one object for joint of both individuals
    # For Temporal stream, have one object per timestep
    # Object shape should correspond to timesteps * num_people (2) * num_dimension
    if rel_type == 'joint_stream' or rel_type == 'temp_stream':
        augmented_stream_objects = []

        for i in range(num_objs): # num_objs = num_joints or num_timesteps
            augmented_stream_input = Input(shape=object_shape, name="joint_object_"+str(i))
            augmented_stream_objects.append(augmented_stream_input)

        # G theta does not change, object size does not change
        g_theta_model = g_theta(object_shape, kernel_init=kernel_init, 
            drop_rate=drop_rate, use_relations=use_relations, model_name="g_theta_"+rel_type, **g_theta_kwargs)
        
        # Create relationships between all joint stream objects (similar to intra)
        x = create_relationships('p1_p1_all', g_theta_model, 
            augmented_stream_objects, None, use_attention=use_attention, use_relations=use_relations, attention_proj_size=projection_size, return_attention=return_attention)

        # Must unpack attention from x
        if return_attention:
            x, attention = x

        # Top of network f model    
        out_f_phi = create_top(x, kernel_init, fc_units=fc_units, drop_rate=drop_rate,
            fc_drop=fc_drop)
        
        f_phi_ins = augmented_stream_objects

        if return_attention:
            model = Model(inputs=f_phi_ins, outputs=[out_f_phi, attention], name="f_phi")
        else:
            model = Model(inputs=f_phi_ins, outputs=out_f_phi, name="f_phi")
        
        return model        

    else:
        person1_joints = []
        person2_joints = []

        for i in range(num_objs):
            object_i = Input(shape=object_shape, name="person1_object"+str(i))
            object_j = Input(shape=object_shape, name="person2_object"+str(i))
            person1_joints.append(object_i)
            person2_joints.append(object_j)
        
        if fuse_type is None:
            g_theta_model = g_theta(object_shape, kernel_init=kernel_init, 
                drop_rate=drop_rate, use_relations=use_relations, model_name="g_theta_"+rel_type, **g_theta_kwargs)
            x = create_relationships(rel_type, g_theta_model,
                person1_joints, person2_joints, use_attention=use_attention, use_relations=use_relations, attention_proj_size=projection_size, return_attention=return_attention)
            
            if return_attention: # Must unpack output
                x, attention = x
        else:
            x = fuse_rel_models(fuse_type, person1_joints, person2_joints,
                object_shape=object_shape, kernel_init=kernel_init, 
                drop_rate=drop_rate, **g_theta_kwargs)
        
        
        out_f_phi = create_top(x, kernel_init, fc_units=fc_units, drop_rate=drop_rate,
            fc_drop=fc_drop)
        
        f_phi_ins = person1_joints + person2_joints

        if return_attention:
            model = Model(inputs=f_phi_ins, outputs=[out_f_phi, attention], name="f_phi")
        else:
            model = Model(inputs=f_phi_ins, outputs=out_f_phi, name="f_phi")

        
        return model

def g_theta(object_shape, kernel_init, drop_rate=0, fc_drop=False, compute_distance=False, 
        compute_motion=False, use_relations=True, model_name="g_theta", num_dim=None, overhead=None):
    if compute_motion or compute_distance:
        timesteps = (object_shape[0]-overhead)//num_dim
    def euclideanDistance(inputs):
        if overhead > 0:
            trimmed = [ inputs[0][:,:-overhead], inputs[1][:,:-overhead] ]
        else:
            trimmed = inputs
        coords = [ K.reshape(obj, (-1, timesteps, num_dim) ) for obj in trimmed ] 
        output = K.sqrt(K.sum(K.square(coords[0] - coords[1]), axis=-1))
        return output
    def motionDistance(inputs):
        if overhead > 0:
            trimmed = [ inputs[0][:,:-overhead], inputs[1][:,:-overhead] ]
        else:
            trimmed = inputs
        shifted = [ trimmed[0][:,:-num_dim], trimmed[1][:,num_dim:] ] 
        coords = [ K.reshape(obj, (-1, timesteps-1, num_dim) ) for obj in shifted ]
        output = K.sqrt(K.sum(K.square(coords[0] - coords[1]), axis=-1))
        return output
    
    object_i = Input(shape=object_shape, name="object_i")
    g_theta_inputs = object_i

    if use_relations:
        object_j = Input(shape=object_shape, name="object_j")
        g_theta_inputs = [object_i, object_j]
    
    if compute_distance:
        distances = Lambda(euclideanDistance, 
            output_shape=lambda inp_shp: (inp_shp[0][0], timesteps),
            name=model_name+'_distanceMerge')([object_i, object_j])
        g_theta_inputs.append(distances)
    
    if compute_motion:
        motions = Lambda(motionDistance, 
            output_shape=lambda inp_shp: (inp_shp[0][0], timesteps-1),
            name=model_name+'_motionMerge')([object_i, object_j])
        g_theta_inputs.append(motions)
        
    if use_relations:
        x = Concatenate()(g_theta_inputs)
    else:
        x = g_theta_inputs
    
    x = Dense(1000, activation='relu', kernel_initializer=kernel_init,
        name=model_name+"_fc1")(x)
    if fc_drop: x = Dropout(drop_rate)(x)
    x = Dense(1000, activation='relu', kernel_initializer=kernel_init,
        name=model_name+"_fc2")(x)
    if fc_drop: x = Dropout(drop_rate)(x)
    x = Dense(1000, activation='relu', kernel_initializer=kernel_init,
        name=model_name+"_fc3")(x)
    if fc_drop: x = Dropout(drop_rate)(x)
    out_g_theta = Dense(500, activation='relu', kernel_initializer=kernel_init,
        name=model_name+"_fc4")(x)
        # name="g_theta_fc4")(x)
    
    if use_relations:
        model = Model(inputs=[object_i, object_j], outputs=out_g_theta, name=model_name)
    else:
        model = Model(inputs=[object_i], outputs=out_g_theta, name=model_name)
    
    return model

def fuse_rn(output_size, new_arch, train_kwargs,
        models_kwargs, weights_filepaths, freeze_g_theta=False, fuse_at_fc1=False, avg_at_end=False):

    prunned_models = []
    for model_kwargs, weights_filepath in zip(models_kwargs, weights_filepaths):
        model = get_model(output_size=output_size, **model_kwargs)
        if weights_filepath != []:
            model.load_weights(weights_filepath)
        
        if not fuse_at_fc1 and not avg_at_end:
            for layer in model.layers[::-1]: # reverse looking for last pool layer
                if layer.name.startswith(('average','concatenate','irn_attention')):
                    out_pool = layer.output
                    break
            prunned_model = Model(inputs=model.input, outputs=out_pool)
        elif fuse_at_fc1: # Prune keeping dropout + f_phi_fc1
            for layer in model.layers[::-1]: # reverse looking for last f_phi_fc1 layer
                if layer.name.startswith(('f_phi_fc1')):
                    out_f_phi_fc1 = layer.output
                    break
            prunned_model = Model(inputs=model.input, outputs=out_f_phi_fc1)
        elif avg_at_end:
            prunned_model = Model(inputs=model.input, outputs=model.output)
        
        if freeze_g_theta:
            for layer in prunned_model.layers: # Freezing model
                layer.trainable = False
        prunned_models.append(prunned_model)
    
    ### Train params
    drop_rate = train_kwargs.get('drop_rate', 0.1)
    kernel_init_type = train_kwargs.get('kernel_init_type', 'TruncatedNormal')
    kernel_init_param = train_kwargs.get('kernel_init_param', 0.045)
    kernel_init_seed = train_kwargs.get('kernel_init_seed')
    
    kernel_init = get_kernel_init(kernel_init_type, param=kernel_init_param, 
        seed=kernel_init_seed)
    
    if new_arch:
        # Building bottom
        joint_stream_objects = []
        temp_stream_objects = []

        for i in range(models_kwargs[0]['num_objs']):
            obj_joint = Input(shape=models_kwargs[0]['object_shape'], name="joint_stream_object"+str(i))
            joint_stream_objects.append(obj_joint)

        for i in range(models_kwargs[1]['num_objs']):
            obj_temp = Input(shape=models_kwargs[1]['object_shape'], name="temp_stream_object"+str(i))
            temp_stream_objects.append(obj_temp)
        
        inputs = joint_stream_objects + temp_stream_objects
        models_outs = [ prunned_models[0](joint_stream_objects), prunned_models[1](temp_stream_objects) ]

    else:
        # Building bottom
        person1_joints = []
        person2_joints = []
        for i in range(model_kwargs['num_objs']):
            object_i = Input(shape=models_kwargs[0]['object_shape'], name="person1_object"+str(i))
            object_j = Input(shape=models_kwargs[0]['object_shape'], name="person2_object"+str(i))
            person1_joints.append(object_i)
            person2_joints.append(object_j)
        inputs = person1_joints + person2_joints
        
        models_outs = [ m(inputs) for m in prunned_models ]
        
    if not avg_at_end:
        x = Concatenate()(models_outs)
            
        # Building top and Model
        top_kwargs = get_relevant_kwargs(model_kwargs, create_top) 
        x = create_top(x, kernel_init, **top_kwargs)
        out_rn = Dense(output_size, activation='softmax', 
            kernel_initializer=kernel_init, name='softmax')(x)
    else:
        # Model outputs already include individual soft max
        out_rn = Average()(models_outs)
    
    model = Model(inputs=inputs, outputs=out_rn, name="fused_rel_net")
    
    return model

