'''
Implements a B network, using same style as the recurrent models, without ramping connections.

The readout does not have lateral connections
'''

import tensorflow as tf
import numpy as np
from .conv_net import ConvNet


class BNet(ConvNet):
    '''
    Class used for building model.

    Attributes:
        output_size:     the number of classes to output with the readout
        layer_features:  a sequence containing the number of feature maps in each layer
        k_size:          the kernel size to use for convolutions at each layer
        stride_size:     length of strides in the bottom-up convolutions
        pool_size:       the size of the pooling kernel at each layer, if None then no pooling is
                         performed
        norm_type:       type of normalisation to apply either 'batch', 'group' or
                         'group-share' (group-norm with parameters shared across time)
        n_norm_groups:   number of groups for group normalisation
                         (ignored if norm_type is None or batch)
        initializer:     type of initialiser to use msra or xavier
        dropout_type:    type of dropout to apply from gaussian or bernoulli, if None then no
                         dropout is applied
        keep_prob:       keep probability for dropout, set as a placeholder to vary from training to
                         validation        
        use_weight_norm: whether to implement the network weights with weight normalisation
        trainable:       whether to make the model variables trainable
        add_readout:     whether to add a readout to the model
        add_readout:     whether to add a readout to the model
        lateral_readout: type of lateral connections for the readout, either
                         'full' for fully connected, 'self' for self-connections or None for no
                         lateral connections
        legacy_mode:     changes variable naming to make it consistent with old networks that were
                         trained
        kernel_clip:     clips kernel size to  its effective size e.g. if we use a 7 x 7 kernel
                         for a 3 x 3 input image, then the kernel will be clipped to 5 x 5 kernel
                         (as only the central 5 x 5 section of the kernel will see the image)
    '''

    def __init__(self, input_images, reuse_variables=False, n_layers=8, default_timesteps=1, 
                 data_format='NCHW', var_device='/cpu:0', model_name='b_net', random_seed=None):
        '''
        Args:
            input_images:      input images to the network should be 4-d for images,
                               e.g. [batch, channels, height, width], or 5-d for movies,
                               e.g. [batch, time, channels, height, width]
            reuse_variables:   whether to create or reuse variables
            n_layers:          number of layers in the network
            default_timesteps: default number of time steps for the network, will be by the time
                               dimension in 5-d inputs
            data_format:       NCHW for GPUs and NHWC for CPUs
            var_device:        device for storing model variables (recognisable by tensorflow),
                               CPU is recommended when parallelising across GPUs, GPU is recommended
                               when a single GPU is used
            model_name:        name of model in the computational graph
            random_seed:       random seed for weight initialisation
        '''

        ConvNet.__init__(self, input_images, reuse_variables, n_layers, default_timesteps,
                         data_format, var_device, model_name, random_seed)

        # monkey patching is safe with these attributes prior to running build model.
        # monkey patching after running build_model is not advised

        self.output_size = 2 # number of classes
        self.layer_features = (  64,   64,   96,   96,  128,  128,  256,  256)
        self.k_size =         (   9,    9,    9,    9,    9,    9,    9,    9)
        self.stride_size =    (   1,    1,    1,    1,    1,    1,    1,    1)
        self.pool_size =      (None,    2,    2,    2,    2,    2,    2,    2)
        self.norm_type = None
        self.n_norm_groups = 32
        self.initializer = 'xavier'
        self.dropout_type = None
        self.keep_prob = 1.0
        self.use_weight_norm = False
        self.trainable = True
        self.add_readout = True
        self.lateral_readout = None
        self.legacy_mode = False
        self.kernel_clip = False

        # these attributes should not need altering if n_layers and n_timesteps are set appropriately

        self.activations = [[None] * self.n_layers for _ in range(self.n_timesteps)]
        self.readout = [None] * self.n_timesteps

        # allows easy access to network weights
        self.weight_dict = {
            'b': [None] * (self.n_layers), 
            'bias': [None] * (self.n_layers)
            }
        self.readout_weights = {
            'b': None,
            'l': None,
            'bias': None
        }

        # if the input is not a sequence then we only have to perform the convolution over the image
        # once, we save the output in this variable
        self.b_conv_image = None

    def b_layer(self, time, layer, name='b_layer'):
        '''
        Adds a B layer to the network
        '''

        # GET INPUTS TO THE B LAYER
        with tf.name_scope(name+'/inputs'):
            with tf.name_scope('b_input'):
                if layer == 0:
                    if self.is_sequence:
                        b_input = self.input[:, time]
                    else:
                        b_input = self.input
                else:
                    b_input = self.activations[time][layer-1]

                if self.pool_size[layer] is not None:
                    b_input = self.max_pool(b_input, layer)

        # GET THE VARIABLES FOR THE LAYER
        if time > 0 or self.reuse_variables:
            reuse = True
        else:
            reuse = False

        # get the convolutional weights
        variable_scope = 'variables/b_layer_l{0}'.format(layer)
        with tf.variable_scope(variable_scope, reuse=reuse), tf.device(self.var_device):
            b_weights = self.get_conv_weights('b_weights', b_input, layer)

            if time == 0:
                # add weights to list used for L2 loss
                self.weight_list.append(b_weights)
                # add weights to weight_dict for easy access during analysis
                self.weight_dict['b'][layer] = b_weights

            if self.norm_type is None:
                # get biases
                biases = tf.get_variable(
                    'biases', shape=[self.layer_features[layer]],
                    initializer=tf.constant_initializer(0.0),
                    trainable=self.trainable)
                if time == 0:
                    self.weight_dict['bias'][layer] = biases

        # PERFORM COMPUTATIONS FOR THE LAYER
        with tf.name_scope(name):
            conv_list = []
            
            # b conv
            with tf.name_scope('b_conv'):
                # if the input does not change then the same convolution is performed over the image
                # at each time step, used the save convolution to remove redundancy
                if time > 0 and layer == 0 and not self.is_sequence:
                    conv_list.append(self.b_conv_image)
                else:
                    conv_list.append(
                        self.conv2d(b_input, b_weights, stride_size=self.stride_size[layer]))

                # XXX save the convolution for the first layer on the first time step
                if time == 0 and layer == 0 and not self.is_sequence:
                    self.b_conv_image = conv_list[-1]

            # calculate the preactivation and activation
            with tf.name_scope('preactivation'):
                
                # add together incoming convolutions
                preactivation = tf.add_n(conv_list)

                if self.norm_type is None:
                    # add biases if no normalisation is performed
                    preactivation = tf.nn.bias_add(
                        preactivation, biases, data_format=self.data_format)

            with tf.name_scope('activation'):
                # apply dropout and batch norm before ReLU
                if self.norm_type is not None:
                    preactivation = self.norm_layer(preactivation, time, variable_scope)
                if self.dropout_type is not None:
                    preactivation = self.dropout(preactivation)
                # apply ReLU
                self.activations[time][layer] = tf.nn.relu(preactivation)
    
    def build_model(self):
        '''
        Builds the computational graph for the model
        '''
        try:
            assert len(self.layer_features) == self.n_layers
            assert len(self.k_size) == self.n_layers
            assert len(self.pool_size) == self.n_layers
            assert len(self.stride_size) == self.n_layers
        except AssertionError:
            raise ValueError(
            'layer_features, k_size, pool_size and stride_size all must have'
            'length equal to number of layers, but are {0}, {1}, {2}, {3}'.format(
                self.layer_features, self.k_size, self.pool_size, self.stride_size))
    
        # check if readout configurations are compatible
        if not self.add_readout and self.lateral_readout:
            raise ValueError('add_readout is {0} but lateral_readout {1}'.format(
                self.add_readout, self.lateral_readout))

        self.calculate_image_sizes()

        for time in range(self.n_timesteps):
            print('BUILDING: TIME {0}'.format(time))

            for layer in range(self.n_layers):
                print('BUILDING: LAYER {0}'.format(layer))
                print('calculated image size: {0}'.format(self.image_sizes[layer]))
                layer_name = self.model_name+'/b_layer_t{0}_l{1}'.format(time, layer)
                self.b_layer(time, layer, name=layer_name)
                print('actual image size:      {0}'.format(
                    self.activations[time][layer].get_shape().as_list()))

            if self.add_readout:
                print('BUILDING: READOUT')
                layer_name = self.model_name+'/readout_t{0}'.format(time)
                self.readout_layer(time, name=layer_name)

        print('MODEL BUILT SUCCESSFULLY')
