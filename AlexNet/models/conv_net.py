'''
Base class for convolutional networks
'''

import types
import warnings

import tensorflow as tf
import numpy as np

class ConvNet(object):
    '''
    Base class for convolutional networks
    '''
    def __init__(self, input_images, reuse_variables=False, n_layers=6, default_timesteps=8, 
                 data_format='NCHW', var_device='/cpu:0', model_name='convnet', random_seed=None):
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

        self.data_format = data_format
        self.var_device = var_device
        self.input = input_images
        self.reuse_variables = reuse_variables
        self.model_name = model_name
        self.batch_size = tf.shape(self.input)[0]
        self.n_layers = n_layers
        self.random_seed = random_seed
        self.weight_list = []

        self.readout_weights = {
            'b': None,
            'l': None,
            'bias': None
        }

        if self.input.get_shape().ndims == 5: 
            self.n_timesteps = self.input.get_shape().as_list()[1]
            self.is_sequence = True
        else:
            self.n_timesteps = default_timesteps
            self.is_sequence = False

        self.image_sizes = [None] * self.n_layers

        # normally altered in the network class inheriting from this
        
        self.output_size = 2 # number of classes
        self.layer_features = (  32,   32,   64,   64,  128,  128)
        self.k_size =         (   5,    5,    5,    5,    5,    5)
        self.stride_size =    (   1,    1,    1,    1,    1,    1)
        self.pool_size =      (None,    2,    2,    2,    2,    2)
        self.norm_type = None
        self.n_norm_groups = 32
        self.initializer = 'xavier'
        self.dropout_type = None
        self.keep_prob = 1.0
        self.top_time = 2 # number of timesteps that top-down input takes
        self.use_weight_norm = True
        self.trainable = True
        self.add_readout = True
        self.lateral_readout = None
        self.legacy_mode = True
        self.net_mode = 'train'
        self.scale_blt = [1, 1, 1]
        self.kernel_clip = False
        self.is_training = False # used for batch normalisation

        self.readout = [None] * self.n_timesteps
        self.activations = [[None] * self.n_layers for _ in range(self.n_timesteps)]

        self.get_params_ignore = [
            'readout', 'activations', 'weight_list', 'reuse_variables']

    def calculate_image_sizes(self):
        '''
        Computes the image sizes for each layer
        '''
        input_shape = self.input.get_shape().as_list()
        if self.data_format == 'NCHW':
            current_height, current_width = input_shape[-2], input_shape[-1]
        elif self.data_format == 'NHWC':
            current_height, current_width = input_shape[-3], input_shape[-2]

        for layer in range(self.n_layers):

            # check that the stride size is compatible
            try:
                assert current_height % self.stride_size[layer] == 0
                assert current_width % self.stride_size[layer] == 0
            except AssertionError:
                raise ValueError(
                    'the input size and width should be divisible by stride size, but are '
                    'height={0}, width={1}, pool size={2}'.format(
                        current_height, current_width, self.stride_size[layer]))

            # adjust image size for strides
            current_height = current_height // self.stride_size[layer]
            current_width = current_width // self.stride_size[layer]

            if self.pool_size[layer] is not None:

                # check that the pooling size is compatible
                try:
                    assert current_height % self.pool_size[layer] == 0
                    assert current_width % self.pool_size[layer] == 0
                except AssertionError:
                    raise ValueError(
                        'the input size and width should be divisible by pool size, but are'
                        'height={0}, width={1}, pool size={2}'.format(
                            current_height, current_width, self.pool_size[layer]))

                # adjust image size for pooling
                current_height = current_height // self.pool_size[layer]
                current_width = current_width // self.pool_size[layer]

            if self.data_format == 'NCHW':
                self.image_sizes[layer] = [
                    self.batch_size, self.layer_features[layer], current_height, current_width
                    ]

            elif self.data_format == 'NHWC':
                self.image_sizes[layer]= [
                    self.batch_size, current_height, current_width, self.layer_features[layer]
                    ]

            else:
                raise ValueError('data format should be either NCHW or NHWC not {0}'.format(
                    self.data_format))

    def euclidean_norm(self, input_tensor, axis):
        '''
        Returns the euclidean norm of the tensor reduced over the dimensions in axis

        Args:
            input_tensor: tensor to compute the euclidean norm over
            axis:         a sequence of the dimensions to reduce over

        Returns:
            norm: the euclidean norm
        '''
        return tf.sqrt(tf.reduce_sum(input_tensor**2, axis=axis))

    def get_weights(self, name, weight_shape, deconv=False):
        '''
        Returns weights initialised using weight normalisation

        Args:
            name:         base name of the weight variable
            weight_shape: shape of the weights
            deconv:       if true then the third dimension is treated as the output feature maps

        Returns:
            weights: the weight tensor
        '''
        if self.initializer == 'xavier':
            initializer = tf.contrib.layers.xavier_initializer(seed=self.random_seed)
        elif self.initializer == 'msra':
            initializer = tf.contrib.layers.variance_scaling_initializer(seed=self.random_seed)
        else:
            raise ValueError(
                'initializer should be \'msra\' or \'xavier\' but is {0}'.format(self.initializer))

        if self.use_weight_norm:
            if deconv:
                reduce_axes = [ax for ax, _ in enumerate(weight_shape)]
                reduce_axes.pop(-2)
                reshape_vector = [1, 1, -1, 1] # for g_vector
            else:
                reduce_axes = [ax for ax, _ in enumerate(weight_shape[:-1])]
                # reshape g vector, also works for linear layers
                reshape_vector = [1] * (len(weight_shape)-1) + [-1]

            with tf.device(self.var_device):
                v_tensor = tf.get_variable(name+'_v', shape=weight_shape, initializer=initializer,
                                           trainable=self.trainable)
                v_norm_init = self.euclidean_norm(v_tensor.initialized_value(), reduce_axes)
                if self.legacy_mode:
                    # earlier networks used a poorly named variable, legacy_mode allows simple
                    # backwards compatibility
                    g_vector = tf.get_variable(name+'_b', initializer=v_norm_init,
                                               trainable=self.trainable)
                else:
                    g_vector = tf.get_variable(name+'_g', initializer=v_norm_init,
                                               trainable=self.trainable)

            weights = tf.reshape(g_vector, reshape_vector)\
                    * tf.nn.l2_normalize(v_tensor, reduce_axes)
        else:
            weights = tf.get_variable(name, shape=weight_shape,
                                      initializer=initializer, trainable=self.trainable)
            
        return weights
    
    def get_conv_weights(self, name, input_tensor, layer, deconv=False):
        '''
        Returns initialized weights for convolutions using weight normalisation

        Args:
            name:         base name of the weight variable
            input_tensor: tensor corresponding to the input to the convolutional weights
            layer:        the layer the convolutional weights are being drawn from
            deconv:       if true then the third dimension is treated as the output feature maps    

        Returns:
            weights: the weight tensor        
        '''

        input_shape = input_tensor.get_shape().as_list()
    
        # check the shape of the input    
        if len(input_shape) != 4:
            raise ValueError('input to a convolutional layer should be 4-d, but is {0}-d'.format(
                len(input_shape)))

        # get the number of incoming features and the minimum of height and width dimensions
        if self.data_format == 'NCHW':
            in_features = input_shape[1]
            min_dim = min(input_shape[2:4])

        elif self.data_format == 'NHWC':
            in_features = input_shape[-1]
            min_dim = min(input_shape[1:3])

        else:
            raise ValueError('{0} is not a valid data format should be NCHW or NHWC'.format(
                self.data_format))

        # optionally implement kernel clipping if the kernel is too big for the input image
        if self.kernel_clip:
            k_size = min((self.k_size[layer], min_dim * 2 - 1))
        
        else:
            k_size = self.k_size[layer]
        
        # get the convolutional weights
        if deconv:
            weight_shape = [k_size, k_size, self.layer_features[layer], in_features]

        else:
            weight_shape = [k_size, k_size, in_features, self.layer_features[layer]]

        return self.get_weights(name, weight_shape, deconv=deconv)

    def conv2d(self, input_tensor, weights, stride_size=1):
        '''
        Performs standard 2d convolution with 1x1 stride and SAME padding

        Args:
            input_tensor: input tensor to the convolution
            weights:      the weights to use with the convolution
            stride:       optional int specifying stride for the convolution

        Returns:
            output_tensor: output of the convolution
        '''
        if self.data_format == 'NHWC':
            strides = [1, stride_size, stride_size, 1]
        else:
            strides = [1, 1, stride_size, stride_size]

        return tf.nn.conv2d(input_tensor, weights, strides, 'SAME', data_format=self.data_format)

    def deconv2d(self, input_tensor, weights, output_layer):
        '''
        Performs transpose of 2d convolution

        Args:
            input_tensor: input tensor to the convolution
            weights:      weights to use with the convolution
            output_layer: layer where the convolution outputs

        Returns:
            output_tensor: output of the convolution
        '''
        if self.pool_size[output_layer+1] is not None:
            stride = self.pool_size[output_layer+1]
        batch_size = tf.shape(input_tensor)[0]
        output_shape = tf.stack([batch_size] + self.image_sizes[output_layer][1:])
        return tf.nn.conv2d_transpose(input_tensor, weights, output_shape, [1, 1, stride, stride],
                                      'SAME', data_format=self.data_format)

    def conv2d_plus_b(self, input_tensor, weights, biases):
        '''
        Performs standard 2d convolution and adds biases

        Args:
            input_tensor: input tensor to the convolution
            weights:      weights to use with the convolution
            biases:       biases to add to the output of the convolution

        Returns:
            output_tensor: output of the convolution        
        '''
        return tf.nn.bias_add(
            self.conv2d(input_tensor, weights), biases, data_format=self.data_format)

    def max_pool(self, input_tensor, layer):
        '''
        Performs max pooling according to self.pool_size, which dictates both the kernel and stride
        size
        
        Args:
            input_tensor: to the max pool layer

        Returns:
            output_tensor: output from the max pooling
        '''
        if self.data_format == 'NCHW':
            pool_ksize = [1, 1, self.pool_size[layer], self.pool_size[layer]]
        elif self.data_format == 'NHWC':
            pool_ksize = [1, self.pool_size[layer], self.pool_size[layer], 1]
        else:
            raise ValueError('data format should be either NCHW or NHWC not {0}'.format(
                self.data_format))

        return tf.nn.max_pool(
            input_tensor, pool_ksize, pool_ksize, 'SAME', data_format=self.data_format)

    def dropout(self, input_tensor):
        '''
        Adds either Gaussian or Bernoulli dropout to the activations. For Bernoulli dropout,
        scaling is done during testing to ensure that the expected sum is the same

        Args:
            input_tensor: a 4-dimensional activation tensor

        Returns:
            activations: activations after dropout is applied
        '''
        assert self.dropout_type in ('gaussian', 'bernoulli')
        
        if self.dropout_type == 'bernoulli':
            return tf.nn.dropout(input_tensor, self.keep_prob)

        elif self.dropout_type == 'gaussian':
            std = tf.sqrt((1 - self.keep_prob) / self.keep_prob)
            dropout_mask = tf.random_normal(input_tensor.get_shape(), mean=1.0, stddev=std)
            activations = input_tensor * dropout_mask
            activations.set_shape(input_tensor.get_shape())
            return activations

        else:
            raise ValueError('dropout_type should be \'bernoulli\' or \'gaussian\' not {0}'.format(
                self.dropout_type))

    def norm_layer(self, input_tensor, time, variable_scope):
        '''
        Adds a normalisation layer to the network, either group or batch normalisation

        Args:
            input_tensor:   a 4-dimensional activation tensor
            norm_type:      a string specifying the normalisation type, either batch or group
            time:           current time step for the network
            variable_scope: the a string specifying the variable scope for the layer
        '''

        if self.norm_type in ('group', 'group-share'):
            # get axes for normalisation
            norm_axes = (-3, (-2, -1)) if self.data_format == 'NCHW' else (-1, (-3, -2))

            # get the variable scope for normalisation and whether to reuse variables
            if self.norm_type == 'group-share':
                # share group-normalisation parameters across time
                norm_variable_scope = '{0}/group_norm'.format(variable_scope)
                reuse = time > 0 or self.reuse_variables
            else:
                # unique group-normalisation parameters across time
                norm_variable_scope = '{0}/group_norm_t{1}'.format(variable_scope, time)
                reuse = self.reuse_variables
            
            with tf.variable_scope(norm_variable_scope, reuse=reuse):
                # perform group normalisation                
                norm_tensor = tf.contrib.layers.group_norm(
                    input_tensor, groups=self.n_norm_groups, channels_axis=norm_axes[0],
                    reduction_axes=norm_axes[1], center=True, scale=True, trainable=self.trainable)

        elif self.norm_type == 'batch':
            # get the dimension of the feature maps
            batch_norm_axis = 1 if self.data_format == 'NCHW' else 3

            with tf.variable_scope(
                '{0}/batch_norm_t{1}'.format(variable_scope, time), reuse=self.reuse_variables):
                # perform batch normlisation
                norm_tensor = tf.layers.batch_normalization(
                    input_tensor, training=self.is_training, trainable=self.trainable,
                    axis=batch_norm_axis, reuse=self.reuse_variables, fused=True)

        elif self.norm_type is None:
            warnings.warn(
                'normalisation layer applied but norm_type is None, skipping normalisation')

        else:
            raise ValueError('norm_type should be batch, group or group-share, but is {}'.format(
                self.norm_type))

        return norm_tensor

    def resize_to_layer(self, input_tensor, layer, name='resize_input'):
        '''
        Resizes input to match the layer size using nearest neighbour interpolation.
        Normally used for t-input

        Args:
            input_tensor: 4-d input tensor to be resized
            layer:        layer to resize to
            name:         used for name scope

        Returns:
            resized_input: the resized tensor
        '''

        if self.data_format == 'NCHW':
            temp_input = tf.transpose(input_tensor, [0, 2, 3, 1])
            resize_shape = self.image_sizes[layer][2:]
        elif self.data_format == 'NHWC':
            temp_input = input_tensor
            resize_shape = self.image_sizes[layer][1:3]
        else:
            raise ValueError('data format should be either NCHW or NHWC not {0}'.format(
                self.data_format))
    
        resized_input = tf.image.resize_images(temp_input, size=resize_shape,
                                               method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

        if self.data_format == 'NCHW':
            resized_input = tf.transpose(resized_input, [0, 3, 1, 2])

        return resized_input

    def calculate_l2_loss(self):
        '''
        Calculates the L2 loss over the weights.

        This works by looking for variables with the name weights.
        Any set of weights created with self.get_conv_weights should work
        '''
        with tf.name_scope('l2_loss'):
            l2_loss = tf.add_n([tf.nn.l2_loss(w) for w in self.weight_list])
        return l2_loss

    def global_avg_pool(self, input_tensor, name='global_avg_pool'):
        '''
        Performs global average pooling over feature maps

        Args:
            input_tensor: 4-d input to the global average pooling layer

        Returns:
            output_tensor: 4-d output from the global average pooling
        '''
        input_size = input_tensor.get_shape().as_list()
        if self.data_format == 'NCHW':
            pool_ksize = [1, 1, input_size[2], input_size[3]]
        elif self.data_format == 'NHWC':
            pool_ksize = [1, input_size[1], input_size[2], 1]
        else:
            raise ValueError('data format should be either NCHW or NHWC not {0}'.format(
                self.data_format))
        
        with tf.name_scope(name):
            return tf.nn.avg_pool(input_tensor, pool_ksize, [1, 1, 1, 1], 'VALID',
                                  data_format=self.data_format)

    def get_readout_input(self, time):
        '''
        Get inputs for the readout
        '''
        b_input = tf.contrib.layers.flatten(self.global_avg_pool(
            self.activations[time][-1]))
                        
        if time == 0:
            l_input = tf.zeros([self.batch_size, self.output_size])
        else:
            l_input = self.readout[time-1]

        return b_input, l_input

    def get_readout_weights(self, name, input_tensor):
        '''
        Returns weights for the linear readout initialised using weight normalisation

        Args:
            name:         base name for the weights
            input_tensor: tensor corresponing to the input to the readout weights

        Returns:
            weights: the weight tensor
        '''
        input_shape = input_tensor.get_shape().as_list()[-1]
        weight_shape = [input_shape, self.output_size]
        return self.get_weights(name, weight_shape)

    def get_lateral_readout_weights(self, l_input):
        '''
        Get lateral weights for the readout
        '''
        if self.lateral_readout == 'full':
            l_weights = self.get_readout_weights('l_weights', l_input)
        elif self.lateral_readout == 'self':
            l_weights = tf.get_variable(
                'self_weights', shape=[1], initializer=tf.random_uniform_initializer(0., 1.),
                trainable=self.trainable)
        elif self.lateral_readout is None:
            l_weights = None
        else:
            raise ValueError('invalid value {0} for lateral readout'.format(
                self.lateral_readout))

        return l_weights

    def compute_lateral_readout(self, l_input, l_weights):
        '''
        Compute lateral readout
        '''
        if self.lateral_readout == 'full':
            l_activation = tf.matmul(l_input, l_weights)
        elif self.lateral_readout == 'self':
            l_activation = l_input * tf.abs(l_weights)
        elif self.lateral_readout is None:
            l_activation = 0
        else:
            raise ValueError('invalid value {0} for lateral readout'.format(
                self.lateral_readout))

        return l_activation

    def readout_layer(self, time, name='readout'):
        '''
        Adds a linear readout layer to the network with optional lateral connections
        '''

        # GET INPUTS TO THE READOUT LAYER
        if time > 0 or self.reuse_variables:
            reuse = True
        else:
            reuse = False

        with tf.name_scope(name):
            b_input, l_input = self.get_readout_input(time)
        
            with tf.variable_scope('variables/readout', reuse=reuse), tf.device(self.var_device):
                b_weights = self.get_readout_weights('b_weights', b_input)
                l_weights = self.get_lateral_readout_weights(l_input)

                biases = tf.get_variable('biases', shape=[self.output_size],
                                        initializer=tf.constant_initializer(0.0),
                                        trainable=self.trainable)

                if time == 0:
                    # add weights to list used for L2 loss
                    self.weight_list.append(b_weights)
                    if l_weights is not None:
                        self.weight_list.append(l_weights)
                    # add weights to weight_dict for easy access during analysis
                    self.readout_weights['b'] = b_weights
                    self.readout_weights['l'] = l_weights
                    self.readout_weights['bias'] = biases     
            
            # PERFORM COMPUTATIONS FOR THE LAYER AND CALCULATE THE READOUT
            b_activation = tf.matmul(b_input, b_weights)
            l_activation = self.compute_lateral_readout(l_input, l_weights)

            self.readout[time] = b_activation + l_activation + biases

    def get_model_params(self, affix='model_params', ignore_attr=[]):
        '''
        Returns a dictionary containing parmeters defining the model ignoring any methods and
        any attributes in self.get_params_ignore or ignore_attr 
        '''

        model_param_dict = {}

        for arg in vars(self):
            # ignore methods
            if type(getattr(self, arg)) == types.MethodType:
                continue
            # ignore attributes in self.get_params_ignore or ignore_attr
            if (arg in self.get_params_ignore) or (arg in ignore_attr):
                continue            
            # add the parameter to the dictionary
            else:
                model_param_dict['{}_{}'.format(affix, arg)] = str(getattr(self, arg))

        return model_param_dict

    def recurrent_stride_warning(self):
        warnings.warn('stride is ignored for recurrent convolutions (only applied to feedforward)')
