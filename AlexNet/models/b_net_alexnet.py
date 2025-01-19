'''
Implements a B network, using same style as the recurrent models, without ramping connections.

The readout does not have lateral connections
'''

import tensorflow as tf
import numpy as np
from .conv_net import ConvNet
from .alexnet import alexnet_v2, slim


class AlexNet(ConvNet):
    '''
    Class used for building model.

    Attributes:
        output_size:     the number of classes to output with the readout
        keep_prob:       keep probability for dropout, set as a placeholder to vary from training to
                         validation        
    '''

    def __init__(self, input_images, reuse_variables=False, n_layers=7, default_timesteps=1, 
                 data_format='NCHW', var_device='/cpu:0', model_name='b_net', random_seed=None):
        '''
        Args:
            input_images:      input images to the network should be 4-d for images,
                               e.g. [batch, channels, height, width], or 5-d for movies,
                               e.g. [batch, time, channels, height, width]
            reuse_variables:   whether to create or reuse variables
            data_format:       NCHW for GPUs and NHWC for CPUs
            var_device:        device for storing model variables (recognisable by tensorflow),
                               CPU is recommended when parallelising across GPUs, GPU is recommended
                               when a single GPU is used
        '''

        default_timesteps = 1
        ConvNet.__init__(self, input_images, reuse_variables, n_layers, default_timesteps,
                         data_format, var_device, model_name, random_seed)

        # the only things that matter
        self.output_size = 2 # number of classes
        self.activations = [[None] * self.n_layers for _ in range(self.n_timesteps)]
        self.readout = [None] * self.n_timesteps
        self.keep_prob = 0.5

        # set in the script but ignored
        self.dropout_type = None
        self.is_training = None

    def get_model_params(self, affix='model_params', ignore_attr=[]):
        '''
        Returns a dictionary containing parmeters defining the model ignoring any methods and
        any attributes in self.get_params_ignore or ignore_attr 
        '''

        model_param_dict = {"ALEXNET": "special case"}

        return model_param_dict


    def build_model(self):
        '''
        Builds the computational graph for the model
        '''

        with slim.arg_scope([slim.model_variable, slim.variable], device=self.var_device):
            readout, act, weights = alexnet_v2(
                self.input, self.output_size, dropout_keep_prob=self.keep_prob,
                reuse_variables=self.reuse_variables, data_format=self.data_format)

        self.readout = [readout]
        self.weight_list = weights
        self.activations = [act]
