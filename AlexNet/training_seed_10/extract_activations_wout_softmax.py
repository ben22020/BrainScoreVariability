'''
Extracts RDMs for networks and saves them in the format [layer, time, distances]
'''

import glob
import os
import sys

import numpy as np
from PIL import Image
from progressbar import ProgressBar
from scipy.misc import imresize
from scipy.spatial.distance import pdist
from scipy.io import loadmat, savemat

main_dir = sys.argv[1] # /imaging/jm03/01_projects/05_DNNs/180118_blt_rcnn/copy_from_HPC_20190509_ecoset_content_paper/PNAS_resubmission_AlexNet_courtney/alexnet/alexnet_ecoset_momentum_seed_01

# increase the logging threshold
os.environ["TF_CPP_MIN_LOG_LEVEL"]="2"
import tensorflow as tf

# path to the checkpoint to restore
CKPT_PATH = os.path.join(main_dir, 'trained_networks/alexnet/model.ckpt_epoch89')

# size to reshape the image to
IMAGE_SIZE = 224

# path to save the RDMs
RDM_SAVE_PATH = os.path.join(main_dir, 'RDMs')

# path to save the RDMs
RDM_SAVE_ACTIVATIONS = os.path.join(main_dir, 'activations')

# directory of images to use for RDMs
IMAGE_DIR = '/imaging/jm03/01_projects/05_DNNs/180118_blt_rcnn/from_images_to_activations/ecoset_PNAS_resubmission_face_selective_cells_combined'

# training seed
TRAINING_SEED = int(main_dir[-2:]) #1 
#print(TRAINING_SEED)
#print(type(TRAINING_SEED))
#os._exit()


# model object defining the network to use
sys.path.append(main_dir)
from models.b_net_alexnet import AlexNet as model
MODEL_CLASS = model

# output size of the network
haed_tail = os.path.split(main_dir)
if 'ecoset' in haed_tail[1]:
    OUTPUT_SIZE = 565
    DNN_training_image_set = 'ecoset'
elif 'imagenet' in haed_tail[1]:
    OUTPUT_SIZE = 1000
    DNN_training_image_set = 'imagenet'
#print(OUTPUT_SIZE)
#os.exit()

# number of time steps to run the network for
N_TIMESTEPS = 1

# type of dropout to use in the network
DROPOUT_TYPE = 'bernoulli'

# keep probability to use for dropout (ignored if None)
KEEP_PROB = 0.5

# number of samples to take per image (useful when using dropout)
SAMPLES_PER_IMAGE = 1

# random seed for the graph (useful when using dropout)
RANDOM_SEED = 0

# GPU to use for the network (set to negative for no GPU)
GPU_ID = 0

# distance measure, must be compatible with scipy.spatial.distance.pdist
DISTANCE_MEASURE = 'correlation'


def central_crop(image):
    '''
    Get the central crop for the images

    Args:
        image: image in NCHW format

    Returns:
        cropped_image: central crop of the image
    '''
    
    image_shape = np.shape(image)

    if len(image_shape) != 3 or image_shape[2] != 3:
        raise ValueError('image should be RGB in HWC format but is shape {0}'.format(image_shape))

    # find the smallest side
    min_dim = min(image_shape[:2])

    # get the coordinates for the crop
    start_h = (image_shape[0] - min_dim) // 2
    start_w = (image_shape[1] - min_dim) // 2

    # crop the image
    cropped_image = image[start_h:(start_h + min_dim), start_w:(start_w + min_dim)]

    return cropped_image

def get_images(image_dir, image_size):
    '''
    Loads images from the directory takes the central crop and loads them into a numpy array

    Args:
        image_dir:  directory containing images to load
        image_size: size to reshape images to after loading

    Returns:
        images: a numpy array in NHWC format
    '''

    # get all files in the directory and sort them
    dir_files = sorted(glob.glob(os.path.join(image_dir, '*')))

    #print(image_dir)
    #print(dir_files)
    #os._exit()

    # initialise the list to store images
    images_list = []

    # keep track of ignored files in folder
    ignored_files = 0

    for fname in dir_files:
        
        # load the image checking if it is a valid file type
        try:
            img_i = Image.open(fname)
        
        except IOError:
            print('not a recognised image file, ignoring: {0}'.format(fname))
            ignored_files += 1
            continue

        # force the image to be RGB and convert to an array
        img_i_array = np.array(img_i.convert('RGB'))

        # crop the image
        cropped_image = central_crop(img_i_array)

        # resize the image
        resized_image = imresize(cropped_image, (image_size, image_size), 'bilinear')

        images_list.append(resized_image)

    # convert images to an array
    images = np.array(images_list)

    print('{0} images loaded, {1} files ignored'.format(images.shape[0], ignored_files))

    return images


def extract_rdms(images, ckpt_path, model_builder, n_timesteps, output_size, dropout_type,
                 keep_prob=None, samples_per_image=1, random_seed=0, gpu_id=-1,
                 distance_measure='correlation'):
    '''
    Extracts RDMs for specifed images and network

    Args:
        images:
        ckpt_path:
        model_builder:
        n_timesteps:
        output_size: 
        dropout_type:
        keep_prob:
        samples_per_image:
        random_seed:
        gpu_ids:
        distance_measure:

    Returns:
        rdms: RDMs in layers, images, time, distances format
    '''
    
    # set the visible devices for the network
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
    if gpu_id >= 0:
        model_device = '/gpu:0'
        data_format = 'NCHW'
        images = np.transpose(images, (0, 3, 1, 2))
    else:
        model_device = '/cpu:0'
        data_format = 'NHWC'


    # build the graph for the network
    graph = tf.Graph()
    with graph.as_default():

        # set the graph based random seed
        tf.set_random_seed(random_seed)

        # define the images placeholder
        img_ph = tf.placeholder(tf.uint8, np.shape(images)[1:], 'images_ph')
        
        # rescale the image
        image_float32 = tf.image.convert_image_dtype(img_ph, tf.float32)
        image_rescaled = (image_float32 - 0.5) * 2

        # tile for the given number of samples
        image_tiled = tf.tile(tf.expand_dims(image_rescaled, 0), [samples_per_image, 1, 1, 1])

        model = model_builder(
            image_tiled, var_device=model_device, default_timesteps=n_timesteps,
            data_format=data_format, random_seed=random_seed)
        model.output_size = output_size
        model.dropout_type = dropout_type
        model.keep_prob = keep_prob
        model.build_model()

        # get the readout and activations averaged over samples per image (mainly for using dropout)
        activations = [tf.reduce_mean(act_l, axis=1) for act_l in zip(*model.activations)]
        readout = tf.reduce_mean(model.readout, axis=1)

        with tf.device('/cpu:0'):
            # create the save object for restoring the graph
            saver = tf.train.Saver()

    # start the session
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Session(config=config, graph=graph) as sess:
        
        # restore the network
        saver.restore(sess, ckpt_path)

        # iterate through each layer to save memory
        print('COMPUTING RDMS FOR EACH LAYER')
        bar = ProgressBar(max_value=model.n_layers + 1)
        bar.start()
        rdms = []
        activation_dict = {}
        for layer in range(model.n_layers):
            # get activations
            layer_act = [sess.run(activations[layer], feed_dict={img_ph: img}) for img in images]
            layer_act = np.stack(layer_act, axis=1) #  format [time, images, ...]
            layer_act = layer_act.reshape(model.n_timesteps, images.shape[0], -1) # flatten activations
            
            layer_act_squeezed = np.squeeze(layer_act)
            #print(layer_act.shape)
            #print(layer_act[:50].shape)
            #print(layer_act[50:].shape)
            #faces = layer_act[:50]
            #places = layer_act[50:]
            #print(faces[-1][:5])
            #print(places[0][:5])
            #os._exit()
            
            activation_dict_faces = {}
            activation_dict_faces['layer_{0}_faces'.format(layer + 1)] = layer_act_squeezed[:50]
            activation_dict_places = {}
            activation_dict_places['layer_{0}_places'.format(layer + 1)] = layer_act_squeezed[50:]

            #print(layer_act.shape)
            #os._exit()
            savemat(os.path.join(RDM_SAVE_ACTIVATIONS, 'AlexNet_{0}_layer_{1}_training_random_seed_{2}_faces.mat'.format(DNN_training_image_set, str(layer + 1).zfill(2), str(TRAINING_SEED).zfill(2)  )), activation_dict_faces)
            savemat(os.path.join(RDM_SAVE_ACTIVATIONS, 'AlexNet_{0}_layer_{1}_training_random_seed_{2}_places.mat'.format(DNN_training_image_set, str(layer + 1).zfill(2), str(TRAINING_SEED).zfill(2)  )), activation_dict_places)

            # compute the RDMs for the layer
            layer_rdms = [pdist(layer_act_t, distance_measure) for layer_act_t in layer_act]
            rdms.append(np.array(layer_rdms))

            bar.update(layer + 1)

        # get the RDMs for the readout
        readout_act = [sess.run(readout, feed_dict={img_ph: img}) for img in images]
        readout_act = np.stack(readout_act, axis=1)

        # compute the RDMs for the readout
        readout_rdms = [pdist(readout_t, distance_measure) for readout_t in readout_act]
        rdms.append(np.array(readout_rdms))

        bar.finish()

    return np.array(rdms)


def main():
    
    # get the images
    images = get_images(IMAGE_DIR, IMAGE_SIZE)
    
    # extract the RDMs
    rdms = extract_rdms(
        images, CKPT_PATH, MODEL_CLASS, N_TIMESTEPS, OUTPUT_SIZE, DROPOUT_TYPE, KEEP_PROB,
        SAMPLES_PER_IMAGE, RANDOM_SEED, GPU_ID, DISTANCE_MEASURE)
    
    # save the RDMs
    print('#####')
    print(RDM_SAVE_PATH)
    np.save(os.path.join(RDM_SAVE_PATH, 'RDMs_{}'.format(os.path.basename(IMAGE_DIR))), rdms)
    print('RDM saved!')

if __name__ == '__main__':
    main()
