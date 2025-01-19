import os
from pprint import pprint
import tensorflow as tf

tf_path = os.path.abspath(r".\AlexNet\training_seed_01\model.ckpt_epoch89")  # Path to our TensorFlow checkpoint
tf_vars = tf.train.list_variables(tf_path)
pprint(tf_vars)