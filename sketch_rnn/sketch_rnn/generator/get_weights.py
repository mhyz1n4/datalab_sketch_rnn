# import the required libraries
import numpy as np
import time
import random
import cPickle
import codecs
import collections
import os
import math
import json
import tensorflow as tf
from six.moves import xrange

# libraries required for visualisation:
from IPython.display import SVG, display
import svgwrite # conda install -c omnia svgwrite=1.1.6
import PIL
from PIL import Image
import matplotlib.pyplot as plt

# set numpy output to something sensible
np.set_printoptions(precision=8, edgeitems=6, linewidth=200, suppress=True)

# import our command line tools
from magenta.models.sketch_rnn.sketch_rnn_train import *
from magenta.models.sketch_rnn.model import *
from magenta.models.sketch_rnn.utils import *
from magenta.models.sketch_rnn.rnn import *


#model_dir = '/Users/mhy/Desktop/data lab/magenta/magenta/models/sketch_rnn/attachments/speedboat'
model_dir = '/Users/mhy/Desktop/data lab/magenta/magenta/models/sketch_rnn/test_sketcher/weights-uncon-7500'
#model_dir = '/Users/mhy/Downloads/sketch_rnn/aaron_sheep/lstm'

[hps_model, eval_hps_model, sample_hps_model] = load_model(model_dir)

# construct the sketch-rnn model here:
reset_graph()
model = Model(hps_model)
eval_model = Model(eval_hps_model, reuse=True)
sample_model = Model(sample_hps_model, reuse=True)

sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())

# loads the weights from checkpoint into our model
load_checkpoint(sess, model_dir)
#v = tf.trainable_variables()
#x = sess.run(v)
output_w_ = [v for v in tf.trainable_variables() if v.name == "vector_rnn/RNN/output_w:0"][0].eval()
output_b_ = [v for v in tf.trainable_variables() if v.name == "vector_rnn/RNN/output_b:0"][0].eval()
lstm_W_xh_ = [v for v in tf.trainable_variables() if v.name == "vector_rnn/RNN/LSTMCell/W_xh:0"][0].eval()
lstm_W_hh_ = [v for v in tf.trainable_variables() if v.name == "vector_rnn/RNN/LSTMCell/W_hh:0"][0].eval()
lstm_bias_ = [v for v in tf.trainable_variables() if v.name == "vector_rnn/RNN/LSTMCell/bias:0"][0].eval()

output_w = output_w_.tolist()
output_b = output_b_.tolist()
lstm_W_xh = lstm_W_xh_.tolist()
lstm_W_hh = lstm_W_hh_.tolist()
lstm_bias = lstm_bias_.tolist()

print("output w:    ")
print(output_w_.shape)
print("output b:    ")
print(output_b_.shape)
print("lstm_W_xh:    ")
print(lstm_W_xh_.shape)
print("lstm_W_hh:    ")
print(lstm_W_hh_.shape)
print("bias:    ")
print(lstm_bias_.shape)

total_weight = [output_w, output_b, lstm_W_xh, lstm_W_hh, lstm_bias]
with open('weight_test.json', 'w') as outfile:
	json.dump(total_weight, outfile)

print(len(total_weight))
