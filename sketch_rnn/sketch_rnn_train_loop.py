# Copyright 2017 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""SketchRNN training."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json
import os
import time
import urllib
import zipfile
from scipy.special import expit

# internal imports

import numpy as np
import requests
import six
from six.moves import cStringIO as StringIO
import tensorflow as tf
import uuid

from magenta.models.sketch_rnn import model as sketch_rnn_model
from magenta.models.sketch_rnn import utils

#from magenta.models.sketch_rnn.sketch_rnn_train import *
#from magenta.models.sketch_rnn.model import *
#from magenta.models.sketch_rnn.utils import *
from magenta.models.sketch_rnn.rnn import *



tf.logging.set_verbosity(tf.logging.INFO)

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string(
    'data_dir',
    'https://github.com/hardmaru/sketch-rnn-datasets/raw/master/aaron_sheep',
    'The directory in which to find the dataset specified in model hparams. '
    'If data_dir starts with "http://" or "https://", the file will be fetched '
    'remotely.')
tf.app.flags.DEFINE_string(
    'log_root', '/tmp/sketch_rnn/models/default',
    'Directory to store model checkpoints, tensorboard.')
tf.app.flags.DEFINE_boolean(
    'resume_training', False,
    'Set to true to load previous checkpoint')
tf.app.flags.DEFINE_string(
    'hparams', '',
    'Pass in comma-separated key=value pairs such as '
    '\'save_every=40,decay_rate=0.99\' '
    '(no whitespace) to be read into the HParams object defined in model.py')

PRETRAINED_MODELS_URL = ('http://download.magenta.tensorflow.org/models/'
                         'sketch_rnn.zip')


def reset_graph():
  """Closes the current default session and resets the graph."""
  sess = tf.get_default_session()
  if sess:
    sess.close()
  tf.reset_default_graph()


def load_env(data_dir, model_dir):
  """Loads environment for inference mode, used in jupyter notebook."""
  model_params = sketch_rnn_model.get_default_hparams()
  with tf.gfile.Open(os.path.join(model_dir, 'model_config.json'), 'r') as f:
    model_params.parse_json(f.read())
  return load_dataset(data_dir, model_params, inference_mode=True)


def load_model(model_dir):
  """Loads model for inference mode, used in jupyter notebook."""
  model_params = sketch_rnn_model.get_default_hparams()
  with tf.gfile.Open(os.path.join(model_dir, 'model_config.json'), 'r') as f:
    model_params.parse_json(f.read())

  model_params.batch_size = 1  # only sample one at a time
  eval_model_params = sketch_rnn_model.copy_hparams(model_params)
  eval_model_params.use_input_dropout = 0
  eval_model_params.use_recurrent_dropout = 0
  eval_model_params.use_output_dropout = 0
  eval_model_params.is_training = 0
  sample_model_params = sketch_rnn_model.copy_hparams(eval_model_params)
  sample_model_params.max_seq_len = 1  # sample one point at a time
  return [model_params, eval_model_params, sample_model_params]


def download_pretrained_models(
    models_root_dir='/tmp/sketch_rnn/models',
    pretrained_models_url=PRETRAINED_MODELS_URL):
  """Download pretrained models to a temporary directory."""
  tf.gfile.MakeDirs(models_root_dir)
  zip_path = os.path.join(
      models_root_dir, os.path.basename(pretrained_models_url))
  if os.path.isfile(zip_path):
    tf.logging.info('%s already exists, using cached copy', zip_path)
  else:
    tf.logging.info('Downloading pretrained models from %s...',
                    pretrained_models_url)
    urllib.urlretrieve(pretrained_models_url, zip_path)
    tf.logging.info('Download complete.')
  tf.logging.info('Unzipping %s...', zip_path)
  with zipfile.ZipFile(zip_path) as models_zip:
    models_zip.extractall(models_root_dir)
  tf.logging.info('Unzipping complete.')


def load_dataset(data_dir, model_params, inference_mode=False):
  """Loads the .npz file, and splits the set into train/valid/test."""

  # normalizes the x and y columns usint the training set.
  # applies same scaling factor to valid and test set.

  datasets = []
  if isinstance(model_params.data_set, list):
    datasets = model_params.data_set
  else:
    datasets = [model_params.data_set]

  train_strokes = None
  valid_strokes = None
  test_strokes = None

  for dataset in datasets:
    data_filepath = os.path.join(data_dir, dataset)
    if data_dir.startswith('http://') or data_dir.startswith('https://'):
      tf.logging.info('Downloading %s', data_filepath)
      response = requests.get(data_filepath)
      data = np.load(StringIO(response.content))
    else:
      if six.PY3:
        data = np.load(data_filepath, encoding='latin1')
      else:
        data = np.load(data_filepath)
    tf.logging.info('Loaded {}/{}/{} from {}'.format(
        len(data['train']), len(data['valid']), len(data['test']),
        dataset))
    if train_strokes is None:
      train_strokes = data['train']
      valid_strokes = data['valid']
      test_strokes = data['test']
    else:
      train_strokes = np.concatenate((train_strokes, data['train']))
      valid_strokes = np.concatenate((valid_strokes, data['valid']))
      test_strokes = np.concatenate((test_strokes, data['test']))

  all_strokes = np.concatenate((train_strokes, valid_strokes, test_strokes))
  num_points = 0
  for stroke in all_strokes:
    num_points += len(stroke)
  avg_len = num_points / len(all_strokes)
  tf.logging.info('Dataset combined: {} ({}/{}/{}), avg len {}'.format(
      len(all_strokes), len(train_strokes), len(valid_strokes),
      len(test_strokes), int(avg_len)))

  # calculate the max strokes we need.
  max_seq_len = utils.get_max_len(all_strokes)
  # overwrite the hps with this calculation.
  model_params.max_seq_len = max_seq_len

  tf.logging.info('model_params.max_seq_len %i.', model_params.max_seq_len)

  eval_model_params = sketch_rnn_model.copy_hparams(model_params)

  eval_model_params.use_input_dropout = 0
  eval_model_params.use_recurrent_dropout = 0
  eval_model_params.use_output_dropout = 0
  eval_model_params.is_training = 1

  if inference_mode:
    eval_model_params.batch_size = 1
    eval_model_params.is_training = 0

  sample_model_params = sketch_rnn_model.copy_hparams(eval_model_params)
  sample_model_params.batch_size = 1  # only sample one at a time
  sample_model_params.max_seq_len = 1  # sample one point at a time

  train_set = utils.DataLoader(
      train_strokes,
      model_params.batch_size,
      max_seq_length=model_params.max_seq_len,
      random_scale_factor=model_params.random_scale_factor,
      augment_stroke_prob=model_params.augment_stroke_prob)

  normalizing_scale_factor = train_set.calculate_normalizing_scale_factor()
  train_set.normalize(normalizing_scale_factor)

  valid_set = utils.DataLoader(
      valid_strokes,
      eval_model_params.batch_size,
      max_seq_length=eval_model_params.max_seq_len,
      random_scale_factor=0.0,
      augment_stroke_prob=0.0)
  valid_set.normalize(normalizing_scale_factor)

  test_set = utils.DataLoader(
      test_strokes,
      eval_model_params.batch_size,
      max_seq_length=eval_model_params.max_seq_len,
      random_scale_factor=0.0,
      augment_stroke_prob=0.0)
  test_set.normalize(normalizing_scale_factor)

  tf.logging.info('normalizing_scale_factor %4.4f.', normalizing_scale_factor)

  result = [
      train_set, valid_set, test_set, model_params, eval_model_params,
      sample_model_params
  ]
  return result


def evaluate_model(sess, model, data_set):
  """Returns the average weighted cost, reconstruction cost and KL cost."""
  total_cost = 0.0
  total_r_cost = 0.0
  total_kl_cost = 0.0
  for batch in range(data_set.num_batches):
    unused_orig_x, x, s = data_set.get_batch(batch)
    feed = {model.input_data: x, model.sequence_lengths: s}
    (cost, r_cost,
     kl_cost) = sess.run([model.cost, model.r_cost, model.kl_cost], feed)
    total_cost += cost
    total_r_cost += r_cost
    total_kl_cost += kl_cost

  total_cost /= (data_set.num_batches)
  total_r_cost /= (data_set.num_batches)
  total_kl_cost /= (data_set.num_batches)
  return (total_cost, total_r_cost, total_kl_cost)


def load_checkpoint(sess, checkpoint_path):
  saver = tf.train.Saver(tf.global_variables())
  ckpt = tf.train.get_checkpoint_state(checkpoint_path)
  tf.logging.info('Loading model %s.', ckpt.model_checkpoint_path)
  saver.restore(sess, ckpt.model_checkpoint_path)


def save_model(sess, model_save_path, global_step):
  saver = tf.train.Saver(tf.global_variables())
  checkpoint_path = os.path.join(model_save_path, 'vector')
  tf.logging.info('saving model %s.', checkpoint_path)
  tf.logging.info('global_step %i.', global_step)
  saver.save(sess, checkpoint_path, global_step=global_step)


def train(sess, model, eval_model, train_set, valid_set, test_set):
  """Train a sketch-rnn model."""
  # Setup summary writer.
  summary_writer = tf.summary.FileWriter(FLAGS.log_root)

  # Calculate trainable params.
  t_vars = tf.trainable_variables()
  count_t_vars = 0
  for var in t_vars:
    num_param = np.prod(var.get_shape().as_list())
    count_t_vars += num_param
    tf.logging.info('%s %s %i', var.name, str(var.get_shape()), num_param)
  tf.logging.info('Total trainable variables %i.', count_t_vars)
  model_summ = tf.summary.Summary()
  model_summ.value.add(
      tag='Num_Trainable_Params', simple_value=float(count_t_vars))
  summary_writer.add_summary(model_summ, 0)
  summary_writer.flush()

  # setup eval stats
  best_valid_cost = 100000000.0  # set a large init value
  valid_cost = 0.0

  # main train loop

  hps = model.hps
  start = time.time()
#
#
#
#   Change Steps # here
#
#
#
#
#

  for _ in range(2000):

    step = sess.run(model.global_step)

    curr_learning_rate = ((hps.learning_rate - hps.min_learning_rate) *
                          (hps.decay_rate)**step + hps.min_learning_rate)
    curr_kl_weight = (hps.kl_weight - (hps.kl_weight - hps.kl_weight_start) *
                      (hps.kl_decay_rate)**step)

    _, x, s = train_set.random_batch()
    feed = {
        model.input_data: x,
        model.sequence_lengths: s,
        model.lr: curr_learning_rate,
        model.kl_weight: curr_kl_weight
    }

    (train_cost, r_cost, kl_cost, _, train_step, _) = sess.run([
        model.cost, model.r_cost, model.kl_cost, model.final_state,
        model.global_step, model.train_op
    ], feed)

    if step % 20 == 0 and step > 0:

      end = time.time()
      time_taken = end - start

      cost_summ = tf.summary.Summary()
      cost_summ.value.add(tag='Train_Cost', simple_value=float(train_cost))
      reconstr_summ = tf.summary.Summary()
      reconstr_summ.value.add(
          tag='Train_Reconstr_Cost', simple_value=float(r_cost))
      kl_summ = tf.summary.Summary()
      kl_summ.value.add(tag='Train_KL_Cost', simple_value=float(kl_cost))
      lr_summ = tf.summary.Summary()
      lr_summ.value.add(
          tag='Learning_Rate', simple_value=float(curr_learning_rate))
      kl_weight_summ = tf.summary.Summary()
      kl_weight_summ.value.add(
          tag='KL_Weight', simple_value=float(curr_kl_weight))
      time_summ = tf.summary.Summary()
      time_summ.value.add(
          tag='Time_Taken_Train', simple_value=float(time_taken))

      output_format = ('step: %d, lr: %.6f, klw: %0.4f, cost: %.4f, '
                       'recon: %.4f, kl: %.4f, train_time_taken: %.4f')
      output_values = (step, curr_learning_rate, curr_kl_weight, train_cost,
                       r_cost, kl_cost, time_taken)
      output_log = output_format % output_values

      tf.logging.info(output_log)

      summary_writer.add_summary(cost_summ, train_step)
      summary_writer.add_summary(reconstr_summ, train_step)
      summary_writer.add_summary(kl_summ, train_step)
      summary_writer.add_summary(lr_summ, train_step)
      summary_writer.add_summary(kl_weight_summ, train_step)
      summary_writer.add_summary(time_summ, train_step)
      summary_writer.flush()
      start = time.time()

    if step % hps.save_every == 0 and step > 0:

      (valid_cost, valid_r_cost, valid_kl_cost) = evaluate_model(
          sess, eval_model, valid_set)

      end = time.time()
      time_taken_valid = end - start
      start = time.time()

      valid_cost_summ = tf.summary.Summary()
      valid_cost_summ.value.add(
          tag='Valid_Cost', simple_value=float(valid_cost))
      valid_reconstr_summ = tf.summary.Summary()
      valid_reconstr_summ.value.add(
          tag='Valid_Reconstr_Cost', simple_value=float(valid_r_cost))
      valid_kl_summ = tf.summary.Summary()
      valid_kl_summ.value.add(
          tag='Valid_KL_Cost', simple_value=float(valid_kl_cost))
      valid_time_summ = tf.summary.Summary()
      valid_time_summ.value.add(
          tag='Time_Taken_Valid', simple_value=float(time_taken_valid))

      output_format = ('best_valid_cost: %0.4f, valid_cost: %.4f, valid_recon: '
                       '%.4f, valid_kl: %.4f, valid_time_taken: %.4f')
      output_values = (min(best_valid_cost, valid_cost), valid_cost,
                       valid_r_cost, valid_kl_cost, time_taken_valid)
      output_log = output_format % output_values

      tf.logging.info(output_log)

      summary_writer.add_summary(valid_cost_summ, train_step)
      summary_writer.add_summary(valid_reconstr_summ, train_step)
      summary_writer.add_summary(valid_kl_summ, train_step)
      summary_writer.add_summary(valid_time_summ, train_step)
      summary_writer.flush()

      if valid_cost < best_valid_cost:
        best_valid_cost = valid_cost

        #save_model(sess, FLAGS.log_root, step)

        end = time.time()
        time_taken_save = end - start
        start = time.time()

        tf.logging.info('time_taken_save %4.4f.', time_taken_save)

        best_valid_cost_summ = tf.summary.Summary()
        best_valid_cost_summ.value.add(
            tag='Best_Valid_Cost', simple_value=float(best_valid_cost))

        summary_writer.add_summary(best_valid_cost_summ, train_step)
        summary_writer.flush()

        (eval_cost, eval_r_cost, eval_kl_cost) = evaluate_model(
            sess, eval_model, test_set)

        end = time.time()
        time_taken_eval = end - start
        start = time.time()

        eval_cost_summ = tf.summary.Summary()
        eval_cost_summ.value.add(tag='Eval_Cost', simple_value=float(eval_cost))
        eval_reconstr_summ = tf.summary.Summary()
        eval_reconstr_summ.value.add(
            tag='Eval_Reconstr_Cost', simple_value=float(eval_r_cost))
        eval_kl_summ = tf.summary.Summary()
        eval_kl_summ.value.add(
            tag='Eval_KL_Cost', simple_value=float(eval_kl_cost))
        eval_time_summ = tf.summary.Summary()
        eval_time_summ.value.add(
            tag='Time_Taken_Eval', simple_value=float(time_taken_eval))

        output_format = ('eval_cost: %.4f, eval_recon: %.4f, '
                         'eval_kl: %.4f, eval_time_taken: %.4f')
        output_values = (eval_cost, eval_r_cost, eval_kl_cost, time_taken_eval)
        output_log = output_format % output_values

        tf.logging.info(output_log)

        summary_writer.add_summary(eval_cost_summ, train_step)
        summary_writer.add_summary(eval_reconstr_summ, train_step)
        summary_writer.add_summary(eval_kl_summ, train_step)
        summary_writer.add_summary(eval_time_summ, train_step)
        summary_writer.flush()


class SketchLSTMCell(object):

    def __init__(self, num_units, input_size, Wxh, Whh, bias):
        self.num_units = num_units;
        self.input_size = input_size;
        self.Wxh = Wxh;
        self.Whh = Whh;
        self.bias = bias;
        self.forget_bias = 1.0;
        self.Wfull=np.transpose(np.concatenate((np.transpose(Wxh), np.transpose(Whh)),axis=1))

    def get_pdf(self,s, dec_output_w_, dec_output_b_):
        h = s[0];
        NOUT = N_mixture;
        z = np.dot(h, dec_output_w_) + dec_output_b_;
        z_pen_logits = z[0:3];
        z_pi = z[3+NOUT*0:3+NOUT*1];
        z_mu1 = z[3+NOUT*1:3+NOUT*2];
        z_mu2 = z[3+NOUT*2:3+NOUT*3];
        z_sigma1 = np.exp(z[3+NOUT*3:3+NOUT*4]);
        z_sigma2 = np.exp(z[3+NOUT*4:3+NOUT*5]);
        z_corr = np.tanh(z[3+NOUT*5:3+NOUT*6]);
        z_pen = softmax(z_pen_logits)
        z_pi = softmax(z_pi);
        return [z_pi, z_mu1, z_mu2, z_sigma1, z_sigma2, z_corr, z_pen];

    def zero_state(self):
        return [np.zeros(self.num_units), np.zeros(self.num_units)]

    def __call__(self,x,h,c):
        concat = np.concatenate((x, h));
        d = np.dot(concat, self.Wfull)
        hidden = d + self.bias
        num_units = self.num_units;
        forget_bias = self.forget_bias;
        i = expit(hidden[0*num_units:1*num_units]);
        g = np.tanh(hidden[1*num_units:2*num_units]);
        f = expit(hidden[2*num_units:3*num_units] + forget_bias);
        o = expit(hidden[3*num_units:4*num_units]);

        new_c = c*f + g*i
        new_h = np.tanh(new_c) * o;
        return [new_h, new_c];


N_mixture = 20
def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)

def sample(z, temperature=None, softmax_temperature=None):
    temp=0.25;
    if temperature is not None:
        temp = temperature;
    softmax_temp = 0.5+temp*0.5;
    if softmax_temperature is not None:
        softmax_temp = softmax_temperature;
    z_0 = adjust_temp(z[0], softmax_temp);
    z_6 = adjust_temp(z[6], softmax_temp);
    idx = sample_softmax(z_0);
    mu1 = z[1][idx];
    mu2 = z[2][idx];
    sigma1 = z[3][idx]*np.sqrt(temp);
    sigma2 = z[4][idx]*np.sqrt(temp);
    corr = z[5][idx];
    pen_idx = sample_softmax(z_6);
    penstate = [0, 0, 0];
    penstate[pen_idx] = 1;
    delta = birandn(mu1, mu2, sigma1, sigma2, corr);
    return [delta[0]*scale_factor, delta[1]*scale_factor, penstate[0], penstate[1], penstate[2]];

def adjust_temp(z_old, temp):
    z = z_old.copy();
    z = np.log(z)/temp
    x = z.max()
    z = z - x
    z = np.exp(z)
    x = z.sum()
    z = z/x
    return z;

def randf(a, b):
    return np.random.random()*(b-a)+a

def sample_softmax(z_sample):
    x = randf(0,1)
    N = z_sample.shape[0]
    accumulate = 0
    for i in range(N):
        accumulate += z_sample[i];
        if accumulate >= x:
            return i
    return -1;
return_v = False;
v_val = 0.0;

def gaussRandom():
    global return_v, v_val
    if return_v:
        return_v = False;
        return v_val;

    u = 2*np.random.random()-1;
    v = 2*np.random.random()-1;
    r = u*u + v*v;
    if r == 0 or  r > 1:
        return gaussRandom();
    c = np.sqrt(-2*np.log(r)/r);
    v_val = v*c;
    return_v = True;
    return u*c;

def randn(mu, std):
    return mu+gaussRandom()*std

def birandn(mu1, mu2, std1, std2, rho):
    z1 = randn(0,1);
    z2 = randn(0,1);
    x = np.sqrt(1-rho*rho)*std1*z1 + rho*std1*z2 + mu1;
    y = std2*z2 + mu2;
    return [x, y];

def generate(dec_lstm, dec_output_w ,dec_output_b, temperature = None, softmax_temperature = None):
    temp=0.25;
    if temperature is not None:
        temp = temperature;
    softmax_temp = 0.5+temp*0.5;
    if softmax_temperature is not None:
        softmax_temp = softmax_temperature
    init_state = dec_lstm.zero_state()
    h = init_state[0];
    c = init_state[1];

    x = np.array([0, 0, 0, 0, 0],dtype=np.float)
    result = [];
    max_seq_len = 125
    for i in range(max_seq_len):
        lstm_input = x;
        rnn_state = dec_lstm(lstm_input, h, c);
        pdf = dec_lstm.get_pdf(rnn_state, dec_output_w, dec_output_b)
        [dx, dy, pen_down, pen_up, pen_end] = sample(pdf, temp, softmax_temp);
        result.append([dx, dy, pen_down, pen_up, pen_end]);
        if pen_end == 1:
            return result;
        x = np.array([dx/scale_factor, dy/scale_factor, pen_down, pen_up, pen_end]);

        h = rnn_state[0];
        c = rnn_state[1];
    result.append([0, 0, 0, 0, 1]);
    return result
max_seq_len = 123
scale_factor = 99.698

def get_sketch(sketch):
    #sketch = generate()
    #print(sketch)
    xsum, ysum = [0],[0]
    for i,l in enumerate(sketch):
        xsum.append(l[0]+xsum[i])
        ysum.append(l[1]+ysum[i])
    width = np.max(xsum) - np.min(xsum)
    height = np.max(ysum) - np.min(ysum)
    drawing = np.zeros((int(height)+40,int(width)+40,3),dtype=np.uint8)
    drawing[:,:] = (255,255,255)
    x_start, y_start = 20-int(np.min(xsum)),20-int(np.min(ysum))

    x, y = x_start,y_start
    prev_pen = [1, 0, 0]

    x_lines = []
    y_lines = []
    for l in sketch:
        dx, dy, pen_down, pen_up, pen_end = l
        dx = int(dx)
        dy = int(dy)
        if prev_pen[2] == 1:
            break
        if prev_pen[0] == 1:
            x_lines.extend([x,x+dx])
            y_lines.extend([y,y+dy])
            #lines.append([[x,y],[x+dx,y+dy]])
        x += dx;
        y += dy;
        prev_pen = [pen_down, pen_up, pen_end]
    return [x_lines,y_lines]

def trainer(model_params, datasets):
  """Train a sketch-rnn model."""

  train_set = datasets[0]
  valid_set = datasets[1]
  test_set = datasets[2]
  model_params = datasets[3]
  eval_model_params = datasets[4]

  reset_graph()
  model = sketch_rnn_model.Model(model_params)
  eval_model = sketch_rnn_model.Model(eval_model_params, reuse=True)
  sample_params = sketch_rnn_model.copy_hparams(eval_model_params)
  sample_params.max_seq_len = 1
  sample_model = sketch_rnn_model.Model(sample_params, reuse=True)

  sess = tf.InteractiveSession()
  sess.run(tf.global_variables_initializer())

  train(sess, model, eval_model, train_set, valid_set, test_set)

  output_w_ = [v for v in tf.trainable_variables() if v.name == "vector_rnn/RNN/output_w:0"][0].eval()
  output_b_ = [v for v in tf.trainable_variables() if v.name == "vector_rnn/RNN/output_b:0"][0].eval()
  lstm_W_xh_ = [v for v in tf.trainable_variables() if v.name == "vector_rnn/RNN/LSTMCell/W_xh:0"][0].eval()
  lstm_W_hh_ = [v for v in tf.trainable_variables() if v.name == "vector_rnn/RNN/LSTMCell/W_hh:0"][0].eval()
  lstm_bias_ = [v for v in tf.trainable_variables() if v.name == "vector_rnn/RNN/LSTMCell/bias:0"][0].eval()

  dec_output_w = output_w_;
  dec_output_b = output_b_;
  dec_lstm_W_xh = lstm_W_xh_;
  dec_lstm_W_hh = lstm_W_hh_;
  dec_lstm_bias = lstm_bias_;
  dec_num_units = dec_lstm_W_hh.shape[0];
  dec_input_size = dec_lstm_W_xh.shape[0];
  dec_lstm = SketchLSTMCell(dec_num_units, dec_input_size, dec_lstm_W_xh, dec_lstm_W_hh, dec_lstm_bias)

  result = generate(dec_lstm, dec_output_w, dec_output_b)

  print(result)
  output = []
  for i in result:
      entry = []
      #print(i)
      if i[2] != 0:
          pen = 0
      else:
          if i[3] == 1:
              pen = i[3]
          else:
              pen = i[4]
      #print(pen)
      #print(i[2])
      entry.extend(i[:2])
      entry.append(pen)
      output.append(entry)
  #print(output)




def main(unused_argv):
  """Load model params, save config file and start trainer."""
  model_params = sketch_rnn_model.get_default_hparams()

  if FLAGS.hparams:
      model_params.parse(FLAGS.hparams)


  np.set_printoptions(precision=8, edgeitems=6, linewidth=200, suppress=True)

  tf.logging.info('sketch-rnn')
  tf.logging.info('Hyperparams:')
  print(model_params.values())
  for key, val in six.iteritems(model_params.values()):
      tf.logging.info('%s = %s', key, str(val))
  tf.logging.info('Loading data files.')
  #print(model_params)
  datasets = load_dataset(FLAGS.data_dir, model_params)

  #parse train, valid and
  train = datasets[0].strokes
  valid = datasets[1].strokes
  test = datasets[2].strokes
  print("\n\ntrain length = %d, valid_length = %d, test length = %d\n\n" % (len(train), len(valid), len(test)))


  #replace data
  a = 2
  #for i in range(a):
      #datasets[0].strokes[i] = -100;
  #for j in range(4):
      #print(datasets[0].strokes[j])


  #retrain_times = 1
  #for i in range(retrain_times):

  trainer(model_params, datasets)


def console_entry_point():
  tf.app.run(main)


if __name__ == '__main__':
  console_entry_point()
