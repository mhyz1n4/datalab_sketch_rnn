import numpy as np
import json
import os
from scipy.special import expit
import cv2

import time
#import BaseHTTPServer

from magenta.models.sketch_rnn.sketch_rnn_train import *
from magenta.models.sketch_rnn.model import *
from magenta.models.sketch_rnn.utils import *
from magenta.models.sketch_rnn.rnn import *


HOST_NAME = 'localhost' # !!!REMEMBER TO CHANGE THIS!!!
PORT_NUMBER = 9020 # Maybe set this to 9000.

import uuid

'''
class MyHandler(BaseHTTPServer.BaseHTTPRequestHandler):
    def do_HEAD(s):
        s.send_response(200)
        s.send_header("Content-type", "application/json")
        s.end_headers()
    def do_GET(s):
        """Respond to a GET request."""
        s.send_response(200)
        s.send_header("Content-type", "application/json")
        s.end_headers()
        lines = get_sketch()
        s.wfile.write(json.dumps(lines))
'''

#server_class = BaseHTTPServer.HTTPServer
#httpd = server_class((HOST_NAME, PORT_NUMBER), MyHandler)

class SketchLSTMCell(object):

    def __init__(self, num_units, input_size, Wxh, Whh, bias):
        self.num_units = num_units;
        self.input_size = input_size;
        self.Wxh = Wxh;
        self.Whh = Whh;
        self.bias = bias;
        self.forget_bias = 1.0;
        self.Wfull=np.transpose(np.concatenate((np.transpose(Wxh), np.transpose(Whh)),axis=1))

    def get_pdf(self,s):
        h = s[0];
        NOUT = N_mixture;
        z = np.dot(h, dec_output_w) + dec_output_b;
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

#
#
#   model file
#
#


model_dir = '/Users/mhy/Desktop/data lab/magenta/magenta/models/sketch_rnn/test_sketcher/weights-uncon-7500'

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

#output_w = output_w_.tolist()
#output_b = output_b_.tolist()
#lstm_W_xh = lstm_W_xh_.tolist()
#lstm_W_hh = lstm_W_hh_.tolist()
#lstm_bias = lstm_bias_.tolist()

#model_file = "weight_test.json"
#rawweights = json.load(open(model_file,'r'))
#weights = []
#for w in rawweights:
    #w = json.loads(w)
    #npw = np.array(w,dtype=np.float32)
    #weights.append(npw)
dec_output_w = output_w_;
dec_output_b = output_b_;
dec_lstm_W_xh = lstm_W_xh_;
dec_lstm_W_hh = lstm_W_hh_;
dec_lstm_bias = lstm_bias_;
dec_num_units = dec_lstm_W_hh.shape[0];
dec_input_size = dec_lstm_W_xh.shape[0];
dec_lstm = SketchLSTMCell(dec_num_units, dec_input_size, dec_lstm_W_xh, dec_lstm_W_hh, dec_lstm_bias)

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

def generate(temperature = None, softmax_temperature = None):
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
        pdf = dec_lstm.get_pdf(rnn_state)
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

def get_sketch():
    sketch = generate()
    print(sketch)
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



img = np.ones((1000,1000,3), np.uint8)
#img = cv2.imread('dave.jpg')
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

gen_sketch=[]
#for i in range(6):
    #temp = get_sketch()
    #gen_sketch.append(temp)
#print(gen_sketch)
temp = get_sketch()
gen_sketch.append(temp)

for k in gen_sketch:
    x_axis = k[0]
    y_axis = k[1]
    length = len(x_axis)
    #print(x_axis)
    #print(y_axis)

    for i in range(length):
        if (i == length - 1):
            break
        else:
            cv2.line(img, (x_axis[i], y_axis[i]), (x_axis[i+1], y_axis[i+1]), (255, 255, 255), 5)

cv2.imshow('image',img)
cv2.waitKey(0)
cv2.destroyAllWindows()


#sample = sample()
#print(sample)

'''
import requests

hostname = "0.0.0.0"
port = 5000
check = 0
for i in range(6):
    x_array,y_array = get_sketch()
    r = requests.post("http://{}:{}/data".format(hostname,port),
                    data = json.dumps({"data":{"x_data":x_array,"y_data":y_array,"id":i,"check":check}}))
'''
