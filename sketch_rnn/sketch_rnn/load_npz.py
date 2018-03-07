import numpy as np
import os

data_dir = "/Users/mhy/Desktop/data lab/Haoyuan's dataset/Sketch-rnn/"
file_name = "speedboat.npz"
data_path = os.path.join(data_dir, file_name)

with np.load(data_path) as data:
    print(data['train'])
