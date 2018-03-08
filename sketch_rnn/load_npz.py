import numpy as np
import os

data_dir = "/Users/mhy/Desktop/data lab/sketch-github/sketch_rnn/data"
file_name = "speedboat.npz"
data_path = os.path.join(data_dir, file_name)

with np.load(data_path) as data:
    print(data['train'])
    #print(data['validate'])
