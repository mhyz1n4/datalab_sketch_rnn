import numpy as np
import os

data_dirpath = "/Users/mhy/Desktop/data lab/Haoyuan's dataset/Sketch-rnn/"
data_name = ["cruiseship.npz", "sailboat.npz", "speedboat.npz"]

for files in data_name:
    data_filepath = os.path.join (data_dirpath, files)
    data = np.load(data_filepath)
    train_strokes = data['train']
    valid_strokes = data['valid']
    test_strokes = data['test']
    print (files)
    print ("train length: %d, train length: %d, train length: %d"% (len(train_strokes), len(valid_strokes), len(test_strokes)))
    print(train_strokes[0])
    total = len(train_strokes) + len(valid_strokes) + len(test_strokes)
    all_strokes = np.concatenate((train_strokes, valid_strokes, test_strokes))

    test_stroke = all_strokes
    x = stroke[0]
    y = stroke[1]
    length = len(x)
    '''
    for i in range(length):
        if (i == length - 1):
            break
        else:
            cv2.line(img, (x[i], y[i]), (x[i+1], y[i+1]), (255, 255, 255), 5)
    break
    '''

    '''
    for j in all_strokes:
        total = len(result)
        hostname = "54.82.94.146"
        port = 80
        check = 0
        x_array,y_array = get_sketch(result)
        print(j)
        print(x_array)
        #rint(y_array)
        r = requests.post("http://{}:{}/Hdata".format(hostname,port),data = json.dumps({"data":{"x_data":x_array,"y_data":y_array,"id":j,"check":check}}))
    '''
