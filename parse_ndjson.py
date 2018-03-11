import jsonlines
import numpy as np
import io
import cv2
import os

img = np.ones((1000,1000,3), np.uint8)
#img = cv2.imread('dave.jpg')
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
total_matrxix = []
with jsonlines.open("Falarmclock.ndjson") as reader:
    for obj in reader:
        drawing_list = obj.get(u'drawing')
        drawing_matrix = np.array(drawing_list)
        total_matrxix.append(drawing_matrix)
#print(drawing_matrix.shape)

test = total_matrxix[0]
for stroke in test:
#stroke = test[0]
    x = stroke[0]
    y = stroke[1]
    length = len(x)
    for i in range(length):
        if (i == length - 1):
            break
        else:
            cv2.line(img, (x[i], y[i]), (x[i+1], y[i+1]), (255, 255, 255), 5)

path = 'Users/mhy/Desktop/test_images'
cv2.imwrite(os.path.join(path, 'test.jpg'), img)
cv2.imshow('image',img)
cv2.waitKey(0)
cv2.destroyAllWindows()


#for i in test[0]:
#print(len(i))
#for i in range(len(total_matrxix[0])):
#cv2.liens



