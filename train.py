from data import DataSet
from paramter import *
from model import model
from get_label import example
import numpy as np

num_list = [4, 5]
theta_list = [i * 10 for i in range(3, 8)]
alpha_list = [i for i in range(36)]
INDEX = [(k, i, j) for k in num_list for i in theta_list for j in alpha_list]

def load_data():
    L = []
    B = []
    I_C = []
    I = []
    D = []
    for x in INDEX:
        k, i, j = x
        data = np.load(example.format(k, i, j))
        action = data[0]
        bbox = data[1:5]
        img_crop = np.reshape(data[5:(5+128*128*3)], [128, 128, 3]).astype(float)
        img = np.reshape(data[(5+128*128*3):(5+128*128*3+480*640*3)], [480, 640, 3]).astype(float)
        depth = np.reshape(data[(5+128*128*3+480*640*3):], [480, 640, 1]).astype(float)
        L.append(action)
        B.append(bbox)
        I_C.append(img_crop)
        I.append(img)
        D.append(depth)
    return np.asarray(L), np.asarray(B), np.asarray(I_C), np.asarray(I), np.asarray(D)
def load_depth():
    pass
    #TODO

def computer_label():
    pass
    #TODO

if __name__ == '__main__':
    # img = load_img()
    # depth = load_depth()
    # label = computer_label()
    np.random.shuffle(INDEX)
    label, _, _, img, depth = load_data()

    SET = DataSet(img, depth, label, batch_size)

    MODEL = model(SET)

    MODEL.train()

