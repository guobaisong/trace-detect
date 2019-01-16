from data import DataSet
from paramter import *
from model import model

def load_img():
    pass
    #TODO

def load_depth():
    pass
    #TODO

def computer_label():
    pass
    #TODO

if __name__ == '__main__':
    img = load_img()
    depth = load_depth()
    label = computer_label()

    SET = DataSet(img, depth, label, batch_size)

    MODEL = model(SET)

    MODEL.train()
