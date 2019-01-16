import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np
from tqdm import tqdm
from paramter import *

class model:
    def __init__(self, DataSet):
        self.DataSet = DataSet
        self.sess = tf.Session()
        self.saver = tf.train.Saver()
        self.get_input()
    def get_input(self):
        self.img = tf.placeholder(tf.float32, (None, HEIGHT, WIDTH, 3))
        self.depth = tf.placeholder(tf.float32, (None, HEIGHT, WIDTH, 1))
        self.label = tf.placeholder(tf.float32, (None, 4+1)) ###up, down, left, right, fix
    def get_model_module(self):
        self.train_loss, self.train_step, self.predictor= self.get_model()
    def get_model(self):
        D1 = slim.conv2d(self.depth, num_outputs=8, kernel_size=(4, 4), stride=(1, 1), padding = 'SAME')
        D2 = slim.conv2d(D1, num_outputs=16, kernel_size=(4, 4), stride=(1, 1), padding = 'SAME')

        C1 = slim.conv2d(self.depth, num_outputs=16, kernel_size=(4, 4), stride=(1, 1), padding = 'SAME')
        C2 = slim.conv2d(C1, num_outputs=32, kernel_size=(4, 4), stride=(1, 1), padding = 'SAME')

        T1 = tf.concat((D2, C2), axis=-1)
        T2 = slim.conv2d(T1, num_outputs = 128, kernel_size = (7, 7), stride = (3, 3), padding = 'VALID')
        T2_p = slim.max_pool2d(T2, stride=(2, 2), padding = 'VALID')
        T3 = slim.conv2d(T2_p, num_outputs = 256, kernel_size = (7, 7), stride = (3, 3), padding = 'VALID')
        T3_p = slim.max_pool2d(T3, stride=(2, 2))
        T4 = slim.conv2d(T3_p, num_outputs=512, kernel_size=(7, 7), stride = (3, 3), padding = 'VALID')

        F1 = tf.reduce_max(tf.reduce_max(T4, axis=-1), axis=-1)
        F2 = slim.fully_connected(F1, num_outputs=256)
        F3 = slim.fully_connected(F2, num_outputs=64)
        F4 = slim.fully_connected(F3, num_outputs=5)

        train_loss = tf.nn.softmax_cross_entropy_with_logits(F4, self.label)
        train_step = tf.train.AdamOptimizer(learning_rate).minimize(train_loss)
        prediction = tf.argmax(F4, axis=-1)

        return train_loss, train_step, prediction

    def train_one_epoch(self):
        batch_img, batch_depth, batch_label = self.DataSet.get_train_buffer()
        self.sess.run(self.train_step, feed_dict={self.img:batch_img, self.depth:batch_depth, self.label:batch_label})


    def evaluate(self):
        test_img, test_depth, test_label = self.DataSet.get_test_buffer()
        pre_result = self.sess.run(self.predictor, feed_dict={self.img: test_img, self.depth: test_depth})

        test_Acc = np.mean(np.max(pre_result, axis=-1)==np.max(test_label, axis=-1))
        return test_Acc

    def train(self):
        for i in tqdm(range(MAX_TRAIN_EPOCH)):
            self.train_one_epoch()
            if i % (MAX_TRAIN_EPOCH/20) == 0:
                acc = self.evaluate()
                print('epoch '+str(i)+' | testAcc= '+'acc')
                self.saver.save(self.sess, save_path=save_path)

