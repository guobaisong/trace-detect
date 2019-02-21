import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np
from tqdm import tqdm
from paramter import *

class model:
    def __init__(self, DataSet):
        self.DataSet = DataSet
        self.sess = tf.Session()
        self.get_input()
        self.get_model_module()
        self.saver = tf.train.Saver()
    def get_input(self):
        self.img = tf.placeholder(tf.float32, (None, HEIGHT, WIDTH, 3))
        self.depth = tf.placeholder(tf.float32, (None, HEIGHT, WIDTH, 1))
        self.label = tf.placeholder(tf.int32, (None)) ###up, down, left, right, fix
    def get_model_module(self):
        self.train_loss, self.train_step, self.predictor= self.get_model()
    def get_model(self):
        D1 = slim.conv2d(self.img, num_outputs=16, kernel_size=(4, 4), stride=(2, 2), padding = 'SAME', scope='d1')
        D2 = slim.conv2d(D1, num_outputs=32, kernel_size=(4, 4), stride=(2, 2), padding = 'SAME', scope='d2')
        D3 = slim.conv2d(D2, num_outputs=64, kernel_size=(4, 4), stride = (2, 2), padding='SAME', scope='d3')

        C1 = slim.conv2d(self.depth, num_outputs=16, kernel_size=(4, 4), stride=(2, 2), padding = 'SAME', scope='c1')
        C2 = slim.conv2d(C1, num_outputs=32, kernel_size=(4, 4), stride=(2, 2), padding = 'SAME', scope='c2')
        C3 = slim.conv2d(C2, num_outputs=64, kernel_size=(4, 4), stride=(2, 2), padding = 'SAME', scope='c3')

        T1 = tf.concat((D3, C3), axis=-1)
        T2 = slim.conv2d(T1, num_outputs = 128, kernel_size = (7, 7), stride = (2, 2), padding = 'VALID', scope='t2')
        T2_p = slim.max_pool2d(T2, kernel_size=(2, 2), stride=(2, 2), padding = 'VALID', scope='t2_p')
        T3 = slim.conv2d(T2_p, num_outputs = 256, kernel_size = (7, 7), stride = (2, 2), padding = 'VALID', scope='t3')
        #T3_p = slim.max_pool2d(T3, kernel_size=(2, 2), stride=(2, 2), scope='t3_p')
        # T4 = slim.conv2d(T3_p, num_outputs=512, kernel_size=(7, 7), stride = (2, 2), padding = 'VALID', scope='t4')

        F1 = tf.reduce_max(tf.reduce_max(T3, axis=-1), axis=-1)
        F2 = slim.fully_connected(F1, num_outputs=256, scope='f2')
        F3 = slim.fully_connected(F2, num_outputs=64, scope='f3')
        F4 = slim.fully_connected(F3, num_outputs=5, scope='f4')

        one_hot_label = tf.one_hot(self.label, 5)
        train_loss = tf.nn.softmax_cross_entropy_with_logits(logits=F4, labels=one_hot_label)
        train_step = tf.train.AdamOptimizer(learning_rate).minimize(train_loss)
        prediction = tf.argmax(F4, axis=-1)

        return train_loss, train_step, prediction

    def train_one_epoch(self):
        batch_img, batch_depth, batch_label = self.DataSet.get_train_buffer()
        self.sess.run(self.train_step, feed_dict={self.img:batch_img, self.depth:batch_depth, self.label:batch_label})


    def evaluate(self):
        test_img, test_depth, test_label = self.DataSet.get_test_buffer()
        pre_result = self.sess.run(self.predictor, feed_dict={self.img: test_img, self.depth: test_depth})
        print(pre_result.shape)
        print(test_label.shape)
        test_Acc = np.mean(np.max(pre_result, axis=-1)==test_label)
        return test_Acc

    def train(self):
        self.sess.run(tf.global_variables_initializer())
        for i in range(MAX_TRAIN_EPOCH):
            print(i)
            self.train_one_epoch()
            if i % (MAX_TRAIN_EPOCH/20) == 0:
                acc = self.evaluate()
                print('epoch '+str(i)+' | testAcc= '+str(acc))
                self.saver.save(self.sess, save_path=save_path)

