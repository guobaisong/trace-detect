import numpy as np

class DataSet:
    def __init__(self, imgs, depth, labels, batch_size):
        self.train_imgs = imgs[0:int(0.8*imgs.shape[0])]
        self.test_imgs = imgs[int(0.8*imgs.shape[0]):]

        self.train_depth = depth[0:int(0.8 * imgs.shape[0])]
        self.test_depth = depth[int(0.8 * imgs.shape[0]):]

        self.train_cnt = 0
        self.batch_size = batch_size
        self.max_batch = self.train_imgs.shape[0] / self.batch_size
        # if labels.all() == None:
        #     #self.train_label, self.test_label = self.get_lagel(imgs)
        #     pass
        # else:
        self.train_labels = labels[0:int(0.8*imgs.shape[0])]
        self.test_labels = labels[int(0.8*imgs.shape[0]):]
    def get_lagel(self, imgs):
        #TODO
        pass

    def get_train_buffer(self):
        if self.train_cnt == self.max_batch:
            left = self.train_cnt * self.batch_size
            right = (self.train_cnt+1) * self.batch_size - self.train_imgs.shape[0]
            self.train_cnt = 0
            return np.concatenate((self.train_imgs[:right], self.train_imgs[left:]), axis = 0), \
                   np.concatenate((self.train_depth[:right], self.train_depth[left:]), axis=0), \
                   np.concatenate((self.train_labels[:right], self.train_labels[left:]), axis=0)
        else:
            left = self.train_cnt * self.batch_size
            right = (self.train_cnt+1) * self.batch_size
            self.train_cnt+=1
            return self.train_imgs[left:right],self.train_depth[left:right],self.train_labels[left:right]

    def get_test_buffer(self):
        return self.test_imgs, self.test_depth, self.test_labels
