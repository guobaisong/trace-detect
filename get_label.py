import path
import json
import numpy as np
import cv2
depth = '/home/guobaisong/PycharmProjects/trace-detect/DATA/Group_12_a/009{0}/depth/{1}_depth_{2}.npy'
bbox = '/home/guobaisong/PycharmProjects/trace-detect/DATA/Group_12_a/009{0}/RGB/{1}_RGB_{2}/bbox.json'
vis_demo = '/home/guobaisong/PycharmProjects/trace-detect/DATA/Group_12_a/009{0}/RGB/{1}_RGB_{2}/vis_demo.json'
img = '/home/guobaisong/PycharmProjects/trace-detect/DATA/Group_12_a/009{0}/RGB/{1}_RGB_{2}.jpg'
example = '/home/guobaisong/PycharmProjects/trace-detect/DATA/Group_12_a/009{0}/RGB/{1}_example_{2}.npy'
Tar_h = 128
Tar_w = 128
def padding(IMG, tar_w, tar_h):
    pre_w = IMG.shape[1]
    pre_h = IMG.shape[0]
    IMG_tar = cv2.copyMakeBorder(IMG, 0, tar_h-pre_h, 0, tar_w-pre_w, cv2.BORDER_CONSTANT, value=(0., 0., 0.))
    return IMG_tar
if __name__ == '__main__':
    WIDTH = 640
    HEIGHT = 480
    #data = np.zeros([8, 8, 36, WIDTH, HEIGHT, 3])
    label = np.zeros([8, 8, 36])
    for k in range(2, 8):
        for i in range(3, 8):
            for j in range(36):
                with open(bbox.format(k, i*10, j), 'r') as f:
                    data = json.load(f)
                    shape = data['shapes']
                    b_box = [0, 0, 0, 0]
                    for x in shape:
                        if x['cls_name'] == 'cube':
                            xmin = x['points']['x_min']
                            xmax = x['points']['x_max']
                            ymin = x['points']['y_min']
                            ymax = x['points']['y_max']
                            b_box = [xmin, xmax, ymin, ymax]
                    image = np.asarray(cv2.imread(img.format(k, i*10, j)))
                    image_crop = padding(image[b_box[2]:b_box[3], b_box[0]:b_box[1]], Tar_h, Tar_w)
#                    cv2.imshow('img', image_crop)
                    with open(vis_demo.format(k, i*10, j), 'r') as r:
                        rate = json.load(r)
                        rate = rate['cube']
 #                       print(rate)
                    r.close()
                f.close()
                action = 0
                MAX = rate
                if i != 7:
                    with open(vis_demo.format(k, (i+1)*10, j), 'r') as u:
                        rate = json.load(u)
                        u_r = rate['cube']
                        #print(u_r)
                        if u_r > MAX:
                            MAX = u_r
                            action = 1
                    u.close()
                if i != 3:
                    with open(vis_demo.format(k, (i-1) * 10, j), 'r') as d:
                        rate = json.load(d)
                        d_r = rate['cube']
                        #print(d_r)
                        if d_r > MAX:
                            MAX = d_r
                            action = 2
                    d.close()
                with open(vis_demo.format(k, i * 10, (j + 35) %36), 'r') as l:
                    rate = json.load(l)
                    l_r = rate['cube']
                    #print(l_r)
                    if l_r > MAX:
                        MAX = l_r
                        action = 3
                l.close()
                with open(vis_demo.format(k, i * 10, (j + 1) % 36), 'r') as r:
                    rate = json.load(r)
                    r_r = rate['cube']
                    #print(r_r)
                    if r_r > MAX:
                        MAX = r_r
                        action = 4
                r.close()
                DEPTH = np.load(depth.format(k, i*10, j))
                c = np.concatenate((np.asarray([action]+b_box), np.reshape(image_crop, [-1]), np.reshape(image, [-1]), np.reshape(DEPTH, [-1])), axis=0)
                np.save(example.format(k, i * 10, j), c)
                # print(action, MAX)
                # cv2.waitKey()
                #
