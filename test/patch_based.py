import tensorflow as tf
import numpy as np
import argparse
import _init_paths

from dataset import get_train_pair
from network import PatchActivationNet
from utils import Avg, get_patch_num

def parse_args():
    parser = argparse.ArgumentParser(description='songofthewind')
    parser.add_argument('--largek', dest='large_k', default=85)
    parser.add_argument('--middik', dest='middle_k', default=61)
    parser.add_argument('--smallk', dest='small_k', default=33)
    parser.add_argument('--stride', dest='stride', default=1000)
    parser.add_argument('--ysize', dest='y_psize', default=5)
    parser.add_argument('--modelname', dest='model_name', default='pan')
    parser.add_argument('--max_iter', dest='max_iter', default=10000)
    parser.add_argument('--bsize', dest='batch_size', default=1)
    args = parser.parse_args()
    return args

if __name__=='__main__':
    args = parse_args()
    print ('-'*50)
    print ('args::')
    for arg in vars(args):
        print ('%15s : %s'%(arg, getattr(args, arg)))
    print ('-'*50)

    save_dir = '/home/mike/models/' + args.model_name + '/a.ckpt'
    kernel1 = int(args.small_k / 2)
    kernel2 = int(args.middle_k / 2)
    kernel_cen = int(args.large_k / 2)
    kernely = int(args.y_psize / 2)

    x_ph = tf.placeholder(tf.float32, [None,None,None])
    y_ph = tf.placeholder(tf.float32, [None,None,None])
    b_ph = tf.placeholder(tf.bool, [None])

    x_resh = tf.expand_dims(x_ph, axis=3)
    x_patch = tf.extract_image_patches(x_resh,
            ksizes=[1,args.large_k,args.large_k,1],
            strides=[1,args.stride,args.stride,1],
            rates=[1,1,1,1],
            padding='VALID')
    x_3 = tf.reshape(x_patch, [-1,args.large_k,args.large_k])
    a, b = (kernel_cen - kernel1, kernel_cen + kernel1+1)
    x_1 = x_3[:,a:b,a:b]
    a, b = (kernel_cen - kernel2, kernel_cen + kernel2+1)
    x_2 = x_3[:,a:b,a:b]

    y_resh = tf.expand_dims(y_ph, axis=3)
    y_patch = tf.extract_image_patches(y_resh,
            ksizes=[1,args.large_k,args.large_k,1],
            strides=[1,args.stride,args.stride,1],
            rates=[1,1,1,1],
            padding='VALID')
    y_p = tf.reshape(y_patch, [-1,args.large_k,args.large_k])
    a, b = (kernel_cen - kernely, kernel_cen + kernely+1)
    y_p = y_p[:,a:b,a:b]

    y_trues = tf.reduce_mean(y_p, axis=[1,2]) * 100

    pan = PatchActivationNet(args.model_name)
    out = pan.get_activation(x_1, x_2, x_3)

    mask = tf.greater(y_trues, 100)
    mask = tf.logical_or(mask, b_ph)
    mask = tf.cast(mask, tf.float32)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver()
    saver.restore(sess, save_dir)

    # redefine for convinience
    kc = kernel_cen
    ims = image_size = 512
    x, y = get_train_pair(1)
    out_activation_x = np.zeros([ims, ims])
    out_activation_y = np.zeros([ims, ims])
    for i in range(kc, ims-kc, 2):
        batch_x = np.zeros([ims-2*kc, 2*kc+1, 2*kc+1])
        batch_y = np.zeros([ims-2*kc, 2*kc+1, 2*kc+1])
        for j in range(kc, ims-kc):
            batch_x[j-kc] = x[:,i-kc:i+kc+1,j-kc:j+kc+1]
            batch_y[j-kc] = y[:,i-kc:i+kc+1,j-kc:j+kc+1]
        fd = {x_ph: batch_x, y_ph: batch_y}
        ox, oy = sess.run([out, y_trues], fd)
        out_activation_x[i, kc:ims-kc] = ox
        out_activation_y[i, kc:ims-kc] = oy
        out_activation_x[i+1, kc:ims-kc] = ox
        out_activation_y[i+1, kc:ims-kc] = oy
        if i % 10 == 0:
            print (i)
    np.save('garbages/test_x.npy', out_activation_x)
    np.save('garbages/test_y.npy', out_activation_y)
