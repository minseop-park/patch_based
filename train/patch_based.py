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
    parser.add_argument('--stride', dest='stride', default=27)
    parser.add_argument('--ysize', dest='y_psize', default=5)
    parser.add_argument('--modelname', dest='model_name', default='pan')
    parser.add_argument('--max_iter', dest='max_iter', default=10000)
    parser.add_argument('--bsize', dest='batch_size', default=8)
    parser.add_argument('--imgsize', dest='imsize', default=512)
    parser.add_argument('--lr', dest='lr', default=1e-4)
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

    y_trues = tf.reduce_mean(y_p, axis=[1,2])


    pan = PatchActivationNet(args.model_name)
    out = pan.get_activation(x_1, x_2, x_3)

    mask = tf.greater(y_trues, 10)
    mask = tf.logical_or(mask, b_ph)
    mask = tf.cast(mask, tf.float32)

    loss = tf.reduce_mean((out - y_trues)**2 * mask + abs(out))
    train_step = tf.train.AdamOptimizer(args.lr).minimize(loss)


    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver()
#    saver.restore(sess, save_dir)

    avg = Avg(desc='loss')
    for i in range(1,1+args.max_iter):
        x, y = get_train_pair(args.batch_size)
        rnd_init = np.random.randint(100, size=2)
        pn = get_patch_num(args.imsize, args.large_k, args.stride,
                rnd_init[0], rnd_init[1],
                args.batch_size)
        rnd_bern = np.random.randint(200, size=pn)
        rnd_bern = (rnd_bern < 3)
        x = x[:,rnd_init[0]:,rnd_init[1]:]
        y = y[:,rnd_init[0]:,rnd_init[1]:]
        fd = {x_ph: x, y_ph: y, b_ph: rnd_bern}

        l, _ = sess.run([loss, train_step], fd)
        avg.add(l, 0)

        if i % 30 == 0:
            avg.show(i)

        if i % 500 == 0:
            avg.description()
            saver.save(sess, save_dir)
