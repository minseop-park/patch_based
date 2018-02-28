import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

import _init_paths
from network import ResAutoEncoderTrainNet
from network import ResAutoEncoderTestNet
from utils import Avg, apply_window
from dataset import get_train_pair

model_name = 'patch_based'
save_dir = '/home/mike/models/' + model_name + '/'
max_iter = 10000

img_x = tf.placeholder(tf.float32, [None,None,None])
img_y = tf.placeholder(tf.float32, [None,None,None])

rnd_init = tf.placeholder(tf.int32, [2])

x_reshape = tf.expand_dims(img_x, axis=3)

model = ResAutoEncoderTrainNet('modelpb')
#model = ResAutoEncoderTestNet('x_autoencoderk')
codes = model.encoder(x_reshape)
recon = model.decoder(codes)


mask = tf.greater(img_y, tf.reduce_mean(img_y))
mask = tf.cast(mask, tf.float32) + .1
loss = tf.reduce_mean(    (img_y - recon)**2   *   mask ) \
        * 100.0


x_im_patch = tf.extract_image_patches(x_reshape,
        ksizes=[1,31,31,1],
        strides=[1,15,15,1],
        rates=[1,1,1,1],
        padding='VALID')

x_init=2
y_init=4
#img_y = img_y[:,rnd_init[0]:, rnd_init[1]:]
y_im_patch = tf.extract_image_patches(tf.expand_dims(img_y, axis=3),
        ksizes=[1,33,33,1],
        strides=[1,17,17,1],
        rates=[1,1,1,1],
        padding='VALID')


#        + tf.reduce_mean( tf.abs(img_y - recon) )
#loss = tf.reduce_mean(tf.square(tf.subtract(img_y, recon)))# + tf.abs(recon))
opt = tf.train.AdamOptimizer(1e-6)
train_op = opt.minimize(loss)
update_op = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

#model.pretrain_load(sess)
saver = tf.train.Saver()
#saver.restore(sess, 'models/supervision.ckpt')

avg = Avg(['loss'])
for i in range(1, 1+max_iter):
    x, y = get_train_pair(8)
    rnd = np.random.randint(100, size=2)
    fd = {img_x: x, img_y: y[:, rnd[0]:, rnd[1]:]}
#    _, _, l = sess.run([train_op, update_op, loss], fd)
    l = 0

    p = sess.run(y_im_patch, fd)
    print (p.shape)
    p = sess.run(img_y, fd)
    print (p.shape)
    avg.add(l, 0)
    if i % 10 == 0:
        avg.show(i)
    if i % 10 == 0:
        rc, rx, ry = sess.run([recon, img_x, img_y], fd)
        for k in range(rc.shape[0]):
            np.save('sample_imgs/a_'+str(k)+'.npy', rc[k])
            np.save('sample_imgs/x_'+str(k)+'.npy', rx[k])
            np.save('sample_imgs/y_'+str(k)+'.npy', ry[k])
            np.save('sample_imgs/p.npy', p)
        avg.description()
        print (np.mean(rc), np.mean(ry), np.mean(rx))
        saver.save(sess, save_dir + 'a.ckpt')
