import tensorflow as tf
import numpy as np
import time

'''
ver@170524
'''

def fc(name, out_channel, prev, r=False, t=True):
    input_shape = prev.get_shape()
    if input_shape.ndims==4:
        dim=1
        for d in input_shape[1:].as_list():
            dim*=d
        in_channel = dim
        prev = tf.reshape(prev, [-1, in_channel])
    else:
        in_channel = input_shape.as_list()[-1]
    shape = [in_channel, out_channel]
    with tf.variable_scope(name):
        if r:
            tf.get_variable_scope().reuse_variables()
            w = tf.get_variable('w', trainable=t)
            b = tf.get_variable('b', trainable=t)
        else:
            init = tf.truncated_normal(shape, stddev=.01)
            w = tf.get_variable('w', initializer=init, trainable=t)
            init = tf.constant(.0, shape=[shape[-1]])
            b = tf.get_variable('b', initializer=init, trainable=t)
    return tf.matmul(prev, w) + b

def _conv2d(x, W, stride):
    return tf.nn.conv2d(x, W, strides=[1,stride,stride,1], padding='SAME')

def pool(x):
    return tf.nn.max_pool(x, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')

def avg_pool(x):
    return tf.nn.avg_pool(x, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')

def unpool(x):
    shape = x.get_shape().as_list()
    outsize = [shape[1]*2, shape[2]*2]
    out = tf.image.resize_bilinear(x, size=outsize)
    return out

def gl_avg_pool(l):
#    assert l.get_shape().ndims == 4
    return tf.reduce_mean(l, [1,2])

def activation(l, type_a='none'):
    if type_a=='none':
        return l
    elif type_a=='relu':
        return tf.nn.relu(l)

def conv(name, shape, prev, reuse=False, trainable=True, stride=1):
    dev_n = shape[1]*shape[1]*shape[3]
    with tf.variable_scope(name):
        if reuse:
            tf.get_variable_scope().reuse_variables()
            w = tf.get_variable('w', trainable=trainable)
            b = tf.get_variable('b', trainable=trainable)
        else:
            init = tf.truncated_normal(shape, stddev=2.0/dev_n)
            w = tf.get_variable('w', initializer=init, trainable=trainable)
            init = tf.constant(.0, shape=[shape[-1]])
            b = tf.get_variable('b', initializer=init, trainable=trainable)
    return _conv2d(prev, w, stride) + b

def deconv(name, shape, prev, reuse=False, trainable=True, activation=True, stride=1):
    dev_n = shape[1]*shape[1]*shape[3]
    with tf.variable_scope(name):
        if reuse:
            tf.get_variable_scope().reuse_variables()
            w = tf.get_variable('w', trainable=trainable)
            b = tf.get_variable('b', trainable=trainbale)
        else:
            init = tf.truncated_normal(shape, stddev=2.0/dev_n)
            w = tf.get_variable('w', initializer=init, trainable=trainable)
            init = tf.constant(.0, shape=[shape[-1]])
            b = tf.get_variable('b', initializer=init, trainable=trainable)
#    return tf.nn.conv2d_transpose(prev, w,
    out = tf.contrib.layers.conv2d_transpose(prev,
            num_outputs=shape[-1],
            kernel_size=[shape[0],shape[1]],
            weights_initializer=\
                    tf.contrib.layers.xavier_initializer_conv2d(uniform=False),
            biases_initializer=\
                    tf.contrib.layers.xavier_initializer(uniform=False),
            activation_fn=tf.nn.relu if activation else None)
    return out

def get_wb(name, shape, r=False, t=True):
    with tf.variable_scope(name):
        if r:
            tf.get_variable_scope().reuse_variables()
            w = tf.get_variable('w', trainable=t)
            b = tf.get_variable('b', trainable=t)
        else:
            init = tf.truncated_normal(shape, stddev=.01)
            w = tf.get_variable('w', initializer=init, trainable=t)
            init = tf.constant(.0, shape=[shape[-1]])
            b = tf.get_variable('b', initializer=init, trainable=t)
    return w, b

def batch_norm(name, x, isTr, r=False, t=True):
    out_channel = x.get_shape().as_list()[3]
    with tf.variable_scope(name):
        if r:
            tf.get_variable_scope().reuse_variables()
            beta = tf.get_variable('w', trainable=t)
            gamma = tf.get_variable('b', trainable=t)
        else:
            beta = tf.get_variable('w', initializer=tf.constant(.0,
                shape=[out_channel]), trainable=t)
            gamma = tf.get_variable('b', initializer=tf.constant(1.0,
                shape=[out_channel]), trainable=t)
    batch_mean, batch_var = tf.nn.moments(x, [0, 1, 2])
    ema = tf.train.ExponentialMovingAverage(decay=.5)
    def mean_var_with_update():
        ema_apply_op = ema.apply([batch_mean, batch_var])
        with tf.control_dependencies([ema_apply_op]):
            return tf.identity(batch_mean), tf.identity(batch_var)

    mean, var = tf.cond(isTr, mean_var_with_update,
            lambda: (ema.average(batch_mean),
            ema.average(batch_var)))
    normed = tf.nn.batch_normalization(x, mean, var, beta, gamma, 1e-3)
    return normed

def bottleneck_block(name, prev, isTr, chan_k):
    shape = prev.get_shape().as_list()
    in_channel = shape[3]
    internal = chan_k * 4
    l = batch_norm(name+'b1', prev, isTr)
    l = tf.nn.relu(l)
    l = conv(name+'c1', [1,1,in_channel, internal], l)
    l = batch_norm(name+'b2', l, isTr)
    l = tf.nn.relu(l)
    l = conv(name+'c2', [3,3,internal, chan_k], l)
    l = tf.concat([prev, l], 3)
    return l

def transition(name, prev, out_chan, isTr):
    shape = prev.get_shape().as_list()
    in_channel = shape[3]
    l = batch_norm(name+'b1', prev, isTr)
    l = tf.nn.relu(l)
    l = conv(name+'c1', [1,1,in_channel,out_chan], l)
    l = avg_pool(l)
    return l

def dense(in_x, isTr, depth=100, growth_K=12, comp_rate=.5):
    recurrent_N = (depth-4)/3/2
    in_x = in_x / 128.0 - 1
    k0 = k = 16
    l = conv('c1', [3,3,3,k0], in_x)
    for i in range(recurrent_N):
        l = bottleneck_block('block1.{}'.format(i), l, isTr, growth_K)
        k = k + growth_K
    k = int(k * comp_rate)
    l = transition('t1', l, k, isTr)

    for i in range(recurrent_N):
        l = bottleneck_block('block2.{}'.format(i), l, isTr, growth_K)
        k = k + growth_K
    k = int(k * comp_rate)
    l = transition('t2', l, k, isTr)

    for i in range(recurrent_N):
        l = bottleneck_block('block3.{}'.format(i), l, isTr, growth_K)

    l = batch_norm('bnlast', l, isTr)
    l = tf.nn.relu(l)
    l = gl_avg_pool(l)
    return l

def cross_entropy(tar_y, y):
    return -tf.reduce_mean(tf.reduce_sum(tar_y*tf.log(y+1e-10),1))

def l2loss(a, b):
    return tf.reduce_mean(tf.reduce_sum((a-b)**2, 1))

def tf_acc(a, b):
    acc = tf.equal(tf.argmax(a, 1), tf.argmax(b, 1))
    return tf.reduce_mean(tf.cast(acc, tf.float32))

def cos_sim(batch, exemplar):
    u = np.dot(batch, np.transpose(exemplar))
    d1 = np.linalg.norm(batch, axis=1) + 1e-10
    d2 = np.linalg.norm(exemplar, axis=1) + 1e-10

    d1 = np.reshape(d1, [-1, 1])
    d2 = np.reshape(d2, [1, -1])
    return u / d1 / d2

def np_acc(pred, true):
    pred = np.argmax(pred, 1)
    true = np.argmax(pred, 1)
    acc = pred == true
    acc = np.mean(acc)
    return acc

class Avg():
    def __init__(s, desc=None):
        n = 4
        s.cnt = np.zeros([n])
        s.vals = np.zeros([n])
        s.sav = np.zeros([n])
        s.start_time = time.time()
        s.end_time = .0
        s.desc = desc
        s.description()

    def description(self):
        self.end_time = time.time()
        print (self.desc)
        print ('time (s) %d '%(self.end_time - self.start_time))
        self.start_time = time.time()

    def get_val(s, n):
        return s.sav[n]

    def add(s, val, n):
        s.cnt[n] += 1
        s.vals[n] += val

    def show(s, step):
        for n in range(np.shape(s.cnt)[0]):
            if s.cnt[n] == 0:
                s.sav[n] = 0
            else:
                s.sav[n] = s.vals[n] / s.cnt[n]
        print ('step: %6d ||  v[0]: %.3f,   v[1]: %.3f,   v[2]: %.3f,  v[3]: %.3f'\
                %(step,s.sav[0], s.sav[1], s.sav[2], s.sav[3]))
        s.vals[:] = .0
        s.cnt[:] = 0

def to1hot(label, N_class):
    n_data = np.shape(label)[0]
    onehot = np.zeros([n_data, N_class])
    onehot[np.arange(n_data), label] = 1
    return onehot

def KL_div(pm, pe):
    pe = tf.reshape(pe, [1,-1,5])
    ent = pm * tf.log(pm + 1e-10) - pm * tf.log(pe + 1e-10)
    ent = tf.reduce_sum(ent, 2)
    ent = tf.reduce_mean(ent, 0)
    return ent

def apply_window(img, window_level=1040, window_width=200):
    img[img<window_level - window_width/2.0] = window_level - window_width / 2.0
    img[img>window_level + window_width/2.0] = window_level + window_width / 2.0

    a = 255.0 / window_width
    b = 255.0 / (0.5 - window_level / window_width)
    img = img * a + b
    return img

def get_patch_num(i, k, s, initw, inith, bsize):
    '''
    i: img size
    k: kernel size
    s: stride
    initw: start loc of w
    '''
    wt = int(512 - k - initw)
    wt = int(wt / s + 1)
    ht = int(512 - k - inith)
    ht = int(ht / s + 1)
    return wt * ht * bsize
