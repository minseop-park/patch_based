import tensorflow as tf
import numpy as np

ksize = 6
stride = 9
img_size = 55

a = tf.placeholder(tf.float32, [img_size,img_size])

_a = tf.expand_dims(a, axis=0)
_a = tf.expand_dims(_a, axis=3)
b = tf.extract_image_patches(_a,
        ksizes=[1,ksize,ksize,1],
        strides=[1,stride,stride,1],
        rates=[1,1,1,1],
        padding='VALID')

sess = tf.Session()
sess.run(tf.global_variables_initializer())


f = np.arange(img_size**2) * 0.01
f = np.reshape(f, [img_size,img_size])
fd = {a: f}

s=  (sess.run(a, fd))
for i in range(img_size):
    print (s[i])

ppp= (sess.run(b, fd))
print (ppp)
print (ppp.shape)

#expected shape
q = int(img_size - ksize + 1)
q = int((q - 1) / stride + 1)
print (q)
