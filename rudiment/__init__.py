import os
import tensorflow as tf

if __name__ == '__main__':
    # os.system("python ANNs_Sample.py")
    v = tf.constant([[1.0, 2.0, 3.0],[4.0, 5.0, 6.0]])

    with tf.Session().as_default():
        print(tf.reduce_mean(v).eval())