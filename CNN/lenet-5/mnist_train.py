import tensorflow as tf
import os
from tensorflow.examples.tutorials.mnist import input_data
import mnist_inference
import numpy as np

BATCH_SIZE = 100
LEARNING_RATE_BASE = 0.8
LEARNING_RATE_DECAY = 0.99
MOVING_AVERAGE_DECAY = 0.99
REGULARIZATION_RATE = 0.0001
TRANING_STEPS = 30000

MODEL_SAVE_PATH = "LeNet_model_saved/"
MODEL_NAME = "model.ckpt"

def train(mnist):

    #todo 这里的 BATCH_SIZE改为None会出错，可能是因为在计算图生成的而过程中，reshape节点需要事先知道BATCH_SIZE
    x = tf.placeholder(dtype=tf.float32, shape=[BATCH_SIZE, mnist_inference.IMAGE_SIZE,mnist_inference.IMAGE_SIZE,mnist_inference.NUM_CHANNELS], name="x-input")
    y_ = tf.placeholder(dtype=tf.float32, shape=[None, mnist_inference.OUTPUT_NODE], name="y-input")

    regularizer = tf.contrib.layers.l2_regularizer(REGULARIZATION_RATE)
    y = mnist_inference.inference(x, True,  regularizer)
    global_step = tf.Variable(0, trainable=False)

    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y, labels=tf.argmax(y_, 1))
    cross_entropy_mean = tf.reduce_mean(cross_entropy)
    loss = cross_entropy_mean + tf.add_n(tf.get_collection('losses'))

    learning_rate = tf.train.exponential_decay(LEARNING_RATE_BASE, global_step, mnist.train.num_examples/BATCH_SIZE, LEARNING_RATE_DECAY)

    variable_average = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY, global_step)
    variable_average_op = variable_average.apply(tf.trainable_variables())

    train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step=global_step)
    with tf.control_dependencies([train_step, variable_average_op]):
        train_op = tf.no_op(name='train')

    saver = tf.train.Saver()

    with tf.Session() as sess:
        tf.global_variables_initializer().run()

        #在训练过程中不再测试模型在验证数据上的表现，验证和测试的过程将会有一个独立的程序完成
        for i in range(TRANING_STEPS):
            xs, ys = mnist.train.next_batch(BATCH_SIZE)

            reshaped_xs = np.reshape(xs, (BATCH_SIZE, mnist_inference.IMAGE_SIZE, mnist_inference.IMAGE_SIZE,mnist_inference.NUM_CHANNELS))

            _, loss_value, step = sess.run([train_op, loss, global_step], feed_dict={x: reshaped_xs, y_: ys})

            print(step)
            if i % 1000 == 0:
                print("After %d training step(s), loss on training batch is %g ."%(step, loss_value))
                #给出global_step参数可以让每个被保存模型的文件名末尾加上训练的轮数，比如model.ckpt=1000表示训练1000论之后的模型
                saver.save(sess, os.path.join(MODEL_SAVE_PATH, MODEL_NAME), global_step=global_step)

def main(argv = None):
    mnist = input_data.read_data_sets("MNIST_data", one_hot=True)
    train(mnist)

if __name__ == '__main__':
    tf.app.run()
