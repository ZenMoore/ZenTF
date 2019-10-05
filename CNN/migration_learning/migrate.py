import glob
import os.path
import tensorflow as tf
import numpy as np
import tensorflow.contrib.slim.python.slim.nets.inception_v3 as inception_v3
from tensorflow.python.platform import gfile
import tensorflow.contrib.slim as slim

INPUT_DATA = 'flower_processed_data.npy'

#如果计算资源充足，可以在训练完最后的全连接层之后再训练所有的网络层，效果会好些
TRAIN_FILE = 'migrated_model'
CKPT_FILE = 'inception_v3.ckpt'

LEARNING_RATE = 0.0001
STEPS = 300
BATCH = 32
N_CLASSES = 5

#不需要加载的参数，给出参数的前缀即可
CHECKPOINT_EXCLUDE_SCOPES = 'InceptionV3/Logits,InceptionV3/AuxLogits'
TRAINABLE_SCOPES = 'InceptionV3/Logits,InceptionV3/AuxLogits'

def get_tuned_variables():
    exclusions = [scope.strip() for scope in CHECKPOINT_EXCLUDE_SCOPES.split(',')]

    variables_to_restore = []

    for var in slim.get_model_variables():#toask why slim but not inception_v3
        excluded = False
        for exclusion in exclusions:
            if var.op.name.startswith(exclusion):
                excluded = True
                break
        if not excluded:
            variables_to_restore.append(var)
    return variables_to_restore

def get_trainble_variables():
    scopes = [scope.strip() for scope in TRAINABLE_SCOPES.split(',')]
    variables_to_train = []
    for scope in scopes:
        variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope)
        variables_to_train.extend(variables) #pay attention to the difference between append and extend
    return variables_to_train

def main(argv= None):
    processed_data = np.load(INPUT_DATA)
    training_images = processed_data[0]
    n_training_example = len(training_images)
    training_labels= processed_data[1]
    validation_images = processed_data[2]
    validation_labels = processed_data[3]
    testing_images = processed_data[4]
    testing_labels = processed_data[5]
    print("%d training examples, %d validation examples and %d testing examples."%(n_training_example, len(validation_images), len(testing_images)))

    images = tf.placeholder(tf.float32, [None, 299, 299, 3], name= 'input_images')
    labels = tf.placeholder(tf.int64, [None], name= 'labels')

    #定义inception-v3模型，因为谷歌给出的只有模型参数取值，因此需要自己定义模型结构
    #虽然理论上需要区分训练和测试中使用的模型，也就是说在测试时应该使用is_training = False， 但是因为预先训练好的inception-v3模型使用的batch_normalization参数与新数据会有差异，导致结果很差，所以这里直接使用同一个模型进行测试
    #toask 这一问题详细解释见：https://github.com/tensorflow/models/issues/1314
    with slim.arg_scope(inception_v3.inception_v3_arg_scope()):
        logits, _ = inception_v3.inception_v3(images, num_classes= N_CLASSES)#toget

        trainable_variables = get_trainble_variables()

        tf.losses.softmax_cross_entropy(tf.one_hot(labels, N_CLASSES), logits, weights= 1.0)#toget

        train_step = tf.train.RMSPropOptimizer(LEARNING_RATE).minimize(tf.losses.get_total_loss())

        with tf.name_scope('evaluation'):
            correct_prediction = tf.equal(tf.argmax(logits, 1), labels)
            evaluation_step = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        #定义加载模型的函数
        load_fn = slim.assign_from_checkpoint_fn(CKPT_FILE, get_tuned_variables(), ignore_missing_vars=True)

        saver = tf.train.Saver()
        with tf.Session() as sess:
            init = tf.global_variables_initializer()
            sess.run(init)

            print('Loading tuned varaibles from %s '%CKPT_FILE)
            load_fn(sess) #toget

            start = 0
            end = BATCH
            for i in range(STEPS):
                sess.run(train_step, feed_dict= {images:training_images[start:end], labels: training_images[start:end]})

                if i % 30 == 0 or i + 1 == STEPS:
                    saver.save(sess, TRAIN_FILE, global_step= i)
                    validation_accuracy = sess.run(evaluation_step, feed_dict={images: validation_images, labels: validation_labels})
                    print('Step %d: Validation accuracy = %.1f%%'%(i, validation_accuracy * 100.0))

                start = end
                if start == n_training_example:
                    start = 0
                end = start + BATCH
                if end > n_training_example:
                    end = n_training_example

            test_accuracy = sess.run(evaluation_step, feed_dict={images: testing_images, labels: testing_labels})
            print('Final test accuracy = %.1f%%'%(test_accuracy * 100))

if __name__ == '__main__':
    tf.app.run()


