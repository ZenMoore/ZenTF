import numpy as np
import tensorflow as tf
import matplotlib as mpl #数学绘图
from matplotlib import pyplot as plt

mpl.use("Agg") #选择画图引擎

HIDDEN_SIZE = 30
NUM_LAYERS = 2

TIMESTEPS = 10 # 序列长度
TRAINING_STEPS = 10000 # 训练轮数
BATCH_SIZE = 32

TRAINING_EXAMPLES = 10000 # 训练数据个数
TESTING_EXAMPLES = 1000
SAMPLE_GAP = 0.01 # 采样间隔，将连续的正弦函数离散化为计算机能够处理的数据序列

def generate_data(seq):
    X = []
    y = []

    # 序列第i项和后面总共TIMESTEPS-1项作为输入，i+TIMESTEPS项作为输出
    for i in range(len(seq) - TIMESTEPS):
        X.append([seq[i : i+TIMESTEPS]])
        y.append([seq[i+TIMESTEPS]])

        return np.array(X, dtype= np.float32), np.array(y, dtype= np.float32)

#注意：这里传入的X已经是TIMESTEPS长度了，事先已经使用上面的这个函数或者其他方式截取了TIMESTEPS
def lstm_model(X, y, is_training):
    cell = tf.nn.rnn_cell.MultiRNNCell([tf.nn.rnn_cell.BasicLSTMCell(HIDDEN_SIZE) for _ in range(NUM_LAYERS)])

    # 使用tf接口将多层的lstm结构连接成RNN网络并计算前向传播结果
    outputs, _ = tf.nn.dynamic_rnn(cell, X, dtype= tf.float32)

    # outputs是顶层lstm在每一步的输出结果，维度为 [BATCH_SIZE, TIMESTEPS, HIDDEN_SIZE],，我们只关心最后时刻的输出

    output = outputs[:, -1, :]

    predictions = tf.contrib.layers.fully_connected(output, 1, activation_fn= None)

    if not is_training:
        return predictions, None, None

    loss = tf.losses.mean_squared_error(labels= y, predictions= predictions)

    train_op = tf.contrib.layers.optimize_loss(
        loss, tf.train.get_global_step(), optimizer= "Adagrad", learning_rate= 0.1
    )

    return predictions, loss, train_op

def train(sess, train_X, train_y):
    # 将训练数据以数据集的方式提供给计算图
    ds = tf.data.Dataset.from_tensor_slices((train_X, train_y))
    ds = ds.repeat().shuffle(1000).batch(BATCH_SIZE)

    X, y = ds.make_one_shot_iterator().get_next()

    with tf.variable_scope("model"):
        predictions, loss, train_op = lstm_model(X, y, True)

    sess.run(tf.global_variables_initializer())

    for i in range(TRAINING_STEPS):
        _, l = sess.run([train_op, loss])
        if i % 100 == 0:
            print("train step: " + str(i) + ", loss: " + str(l))

def run_eval(sess, test_X, test_y):
    ds = tf.data.Dataset.from_tensor_slices((test_X, test_y))
    ds = ds.batch(1)
    X, y = ds.make_one_shot_iterator().get_next()

    with tf.variable_scope("model", reuse= True):
        prediction, _, _ = lstm_model(X, [0.0], False) # [0.0]没有实际意义

    predictions = []
    labels = []
    for i in range(TESTING_EXAMPLES):
        p, l = sess.run([prediction, y])
        predictions.append(p)
        labels.append(l)

    predictions = np.array(predictions).squeeze() # 将行向量转换为列向量，实质上是将多维向量挤压成单维向量
    labels = np.array(labels).squeeze()
    rmse = np.sqrt(((predictions - labels) ** 2).mean(axis= 0))
    print("Mean Square Error is: %f" % rmse)

    plt.figure()
    plt.plot(predictions, label= 'predictions')
    plt.plot(labels, label= 'real_sin')
    plt.legend()
    plt.show()

test_start = (TRAINING_EXAMPLES + TIMESTEPS) * SAMPLE_GAP
test_end = test_start + (TESTING_EXAMPLES + TIMESTEPS) * SAMPLE_GAP
train_X, train_y = generate_data(np.sin(np.linspace(0, test_start, TRAINING_EXAMPLES + TIMESTEPS, dtype=np.float32)))
test_X, test_y = generate_data(np.sin(np.linspace(test_start, test_end, TESTING_EXAMPLES + TIMESTEPS, dtype=np.float32)))

with tf.Session() as sess:
    train(sess, train_X, train_y)
    run_eval(sess, test_X, test_y)
