import tensorflow as tf

lstm_hidden_size = 10 # lstm结构是一个神经网络，包含多个隐藏层，这里包含10个lstm单元
batch_size = 10
num_steps = 10 # 循环神经网络展开的序列长度
input = []  # 输入序列


# lstm中所使用的变量也在这个函数中声明
lstm = tf.nn.rnn_cell.BasicLSTMCell(lstm_hidden_size)

# 将lstm的初始状态设置为全0数组，BasicLSTMCell提供了zero_state函数来生成全0的初始状态
# state是一个包含两个张量的LSTMStateTuple类
# state.c 代表c状态，state.h代表h状态
state = lstm.zero_state(batch_size= batch_size, dtype= tf.float32)

loss = 0.0

for i in range(num_steps):
    # 在第一个时刻声明LSTM中使用的变量，之后的每一个时刻都要复用这个变量以实现共享参数
    if i > 0 : tf.get_variable_scope().reuse_variables()

    # 每一步处理一个时刻
    # 将current_input和state(上一时刻)传入lstm结构中得到：
    # lstm_output即h_t, state即h_t和c_t
    # lstm_output用于输出给其他层，state用于输出给下一时刻，它们在dropout中可以有不同的处理方式
    lstm_output, state = lstm(input[i], state)
    final_output = fully_connected(lstm_output)
    loss += calc_loss(final_output, expected_output)

# 之后使用类似第四章介绍的方法训练模型





