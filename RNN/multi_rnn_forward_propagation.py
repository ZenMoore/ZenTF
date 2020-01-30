import tensorflow as tf

lstm_size = 10
numbre_of_layers = 5 # 纵向层数，即从x到o的层数
baatch_size = 10
num_steps = 10

lstm_cell = tf.nn.rnn_cell.BasicLSTMCell

# 不能使用[lstm_cell(lstm_size)] * N, 否则会共享参数(每一层循环体中参数是一致的，而不同层中的参数可以不同。)
stacked_lstm = tf.nn.rnn_cell.MultiRNNCell([lstm_cell(lstm_size) for _ in range(numbre_of_layers)])

state = stacked_lstm.zero_state(baatch_size, dtype= tf.float32)
for i in range(num_steps):
    if i > 0: tf.get_variable_scope().reuse_variables()

    stacked_lstm_output, state = stacked_lstm(current_input, state)
    final_output = fully_connected(stacked_lstm_output)
    loss += calc_loss(final_output, expected_output)


# 由此可见，在TensorFlow中只需要在BasicLSTMCell的基础上再封装一层MultiRNNCell就能够很容易的实现DRNN


