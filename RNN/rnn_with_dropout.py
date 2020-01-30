import tensorflow as tf

lstm_cell = tf.nn.rnn_cell.BasicLSTMCell

# DropoutWrapper类包装lstm_cell，控制出和入两个方面，即出有dropout入也有
# 入的保留率(100%-dropout率)通过参数(or rather, 成员变量)input_keep_prob
# 出的保留率(100%-dropout率)通过参数(or rather, 成员变量)output_keep_prob
stack_lstm = tf.nn.rnn_cell.MultiRNNCell([tf.nn.rnn_cell.DropoutWrapper(lstm_cell(lstm_size)) for _ in range(num_of_layers)])