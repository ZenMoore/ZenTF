import tensorflow as tf

# 四维，前两个维度表示过滤器尺寸，第三个表示当前层深度，最后一个表示过滤器深度
filter_weight = tf.get_variable('weights', [5,5,3,16], initializer=tf.truncated_normal_initializer(stddev=0.1))

biases = tf.get_variable('biases', [16], initializer=tf.constant_initializer(0.1))

#input是一个四维矩阵，第一维表示batch, 后三维表示节点矩阵。步长提供四维(batch步长、长度步长、宽度步长、深度步长，因此，首尾两维为1)
#SAME表示全0填充，VALID表示不添加
conv = tf.nn.conv2d(input, filter_weight, strides=[1,1,1,1],padding='SAME')

#这个函数提供了一个简便的方法给每一个节点加上偏置项
bias = tf.nn.bias_add(conv, biases)

actived_conv = tf.nn.relu(bias)