
激活函数<br>
```
 a = tf.nn.relu(tf.matmul(x, w1) + biases1)
 y = tf.nn.relu(tf.matmul(a, w2) + biases2)
```

交叉熵损失函数
```
 cross_entropy = -tf.reduce_mean(y_ * tf.log(tf.clip_by_value(y, 1e-10, 1.0)))
 cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y)  logits指逻辑回归、评定模型
 cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y_, logits=y)  在只有一个正确答案的分类问题中，通过该函数加速计算过程
```

均方误差损失函数
```
 mse = tf.reduce_mean(tf.square(y_, y))
```

自定义损失函数示例
```
 loss = tf.reduce_sum(tf.where(tf.greater(v1, v2), (v1-v2)*a, (v2-v1)*b))  选择函数
```

采用batch的随机梯度下降
```
 可见/rudiment/ANNs_Sample.py
```

学习率指数衰减法
```
 global_step = tf.Variable(0)
 learning_rate = tf.train.exponential_decay(0.1, global_step, 100, 0.96, staircase=True)
 learning_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(...my_loss..., global_step=global_step) 传入global_step将自动更新global_step参数，从而使得学习率也得到相应更新
```

L2正则化去过拟合
```
 loss = tf.reduce_mean(tf.square(y_ - y))+tf.contrib.layers.l2_regularizer(lambda)(w)  l2_regularizer返回一个函数，类似函数式编程的柯西化
 当神经网络参数增多，上面的方式可能导致损失函数定义过长，而且可能网络结构定义和损失函数计算不在一个函数中，因此通过变量计算损失函数不很方便
 可以使用TensorFlow提供的集合
 示例：通过集合计算一个5层神经网络带L2正则化的损失函数的计算方法 /deep_nn/L2_with_collection.py
```

滑动平均模型
```
 见/deep_nn/MovingAverageSample.py
```

