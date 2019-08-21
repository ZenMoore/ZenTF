import tensorflow as tf

v1 = tf.Variable(0, dtype=tf.float32) #定义一个变量用于计算滑动平均，初始值为0,必须为实数型
step = tf.Variable(0, trainable=False) #模拟迭代轮数，赋给num_updates,可以用于动态控制衰减率

ema = tf.train.ExponentialMovingAverage(0.99, step)

maintain_averages_op = ema.apply([v1])

with tf.Session() as sess:
    init_op = tf.global_variables_initializer()
    sess.run(init_op)

    print(sess.run([v1, ema.average(v1)]))

    sess.run(tf.assign(v1, 5))
    sess.run(maintain_averages_op)
    print([v1, ema.average(v1)])

    sess.run(tf.assign(step, 10000))
    sess.run(tf.assign(v1, 10))
    sess.run(maintain_averages_op)
    print(sess.run([v1, ema.average(v1)]))

    sess.run(maintain_averages_op)
    print(sess.run([v1, ema.average(v1)]))
