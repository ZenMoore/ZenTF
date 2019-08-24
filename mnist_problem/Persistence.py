import tensorflow as tf

#保存模型
v1 = tf.Variable(tf.constant(1.0, shape=[1]), name="v1")
v2 = tf.Variable(tf.constant(2.0, shape=[1]), name="v2")
result = v1 + v2

init_op = tf.global_variables_initializer()
saver = tf.train.Saver()

with tf.Session() as sess:
    sess.run(init_op)
    saver.save(sess, "model_saved/model.ckpt")

#加载模型: 加载值不加载结构
v1 = tf.Variable(tf.constant(1.0, shape=[1]), name="v1")
v2 = tf.Variable(tf.constant(2.0, shape=[1]), name="v2")
result = v1 + v2    #由于result没有指定name，所以加载模型时候只能匹配到v1和v2并赋值

saver = tf.train.Saver()

with tf.Session() as sess:
    saver.restore(sess, "model_saved/model.ckpt")
    print(sess.run(result))

#加载模型: 加载结构
saver = tf.train.import_meta_graph("model_saved/model.ckpt/model.ckpt.meta")

with tf.Session() as sess:
    saver.restore(sess, "model_saved/model.ckpt")
    print(sess.run(tf.get_default_graph().get_tensor_by_name("add:0")))

#加载模型: 指定加载变量的列表
saver = tf.train.Saver([v1])    #未加载的变量不会被初始化

#保存或者加载模型是对变量重命名
v1 = tf.Variable(tf.constant(1.0, shape=[1]), name="other-v1")
v2 = tf.Variable(tf.constant(2.0, shape=[1]), name="other-v2")
#直接调用tf.train.Saver的话不能匹配到变量，因为保存模型中v1\v2的变量名无法与other-v1\v2匹配，报错
saver = tf.train.Saver({"v1": v1, "v2": v2})

#变量重命名的一个作用就是：方便使用变量的滑动平均值
v = tf.Variable(0, dtype=tf.float32, name="v")
for variables in tf.global_variables():
    print(variables.name) #v:0

ema = tf.train.ExponentialMovingAverage(0.99)
maintain_averages_op = ema.apply(tf.global_variables())
for variables in tf.global_variables():
    print(variables.name) #v:0和影子变量v/ExponentialMovingAverage:0

saver =  tf.train.Saver()
with tf.Session() as sess:
    init_op = tf.global_variables_initializer()
    sess.run(init_op)

    sess.run(tf.assign(v, 10))
    sess.run(maintain_averages_op)
    saver.save(sess, "model_saved/model.ckpt")
    print(sess.run([v, ema.average(v)]))
# 直接加载滑动平均值
v = tf.Variable(0, dtype=tf.float32, name="v")

saver = tf.train.Saver({"v/ExponentialMovingAverage": v})
with tf.Session() as sess:
    saver.restore(sess, "model_saved/model.ckpt")
    print(sess.run(v)) #输出滑动平均值

#加载滑动平均值: 使用自动生成的字典
print(ema.variables_to_restore()) #字典
saver = tf.train.Saver(ema.variables_to_restore())