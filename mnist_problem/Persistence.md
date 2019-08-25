### 保存模型
```python
v1 = tf.Variable(tf.constant(1.0, shape=[1]), name="v1")
v2 = tf.Variable(tf.constant(2.0, shape=[1]), name="v2")
result = v1 + v2

init_op = tf.global_variables_initializer()
saver = tf.train.Saver()

with tf.Session() as sess:
    sess.run(init_op)
    saver.save(sess, "model_saved/model.ckpt")
```

### 加载模型: 加载值不加载结构
```python
v1 = tf.Variable(tf.constant(1.0, shape=[1]), name="v1")
v2 = tf.Variable(tf.constant(2.0, shape=[1]), name="v2")
result = v1 + v2    #由于result没有指定name，所以加载模型时候只能匹配到v1和v2并赋值

saver = tf.train.Saver()

with tf.Session() as sess:
    saver.restore(sess, "model_saved/model.ckpt")
    print(sess.run(result))
```

### 加载模型: 加载结构
```python
saver = tf.train.import_meta_graph("model_saved/model.ckpt/model.ckpt.meta")

with tf.Session() as sess:
    saver.restore(sess, "model_saved/model.ckpt")
    print(sess.run(tf.get_default_graph().get_tensor_by_name("add:0")))
```

### 加载模型: 指定加载变量的列表
```python
saver = tf.train.Saver([v1])    #未加载的变量不会被初始化
```

### 保存或者加载模型是对变量重命名
```python
v1 = tf.Variable(tf.constant(1.0, shape=[1]), name="other-v1")
v2 = tf.Variable(tf.constant(2.0, shape=[1]), name="other-v2")
#直接调用tf.train.Saver的话不能匹配到变量，因为保存模型中v1\v2的变量名无法与other-v1\v2匹配，报错
saver = tf.train.Saver({"v1": v1, "v2": v2})
```

### 变量重命名的一个作用就是：方便使用变量的滑动平均值
```python
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
```

### 直接加载滑动平均值
```python
v = tf.Variable(0, dtype=tf.float32, name="v")

saver = tf.train.Saver({"v/ExponentialMovingAverage": v})
with tf.Session() as sess:
    saver.restore(sess, "model_saved/model.ckpt")
    print(sess.run(v)) #输出滑动平均值
```

### 加载滑动平均值: 使用自动生成的字典
```python
print(ema.variables_to_restore()) #字典
saver = tf.train.Saver(ema.variables_to_restore())
```

---
###  下面的代码解决两个问题：
1. Saver保存了全部信息，而有时候很多信息并不需要(比如变量初始化以及辅助节点等等)

2. 将变量的取值和计算图结构分成不同的文件保存不方便

##### 使用convert_variables_to_constants函数可以将图中的变量以及取值以常量的方式保存，这样整个计算图可以存在一个文件中

### 保存_代码
```python
import tensorflow as tf
from tensorflow.python.framework import graph_util

v1 = tf.Variable(tf.constant(1.0, shape=[1]), name="v1")
v2 = tf.Variable(tf.constant(2.0, shape=[1]), name="v2")
result = v1 + v2

init_op = tf.global_variable_initializer()

with tf.Session() as sess:
    sess.run(init_op)
    #导出当前计算图中的Graph_def部分，只需要这一部分就能完成从输入层到输出层的计算过程
    graph_def = tf.get_default_graph().as_graph_def()
    #将图中的变量值转化为常量，同时去掉不必要的节点
    #最后一个参数['add']给出了需要保存的节点的名字，注意是节点名字所以没有:0
    output_graph_def = graph_util.convert_variables_to_constants(sess, graph_def, ['add'])

    with tf.gfile.GFile("model_saved/combined_model.pb", "wb") as f:
        f.write(output_graph_def.SerializeToString())
```

### 使用_代码
```python
import tensorflow as tf
from tensorflow.python.platform import gfile

with tf.Session() as sess:
    model_filename = "model_saved/combined_model.pb"
    with gfile.FastGFile(model_filename, "rb") as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
    result = tf.import_graph_def(graph_def, return_elements=["add:0"])
    print(sess.run(result))
```