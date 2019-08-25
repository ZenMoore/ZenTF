
### 创建变量

```python
    #下面两句等价
v = tf.get_variable("v", shape=[1], initializer=tf.constant_initializer(1.0))
v = tf.Variable(tf.constant(1.0, shape=[1]), name="v")

    #在命名空间中创建变量
with tf.variable_scope("foo"):
    v = tf.get_variable("v", shape=[1], initializer=tf.constant_initializer(1.0))
```

### 获取变量
```python
    #需在相应命名空间提前创建变量才能获取
with tf.variable_scope("foo"):
    tf.get_variable(name="v", shape=[1])    #报错，已有重名变量

with tf.variable_scope("foo", reuse=True):  #只能获取不能创建变量
    tf.get_variable(name="v", shape=[1])

with tf.variable_scope("bar", reuse=True):
    tf.get_variable(name="v", shape=[1])    #报错，命名空间中未定义v
```

### 嵌套命名空间
```python
with tf.variable_scope("root"):
    print(tf.get_variable_scope().reuse)    #false

    with tf.variable_scope("foo",reuse=True):
        print(tf.get_variable_scope().reuse)    #true

        with tf.variable_scope("bar"):
            print(tf.get_variable_scope().reuse)    #true


    print(tf.get_variable_scope().reuse)    #true
```

### 变量管理
```python
v1 = tf.get_variable("v", [1])
print(v1.name)  #v:0

with tf.variable_scope("foo"):
    v2 = tf.get_variable("v", [1])
    print(v2.name)  #foo/v:0

v3 = tf.get_variable("foo/v",[1])
print(v3 == v2) #true
```

### 通过变量管理改进的前向传播函数inference
```python
def inference(input_tensor, reuse=False):
    with tf.variable_scope('layer1', reuse=reuse):
        weights = tf.get_variable("weights",...)
        biases = ...
        layer1 = ...

    with tf.variable_scope('layer2', reuse=reuse):
        ...layer2

    return layer2

y = inference(x)
new_y = inference(x, True)  #在第一次使用的时候创建变量，使用默认设置，之后每次使用前向传播的时候都指定True
```

