import tensorflow as tf

reader = tf.TFRecordReader()
#创建一个队列来维护输入文件列表
filename_queue = tf.train.string_input_producer(['output.tfrecords'])

#从文件中读取一个样例，也可以使用read_up_to一次读取多个样例
_, serialized_example = reader.read(filename_queue)
#解析读入的一个样例，用parse_example解析多个样例
features = tf.parse_single_example(
    serialized_example,
    features={
        #TensorFlow提供两种不同的属性解析方法
        #FixedLenFeature: 解析为Tensor
        #VarLenFeature: 解析为SparseTensor, 用于处理稀疏数据
        #这里解析的格式需要和存储时候的格式一致
        'image_raw' : tf.FixedLenFeature([], tf.string),
        'pixels' : tf.FixedLenFeature([], tf.int64),
        'label' : tf.FixedLenFeature([], tf.int64),
    }
)

#解编码，从string到像素数组
image = tf.decode_raw(features['image_raw'], tf.uint8)
label = tf.cast(features['label'], tf.int32)
pixels = tf.cast(features['pixels'], tf.int32)

with tf.Session() as sess:
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess= sess, coord = coord)

    #每次读取一个样例，读完后重头读取
    for i in range(10):
        print(sess.run([image, label, pixels]))
