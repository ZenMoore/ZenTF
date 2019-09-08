import tensorflow as tf

slim = tf.contrib.slim



#通过slim可以在一行中实现卷积过程

#arg_scope函数中第一个参数为函数列表，后面的参数为列表中函数的同名参数提供默认值
#当在函数调用时候设置了另外的默认值，则在arg_scope中给出的默认值失效
with slim.arg_scope([slim.conv2d, slim.max_pool2d, slim.avg_pool2d], stride= 1, padding= 'VALID'):
    net = [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1]#上一层输出矩阵

    #为每一个inception模块声明一个命名空间
    with tf.variable_scope('Mixed_7c'):
        with tf.variable_scope('branch_0'):

            #conv2d有三个参数必填：第一个为输入矩阵，第二个为当前卷积层过滤器深度，第三个为过滤器尺寸
            branch_0 = slim.conv2d(net, 320, [1, 1], scope='conv2d_0a_1x1')

        with tf.variable_scope('branch_1'):
            branch_1 = slim.conv2d(net, 384, [1, 1], scope='conv2d_0b_1x1')

            # 3表示在深度这个维度上
            branch_1 = tf.concat(3, [slim.conv2d(branch_1, 384, [1, 3], scope= 'conv2d_0b_1x3'),
                                     slim.conv2d(branch_1, 384, [3,1], scope= 'conv2d_0c_3x1')])

        with tf.variable_scope('branch_2'):
            branch_2 = slim.conv2d(net, 448, [1,1], scope= 'conv2d_0a_1x1')
            branch_2 = slim.conv2d(branch_2, 384, [3,3], scope= 'conv2d_0b_3x3')

            branch_2 = tf.concat(3, [slim.conv2d(branch_2, 384, [1,3], scope= 'conv2d_0c_1x3'),
                                     slim.conv2d(branch_2, 384, [3,1], scope= 'conv2d_0d_3x1')])
        with tf.variable_scope('branch_3'):
            branch_3 = slim.avg_pool2d(net, [3, 3], scope='avgpool_0a_3x3')
            branch_3 = slim.conv2d(branch_3, 192, [1,1], scope='conv2d_0b_1x1')

        net = tf.concat(3, [branch_0, branch_1, branch_2, branch_3])