import tensorflow as tf

img_data = 0.0

# 处理函数要求dtype为实数，所以需要转换数据类型
# 该处理函数的输入是一个batch的数据，即四维矩阵，需要将解码之后的图像矩阵加一维
batched = tf.expand_dims(img_data, 0)
# 给出每一张图像的所有标注框，一个图像四个数字, [y_min, x_min, y_max, x_max], 为相对图像比例位置
# 这里的constant为三维矩阵，第一维表示框，第二维表示图，第三维表示batch
boxes = tf.constant([[[0.05, 0.05, 0.9, 0.7], [0.35, 0.47, 0.5, 0.56]]])
result = tf.image.draw_bounding_boxes(batched, boxes)



# 根据提供的标注框随机截取图像(随机截取比例)
# 提供标注框，告诉下面下面的那个函数图像的哪一部分是“有信息量”的
boxes = tf.constant([[[0.05, 0.05, 0.9, 0.7], [0.35, 0.47, 0.5, 0.56]]])
# min_object_covered=0.4表示截取部分至少包含标注框40%的内容
# 返回截取的一些操作值，begin表示截图位置起始点，size表示截取大小，bbox_for_draw表示截取部分等大的那个新标注框
# 注意这里给出的shape是三维矩阵，不是batch
begin, size, bbox_for_draw = tf.image.sample_distorted_bounding_box(
    tf.shape(img_data), bounding_boxes=boxes, min_object_covered=0.4
)

batched = tf.expand_dims(img_data, 0)
# 带新标注框的全图
image_with_box = tf.image.draw_bounding_boxes(batched, bbox_for_draw)

# 根据新标注框截取后的部分图像
distorted_image = tf.slice(img_data, begin, size)


