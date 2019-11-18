import tensorflow as tf

img_data = 0
flipped = tf.image.flip_up_down(img_data)
flipped = tf.image.flip_left_right(img_data)
transposed = tf.image.transpose_image(img_data)

#随机翻转, 50% 概率
flipped = tf.image.random_flip_left_right(img_data)
