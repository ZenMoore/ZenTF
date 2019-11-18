import tensorflow as tf

img_data = 0

brightness = tf.image.adjust_brightness(img_data,-0.5)

# 色彩调整的时候可能导致像素值小于0，所以需要将值截断在0.0-1.0否则无法显示影响训练，且这一截断应该在所有处理完成后进行
adjusted = tf.clip_by_value(brightness, 0.0, 1.0)

# 在区间 [-delta, delta) 随机调整亮度
delta = 0.2
adjusted = tf.image.random_brightness(img_data, delta)


# 对比度
# _contrast(img_data, 倍数)
# .random_contrast(img_data, lower, upper) 闭区间内随机

# 色相
# _hue(img_data, delta), 将色相加delta(零到一)
# .random_hue(img_data, delta), delta取值在[0, 0.5]

# 饱和度
# _saturation(img_data, delta)
# .random_saturation(img_data, lower, upper)

# 标准化，即亮度均值变为0，方差变为1
adjusted = tf.image.per_image_standardization(img_data)