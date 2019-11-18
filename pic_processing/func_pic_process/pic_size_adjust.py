import tensorflow as tf

img_data = 0
#将图片数据转换为实数类型，将0-255转换为0.0-1.0
#大多数图像处理接口支持实数和整数，但是输入整数时先转换为实数操作再输出为整数
#多次转换会导致精度损失，所以建议从一开始就转换为实数
img_data  = tf.image.convert_image_dtype(img_data, dtype=tf.float32)

#对于method
#0:双线性插值法 bilinear interpolation
#1: 最近邻居法 nearest neighbor interpolation
#2：双三次插值法 bicubic interpolation
#3：面积插值法 area interpolation
resized = tf.image.resize_images(img_data, [300, 300], method= 0)

#下面是剪裁或者自动填充的算法，剪裁为居中剪裁，填充为全零背景
cropped_or_padded = tf.image.resize_image_with_crop_or_pad(img_data, 1000, 1000)

#通过剪裁比例(零到一的实数)调整
central_cropped = tf.image.central_crop(img_data, 0.5)

#tf.image.crop_to_bounding_box, tf.image.pad_to_bounding_box 剪裁填充到特定区域，原图像大小和目标大小的要求必须符合否则报错(即原图大到足够剪裁到目标尺寸)

