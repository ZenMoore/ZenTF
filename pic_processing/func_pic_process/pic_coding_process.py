import matplotlib.pyplot as plt
import tensorflow as tf

image_raw_data = tf.gfile.FastGFile('/path/to/pic', 'r').read()

with tf.Session() as sess:
    #结果为张量，需要明确调用与运行过程
    image_decoded_data = tf.image.decode_jpeg(image_raw_data)
    print(image_decoded_data.eval())

    plt.imshow(image_decoded_data.eval())
    plt.show()

    encoded_image = tf.image.encode_jpeg(image_decoded_data)
    with tf.gfile.GFile('/path/to/output','wb') as f:
        f.write(encoded_image.eval())

