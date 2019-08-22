from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("/MNIST_data", one_hot= True) #这里的one_hot是指输出结果的独热而不是输入

print("Training data_size: ", mnist.train.num_examples)
print("Validation data_size: ", mnist.validation.num_examples)
print("Testing data_size: ", mnist.test.num_examples)
print("Example training_data: ", mnist.train.images[0])
print("Example training_data_label: ", mnist.train.labels[0])

batch_size = 100
xs, ys = mnist.train.next_batch(batch_size)
print("X shape: ", xs.shape)
print("Y shape: ", ys.shape)