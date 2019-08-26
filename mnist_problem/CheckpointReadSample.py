import tensorflow as tf

#NewCheckpointReader可以读取checkpoint文件中保存的所有变量
#后面的.index和.data可以省去
reader = tf.train.NewCheckpointReader("model_saved/model.ckpt")

#获得变量名-变量维度的字典
global_variables = reader.get_variable_to_shape_map()
for variable_name in global_variables:
    print(variable_name, global_variables[variable_name])

print("Value for variable v1 is ", reader.get_tensor("v1"))
