import tensorflow as tf 
import pandas as pd
import matplotlib.pyplot as plt
import os
  
print("Tensorflow version: {}".format(tf.version))
(train_image,train_label),(test_image,test_label) = tf.keras.datasets.fashion_mnist.load_data()
#网络下载
print("train_image :" + str(train_image.shape))
#建立新的模型用load_weights,载入权重


#归一化
train_image = train_image/255
test_image = test_image/255

model = tf.keras.Sequential()
model.add(tf.keras.layers.Flatten(input_shape=(28,28)))#扁平化为长度是28*28的向量
model.add(tf.keras.layers.Dense(128,activation='relu'))#d过大会导致过拟合，太小会丢失特征
model.add(tf.keras.layers.Dense(10,activation='softmax'))#计算映射到概率
model.compile(
    optimize='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['acc']
)
#check_path="./tensorflow-model/lessonThree/ch-1.ckpt"
#dir_path = os.path.dirname(check_path)
#callback = tf.keras.callbacks.ModelCheckpoint(check_path,save_weights_only=True,verbose=1,period=10)
#第一个参数自动保存的路径，第二个是保存权重还是模型，第三个参数表示是否显示提示，第四个指步长

model.fit(train_image,train_label,epochs=10)


#model.save("./tensorflow-model/lessonThree/modal.h5")#保存模型
#model.save_weights("./tensorflow-model/lessonThree/modal_weight")
#建立新的模型用load_weights,载入权重
model.evaluate(test_image,test_label)