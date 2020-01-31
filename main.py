import tensorflow as tf 
import pandas as pd
import matplotlib.pyplot as plt

print("Tensorflow version: {}".format(tf.version))
data = pd.read_csv('./tensorflow-data/Income1.csv')
x = data.Education
y = data.Income
model = tf.keras.Sequential()#顺序模型
model.add(tf.keras.layers.Dense(1,input_shape=(1,)))#建立模拟函数,参数：输入数据的维度与输出数据的维度
model.compile(optimizer='adam',loss='mse')#优化方法计算梯度贴近真实值，损失函数均方差
print(model.summary())#返回模型的形状
print(model)
#history = model.fit(x,y,epochs=5000)
#plt.scatter(data.Education,data.Income)
plt.show()