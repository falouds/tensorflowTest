import tensorflow as tf 
import pandas as pd
import matplotlib.pyplot as plt

print("Tensorflow version: {}".format(tf.version))
data = pd.read_csv('./tensorflow-data/Advertising.csv')
print(data.head())
x = data.iloc[:,1:-1]
y = data.iloc[:,-1]
model = tf.keras.Sequential([tf.keras.layers.Dense(10,input_shape=(3,),activation='relu'),
    tf.keras.layers.Dense(1)
])
print(model.summary())
model.compile(optimizer='adam',loss='mse')
history = model.fit(x,y,epochs=5000)
model.save("./tensorflow-model/lessonTwo/modal.h5")#保存模型
# plt.scatter(data.radio,data.sales)
plt.show()