import tensorflow as tf 
import pandas as pd
import matplotlib.pyplot as plt
import os

print("Tensorflow version: {}".format(tf.version))
data = pd.read_csv('./tensorflow-data/Advertising.csv')
print(data.head())
x = data.iloc[:,1:-1]
y = data.iloc[:,-1]
model = tf.keras.Sequential([tf.keras.layers.Dense(10,input_shape=(3,),activation='relu'),
    tf.keras.layers.Dense(1)
])
#
check_path="./tensorflow-model/lessonTwo/ch-1.ckpt"
dir_path = os.path.dirname(check_path)
callback = tf.keras.callbacks.ModelCheckpoint(check_path,save_weights_only=True,verbose=1,period=10)
#第一个参数自动保存的路径，第二个是保存权重还是模型，第三个参数表示是否显示提示，第四个指步长
print(model.summary())
model.compile(optimizer='adam',loss='mse')
history = model.fit(x,y,epochs=2000,callbacks=[callback])
model.save("./tensorflow-model/lessonTwo/modal.h5")#保存模型
model.save_weights("./tensorflow-model/lessonTwo/modal_weight")
#建立新的模型用load_weights,载入权重

plt.show()