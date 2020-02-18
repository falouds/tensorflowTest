import tensorflow as tf 
import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np
from sklearn.datasets.california_housing import fetch_california_housing
from sklearn import tree
from sklearn.model_selection import train_test_split#分割训练集
from sklearn.ensemble import RandomForestRegressor#系统自己调整参数

print("Tensorflow version: {}".format(tf.version))
data= pd.read_csv('./tensorflow-data/20190905(16-24) - 标记.csv',encoding='ANSI')

dtr = RandomForestRegressor(random_state = 42,max_depth=12)


resourse = data.iloc[:,2:-1]
label = np.array(data.iloc[:,-1])


columnsName = resourse.columns.values.tolist()
for indexs in resourse.columns:
    lista = []
    print(indexs)
    for i, v in resourse[indexs].items():
        if(v in lista):
            resourse.loc[i:i,(indexs)] = lista.index(v)
        elif(isinstance(v,int) or isinstance(v,float)):
            lista.append(v)
            resourse.loc[i:i,(indexs)] = lista.index(v)
        else:
            lista.append(v)
            resourse.loc[i:i,(indexs)] = lista.index(v)
    with open("./tensorflow-model/tpot/sign.txt",'a', encoding='ANSI') as file:
        if(file.writable()):
            file.write(str(lista))

resourse = np.array(resourse)
pd_data = pd.DataFrame(resourse)
pd_data.to_csv("./tensorflow-model/tpot/change.csv",'w',encoding="ANSI")

print(resourse)


data_train,data_test,target_train,target_test = \
    train_test_split(resourse,label,test_size=0.1,random_state = 42)



dtr.fit(data_train,target_train)
print(dtr.score(data_test,target_test))
import os
import pydotplus
os.environ["PATH"] += os.pathsep + 'D:/Program Files (x86)/Graphviz2.38/bin/'
for i,estimators in enumerate(dtr.estimators_):
    dot_data = \
        tree.export_graphviz(
            estimators,
            out_file = None,
            feature_names=columnsName,
            filled = True,
            impurity = False,
            rounded = True
        ) 
    graph = pydotplus.graph_from_dot_data(dot_data)
    graph.get_nodes()[7].set_fillcolor("#FFF2DD")
    graph.write_png("./resForeast/res" + str(i) + ".png")

print(dtr.score(data_test,target_test))




# model = tf.keras.Sequential()
# model.add(tf.keras.layers.Dense(10,input_shape=(78,),activation='relu'))
# model.add(tf.keras.layers.Dense(128,activation='relu'))#d过大会导致过拟合，太小会丢失特征
# model.add(tf.keras.layers.Dense(128,activation='relu'))
# model.add(tf.keras.layers.Dense(128,activation='relu'))
# model.add(tf.keras.layers.Dense(6,activation='softmax'))#计算映射到概率

# check_path="./tensorflow-model/tpot/ch-1.ckpt"
# dir_path = os.path.dirname(check_path)
# callback = tf.keras.callbacks.ModelCheckpoint(check_path,save_weights_only=True,verbose=1,period=10)
# #第一个参数自动7保存的路径，第二个是保存权重还是模型，第三个参数表示是否显示提示，第四个指步长
# print(model.summary())
# model.compile(
#     optimizer='adam',
#     loss='sparse_categorical_crossentropy',
#     metrics=['acc']
#     )
# history = model.fit(
#     train_resourse,
#     train_label,
#     epochs=200,
#     callbacks=[callback],
#     validation_data=(test_resourse,test_label)
#     )
#-----------------------------------------------------
# model.save("./tensorflow-model/tpot/modal.h5")#保存模型
# model.save_weights("./tensorflow-model/tpot/modal_weight")
# #建立新的模型用load_weights,载入权重

# plt.plot(history.epoch,history.history.get('loss'),label='loss')
# plt.plot(history.epoch,history.history.get('val_loss'),label='val_loss')
# plt.show()

# plt.plot(history.epoch,history.history.get('acc'),label='acc')
# plt.plot(history.epoch,history.history.get('val_acc'),label='val_acc')
# plt.show()