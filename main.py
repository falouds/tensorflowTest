import tensorflow as tf 
import pandas as pd
import matplotlib.pyplot as plt

# print("Tensorflow version: {}".format(tf.version))
# data = pd.read_csv('./tensorflow-data/Income1.csv')
# x = data.Education
# y = data.Income
# model = tf.keras.Sequential()#顺序模型
# model.add(tf.keras.layers.Dense(1,input_shape=(1,)))#建立模拟函数,参数：输入数据的维度与输出数据的维度
# model.compile(optimizer='adam',loss='mse')#优化方法计算梯度贴近真实值，损失函数均方差
# print(model.summary())#返回模型的形状
# print(model)
# #history = model.fit(x,y,epochs=5000)
# #plt.scatter(data.Education,data.Income)
# plt.show()

data= pd.read_csv('./tensorflow-data/20190905(16-24) - 标记.csv',encoding='ANSI')

attark = data.iloc[:,11]
count = attark.value_counts()
print(attark.value_counts())
input()
list = []
for index,item in count.items():
    if((index == "-1") | (index == "1.00E+40")):
        continue
    i = 0
    strres = ""
    
    #print("item  : " + str(item))
    for char in index:
        if(i%2!=0):
            int_str = index[i-1:i+1]
            #print(int_str)
            strres += chr(int(int_str,16))
        #input() 
        i += 1   
    list.append(strres)

    print("[number]: " + str(item))
    print("[string]: ")
    print(strres.strip())
    # print("[index ]: ")
    # print(str(index)) 
    print("====================================")       
        

    
    
    
print(type(attark.value_counts()))
