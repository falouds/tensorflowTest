import tensorflow as tf 
import pandas as pd
import matplotlib.pyplot as plt
import os


head = [
    "@timestamp",
    "_id","_index",
    'alert.action',
    "alert.category",
    "alert.gid",
    "alert.metadata.created_at",
    "alert.metadata.updated_at",
    "alert.rev",
    "alert.severity",
    "alert.signature",
    "alert.signature_id",
    "app_proto",
    "attack_connection.payload.data_hex",
    "attack_connection.payload.length",
    "attack_connection.payload.md5_hash",
    "attack_connection.payload.sha512_hash",
    "attack_connection.protocol",
    "connection.protocol",
    "connection.transport",
    "connection.type,compCS",
    "dest_ip","dest_port",
    "destfile",
    "dist",
    "dns.answers",
    "dns.flags",
    "dns.grouped.A",
    "dns.id",
    "dns.qr",
    "dns.ra",
    "dns.rcode",
    "dns.rd",
    "dns.rrname",
    "dns.rrtype",
    "dns.tx_id",
    "dns.type",
    "dns.version",
    "download_count",
    "download_tries",
    "duplicate",
    "duration",
    "encCS",
    "end_time",
    "event_type",
    "eventid",
    "fileinfo.filename",
    "fileinfo.gaps",
    "fileinfo.magic",
    "fileinfo.md5",
    "fileinfo.size",
    "fileinfo.state",
    "fileinfo.stored",
    "fileinfo.tx_id","flow.bytes_toclient",
    "flow.bytes_toserver",
    "flow.pkts_toclient",
    "flow.pkts_toserver",
    "flow.start",
    "flow_id","geoip.as_org",
    "geoip.asn",
    "geoip.city_name",
    "geoip.continent_code",
    "geoip.country_code2",
    "geoip.country_code3",
    "geoip.country_name",
    "geoip.dma_code",
    "geoip.ip",
    "geoip.latitude",
    "geoip.location",
    "geoip.longitude",
    "geoip.postal_code",
    "geoip.region_code",
    "geoip.region_name",
    "geoip.timezone",
    "host,http.hostname",
    "http.http_content_type",
    "http.http_method",
    "http.length",
    "http.protocol",
    "http.status",
    "http.url",
    "in_iface",
    "input",
    "ip_rep",
    "is_virtual",
    "link",
    "message",
    "metadata"".flowbits",
    "mod",
    "operation_mode",
    "os",
    "outfile",
    "params",
    "password",
    "path",
    "payload",
    "payload_printable",
    "proto",
    "protocol",
    "proxy_connection.local_ip",
    "proxy_connection.local_port",
    "proxy_connection.payload.data_hex",
    "proxy_connection.payload.length",
    "proxy_connection.payload.md5_hash",
    "proxy_connection.payload.sha512_hash",
    "proxy_connection.protocol",
    "proxy_connection.remote_ip",
    "proxy_connection.remote_port",
    "raw_freq",
    "raw_hits",
    "raw_mtu",
    "raw_sig",
    "reason","sensor",
    "session",
    "src_hostname",
    "src_ip",
    "src_port",
    "ssh.client.proto_version",
    "ssh.client.software_version",
    "ssh.server.proto_version",
    "ssh.server.software_version",
    "start_time",
    "stream",
    "subject",
    "t-pot_hostname",
    "t-pot_ip_ext",
    "t-pot_ip_int",
    "tags",
    "timestamp",
    "ttylog",
    "tx_id",
    "type",
    "uptime",
    "url",
    "username",
    "version",
    "label"] 
print("Tensorflow version: {}".format(tf.version))
data= pd.read_csv('./tensorflow-data/20190905(16-24) - 标记.csv',encoding='ANSI')
print(data.head())


resourse = data.iloc[:,2:-1]
label = data.iloc[:,-1]
print(label)
for indexs in resourse.columns:
    list = []
    print(indexs)
    for i, v in resourse[indexs].items():
        if(v in list):
            resourse.loc[i:i,(indexs)] = list.index(v)
        elif(isinstance(v,int) or isinstance(v,float)):
            resourse.loc[i:i,(indexs)] = v
        elif(v.isdigit()):
            resourse.loc[i:i,(indexs)] = float(v)
        else:
            list.append(v)
            resourse.loc[i:i,(indexs)] = list.index(v)
    with open("./tensorflow-model/tpot/sign.txt",'a', encoding='ANSI') as file:
        if(file.writable()):
            file.write(str(list))

resourse["label"] = label
resourse.to_csv("./tensorflow-model/tpot/change.csv",'w',encoding="utf-8")

train_resourse = resourse[0:13000]
train_label = label[0:13000]
test_resourse = resourse[13000:15687]
test_label = label[13000:15687]




model = tf.keras.Sequential()
model.add(tf.keras.layers.Dense(10,input_shape=(78,),activation='relu'))
model.add(tf.keras.layers.Dense(128,activation='relu'))#d过大会导致过拟合，太小会丢失特征
model.add(tf.keras.layers.Dense(128,activation='relu'))
model.add(tf.keras.layers.Dense(128,activation='relu'))
model.add(tf.keras.layers.Dense(6,activation='softmax'))#计算映射到概率

check_path="./tensorflow-model/tpot/ch-1.ckpt"
dir_path = os.path.dirname(check_path)
callback = tf.keras.callbacks.ModelCheckpoint(check_path,save_weights_only=True,verbose=1,period=10)
#第一个参数自动7保存的路径，第二个是保存权重还是模型，第三个参数表示是否显示提示，第四个指步长
print(model.summary())
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['acc']
    )
history = model.fit(
    train_resourse,
    train_label,
    epochs=200,
    callbacks=[callback],
    validation_data=(test_resourse,test_label)
    )
model.save("./tensorflow-model/tpot/modal.h5")#保存模型
model.save_weights("./tensorflow-model/tpot/modal_weight")
#建立新的模型用load_weights,载入权重
plt.plot(history.epoch,history.history.get('loss'),label='loss')
plt.plot(history.epoch,history.history.get('val_loss'),label='val_loss')
plt.show()

plt.plot(history.epoch,history.history.get('acc'),label='acc')
plt.plot(history.epoch,history.history.get('val_acc'),label='val_acc')
plt.show()