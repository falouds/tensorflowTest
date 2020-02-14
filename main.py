from sklearn.datasets.california_housing import fetch_california_housing
from sklearn import tree
hoseing = fetch_california_housing()
print(type(hoseing))
print(hoseing["data"])
print(type(hoseing["data"]))

dtr = tree.DecisionTreeRegressor(max_depth=2)
dtr.fit(hoseing["data"][:,[6,7]],hoseing["target"])

dot_data = \
    tree.export_graphviz(
        dtr,
        out_file = None,
        feature_names=hoseing["feature_names"][6:8],
        filled = True,
        impurity = False,
        rounded = True
    )
import os     
os.environ["PATH"] += os.pathsep + 'D:/Program Files (x86)/Graphviz2.38/bin/'
import pydotplus
graph = pydotplus.graph_from_dot_data(dot_data)
graph.get_nodes()[7].set_fillcolor("#FFF2DD")
graph.write_png("./res.png")



from sklearn.model_selection import train_test_split#分割训练集

data_train,data_test,target_train,target_test = \
    train_test_split(hoseing["data"],hoseing["target"],test_size=0.1,random_state = 42)
dtr = tree.DecisionTreeRegressor(random_state = 42)
dtr.fit(data_train,target_train)
print(dtr.score(data_test,target_test))


from sklearn.ensemble import RandomForestRegressor#系统自己调整参数
rfr = RandomForestRegressor(random_state = 42)
rfr.fit(data_train,target_train)
print(rfr.score(data_test,target_test))