# 数据集

数据集名称|类别|数据集类型
-|-|-
brats 2018|脑肿瘤MRI分割影像|1
mnist|手写识别数据集|2
shakespeare|莎士比亚作品集数据集|3
synthetic|生成式数据集|2

数据集类型说明:

1. 所有的客户端共享一个整体的数据集. 在对应数据集下的`train`和`test`目录中保存的是对应客户端的拥有的数据在整体数据集下索引
2. `train`和`test`目录中保存的是对应客户端的拥有的数据, 这些数据已经经过处理, 可以作为模型的输入数据
3. `train`和`test`目录中保存的是对应客户端的拥有的数据, 这些数据已经经过处理, 但不可以作为模型的输入数据, 需要对应的类进行转换.

## LEAF 格式数据集
LEAF 数据集如果在预处理阶段使用参数 `-t sample` 则会将保存在 `all_data` 数据切分为 train 和 test. 格式为 JSON, 这些 JSON 保存对应客户端端拥有的数据

# 参考
- [关于 Shakespeare 数据集](https://github.com/litian96/FedProx/tree/master/data/shakespeare)
- [关于 Synthetic 数据集](https://github.com/lx10077/fedavgpy)