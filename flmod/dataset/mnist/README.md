# MNIST 的数据

不同的算法要求的数据格式不同. MNIST的原始数据保存在本目录下的 `data` 目录下的 `MNIST` 文件夹. Torchvision 将会自动下载

## FedProx 相关

FedProx的原始文章参考: [Federated Optimization in Heterogeneous Networks](https://arxiv.org/abs/1812.06127)

`generate_random_user1000_niid_alldata.py`: 包含了1000个联邦学习的客户端, 主要的参数是是否进行扁平化(flatten)和均匀化(normalize)

## FedAvg(Li Xiang 等人)

参考: [On the Convergence of FedAvg on Non-IID Data](https://arxiv.org/abs/1907.02189). 他们的工作包含了两种划分模式. 每种格式都是100个客户端. 

- `generate_equal.py`: 每个客户端的样本数量相同
- `generate_random_niid.py`: 和 `generate_equal.py` 相反