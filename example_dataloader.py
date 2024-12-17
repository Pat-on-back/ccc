import torch
from torch_geometric.data import Data, DataLoader

# 创建一些简单的示例图数据
graphs = []
for i in range(10):  # 假设我们有10个图
    x = torch.randn(100, 16)  # 100个节点，每个节点16维特征
    edge_index = torch.randint(0, 100, (2, 300))  # 300条边
    y = torch.randint(0, 2, (1,))  # 每个图有一个标签

    data = Data(x=x, edge_index=edge_index, y=y)
    graphs.append(data)

# 使用 DataLoader 来分批加载图数据
loader = DataLoader(graphs, batch_size=2, shuffle=True)

for batch in loader:
    print(batch)
    print("Batch size:", batch.num_graphs)  

