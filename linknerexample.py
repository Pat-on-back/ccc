from torch_geometric.datasets import KarateClub
from torch_geometric.loader import LinkNeighborLoader
import matplotlib.pyplot as plt
import networkx as nx
from torch_geometric.utils import to_networkx
import torch
# 加载空手道数据集
dataset = KarateClub()
data = dataset[0]

# 可视化图
def visualize_graph(data, fname):
    # 将 PyG 数据对象转换为 NetworkX 图，
    G = to_networkx(data, to_undirected=True, node_attrs=['x'])  
    
    # 获取布局
    plt.figure(figsize=(8, 6))
    pos = nx.spring_layout(G)  # 使用弹簧布局
    nx.draw(G, pos, with_labels=True, node_size=500, node_color="skyblue", 
            font_size=10, font_weight="bold", edge_color="gray")
    plt.title(f"Visualized {fname}")
    plt.savefig(f'{fname}.png')
    plt.show()

# 设置link邻居采样器
loader = LinkNeighborLoader(
    data,
    num_neighbors=[10] * 2,  # 每个节点采样10个邻居，2轮采样
    edge_label_index=data.edge_index,
    edge_label=torch.ones(data.edge_index.size(1))
)

# 获取一个批次的数据
sampled_data = next(iter(loader))
# 可视化原始图
visualize_graph(data, 'original')
# 可视化采样的数据
visualize_graph(sampled_data, 'sampled')
print(sampled_data)
