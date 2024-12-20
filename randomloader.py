import math
from typing import Union
import torch
from torch import Tensor
from torch_geometric.data import Data, HeteroData
from torch_geometric.data.hetero_data import to_homogeneous_edge_index

class RandomNodeLoader(torch.utils.data.DataLoader):
    r"""一个数据加载器，它从图中随机采样节点，并返回这些节点构成的诱导子图。   
    参数:
        data (torch_geometric.data.Data 或 torch_geometric.data.HeteroData):
            传入的图数据，可以是 `Data` 或 `HeteroData` 对象。
        num_parts (int): 划分的部分数量，用于控制每个批次的节点数量。
        **kwargs (optional): `torch.utils.data.DataLoader` 的其他可选参数。
    """
    
    def __init__(
        self,
        data: Union[Data, HeteroData],
        num_parts: int,
        **kwargs,
    ):
        """
        初始化函数，接收图数据对象和划分部分数。

        参数:
            data: 输入的图数据，可以是 `Data` 或 `HeteroData`。
            num_parts: 划分部分数，决定每个批次的大小。
            **kwargs: 传递给父类 `DataLoader` 的其他参数。
        """
        self.data = data  # 存储输入的图数据
        self.num_parts = num_parts  # 存储划分的部分数

        # 如果数据是 HeteroData（异构图），则需要将其转化为同质图
        if isinstance(data, HeteroData):
            # 将异构图的边信息转化为同质图（所有节点和边合并为一个图）
            edge_index, node_dict, edge_dict = to_homogeneous_edge_index(data)
            self.node_dict, self.edge_dict = node_dict, edge_dict
        else:
            # 如果是普通的同质图，直接获取边索引
            edge_index = data.edge_index

        self.edge_index = edge_index  # 存储图的边索引
        self.num_nodes = data.num_nodes  # 存储图中节点的总数

        # 调用父类 DataLoader 的初始化函数，生成批次
        super().__init__(
            range(self.num_nodes),  # 数据集的索引为节点的编号
            batch_size=math.ceil(self.num_nodes / num_parts), 
            collate_fn=self.collate_fn,  # 自定义的合并函数
            **kwargs,  # 传递其他参数
        )

    def collate_fn(self, index):
        """
        自定义的合并函数，根据索引生成诱导子图。

        参数:
            index: 当前批次的节点索引列表。

        返回:
            子图：基于当前索引生成的诱导子图。
        """
        if not isinstance(index, Tensor):
            index = torch.tensor(index)  # 如果索引不是 Tensor 类型，转为 Tensor

        # 如果输入的图数据是 Data（同质图）
        if isinstance(self.data, Data):
            # 调用 Data 的 `subgraph` 方法，返回包含这些节点的诱导子图
            return self.data.subgraph(index)

        # 如果输入的是 HeteroData（异构图）
        elif isinstance(self.data, HeteroData):
            # 根据每种类型的节点，更新每个节点的索引
            node_dict = {
                key: index[(index >= start) & (index < end)] - start  # 为每种类型的节点更新索引
                for key, (start, end) in self.node_dict.items()  # 根据节点的起始和结束位置更新索引
            }
            # 返回异构图的诱导子图
            return self.data.subgraph(node_dict)
