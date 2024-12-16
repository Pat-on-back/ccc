import copy
import os
import os.path as osp
import sys
from dataclasses import dataclass
from typing import List, Literal, Optional

import torch
import torch.utils.data
from torch import Tensor

import torch_geometric.typing
from torch_geometric.data import Data
from torch_geometric.index import index2ptr, ptr2index
from torch_geometric.io import fs
from torch_geometric.typing import pyg_lib
from torch_geometric.utils import index_sort, narrow, select, sort_edge_index
from torch_geometric.utils.map import map_index


@dataclass
class Partition:
    indptr: Tensor  # CSR格式的指针数组
    index: Tensor  # 节点索引
    partptr: Tensor  # 每个分区的节点指针
    node_perm: Tensor  # 节点排序
    edge_perm: Tensor  # 边排序
    sparse_format: Literal['csr', 'csc']  # 稀疏矩阵的格式，CSR或CSC


class ClusterData(torch.utils.data.Dataset):
    r"""将图数据对象划分成多个子图，这一方法受到
    `"Cluster-GCN: An Efficient Algorithm for Training Deep and Large Graph Convolutional Networks"
    <https://arxiv.org/abs/1905.07953>`_ 论文启发。

    .. note::
        底层的 METIS 算法要求输入图为无向图。

    参数:
        data (torch_geometric.data.Data): 图数据对象。
        num_parts (int): 划分的子图数量。
        recursive (bool, 可选): 如果为 True，使用多层递归二分法进行划分，否则使用多层k-way划分。
        save_dir (str, 可选): 如果设置，会将划分后的数据保存在该目录中以便下次复用。
        filename (str, 可选): 保存的划分文件的名称。
        log (bool, 可选): 是否打印进度信息。默认为 True。
        keep_inter_cluster_edges (bool, 可选): 是否保留跨集群边连接。默认为 False。
        sparse_format (str, 可选): 用于计算划分的稀疏矩阵格式，默认为 "csr"。
    """
    def __init__(
        self,
        data,
        num_parts: int,
        recursive: bool = False,
        save_dir: Optional[str] = None,
        filename: Optional[str] = None,
        log: bool = True,
        keep_inter_cluster_edges: bool = False,
        sparse_format: Literal['csr', 'csc'] = 'csr',
    ):
        # 确保图数据中包含边信息
        assert data.edge_index is not None
        assert sparse_format in ['csr', 'csc']

        self.num_parts = num_parts
        self.recursive = recursive
        self.keep_inter_cluster_edges = keep_inter_cluster_edges
        self.sparse_format = sparse_format

        # 设置文件路径，用于保存划分结果
        recursive_str = '_recursive' if recursive else ''
        root_dir = osp.join(save_dir or '', f'part_{num_parts}{recursive_str}')
        path = osp.join(root_dir, filename or 'metis.pt')

        # 如果已经存在划分结果文件，则加载
        if save_dir is not None and osp.exists(path):
            self.partition = fs.torch_load(path)
        else:
            if log:  # 如果需要日志，则输出划分进度
                print('正在计算 METIS 划分...', file=sys.stderr)

            # 调用 METIS 算法进行图划分
            cluster = self._metis(data.edge_index, data.num_nodes)
            self.partition = self._partition(data.edge_index, cluster)

            # 如果设置了保存目录，则保存划分结果
            if save_dir is not None:
                os.makedirs(root_dir, exist_ok=True)
                torch.save(self.partition, path)

            if log:  # 如果需要日志，则输出划分完成信息
                print('完成!', file=sys.stderr)

        # 根据划分结果对数据进行重新排列
        self.data = self._permute_data(data, self.partition)

    def _metis(self, edge_index: Tensor, num_nodes: int) -> Tensor:
        # 通过 METIS 算法计算节点级别的划分结果
        if self.sparse_format == 'csr':
            # 计算 CSR 格式的表示
            row, index = sort_edge_index(edge_index, num_nodes=num_nodes)
            indptr = index2ptr(row, size=num_nodes)
        else:
            # 计算 CSC 格式的表示
            index, col = sort_edge_index(edge_index, num_nodes=num_nodes, sort_by_row=False)
            indptr = index2ptr(col, size=num_nodes)

        cluster: Optional[Tensor] = None

        # 尝试使用 torch_sparse 库进行图划分
        if torch_geometric.typing.WITH_TORCH_SPARSE:
            try:
                cluster = torch.ops.torch_sparse.partition(
                    indptr.cpu(),
                    index.cpu(),
                    None,
                    self.num_parts,
                    self.recursive,
                ).to(edge_index.device)
            except (AttributeError, RuntimeError):
                pass

        # 如果没有成功，尝试使用 pyg_lib 中的 METIS 算法
        if cluster is None and torch_geometric.typing.WITH_METIS:
            cluster = pyg_lib.partition.metis(
                indptr.cpu(),
                index.cpu(),
                self.num_parts,
                recursive=self.recursive,
            ).to(edge_index.device)

        # 如果仍然无法使用 METIS 划分，则抛出错误
        if cluster is None:
            raise ImportError(f"'{self.__class__.__name__}' 需要 'pyg-lib' 或 'torch-sparse'")

        return cluster

    def _partition(self, edge_index: Tensor, cluster: Tensor) -> Partition:
        # 根据节点级和边级的排列，计算并返回新的划分信息

        # 对 cluster 进行排序，并计算每个子图的边界 `partptr`
        cluster, node_perm = index_sort(cluster, max_value=self.num_parts)
        partptr = index2ptr(cluster, size=self.num_parts)

        # 根据节点排列对 `edge_index` 进行重排列
        edge_perm = torch.arange(edge_index.size(1), device=edge_index.device)
        arange = torch.empty_like(node_perm)
        arange[node_perm] = torch.arange(cluster.numel(),
                                         device=cluster.device)
        edge_index = arange[edge_index]

        # 计算最终的 CSR 格式的表示
        (row, col), edge_perm = sort_edge_index(
            edge_index,
            edge_attr=edge_perm,
            num_nodes=cluster.numel(),
            sort_by_row=self.sparse_format == 'csr',
        )
        if self.sparse_format == 'csr':
            indptr, index = index2ptr(row, size=cluster.numel()), col
        else:
            indptr, index = index2ptr(col, size=cluster.numel()), row

        return Partition(indptr, index, partptr, node_perm, edge_perm,
                         self.sparse_format)

    def _permute_data(self, data: Data, partition: Partition) -> Data:
        # 根据划分结果对节点和边的属性进行排列
        out = copy.copy(data)
        for key, value in data.items():
            if key == 'edge_index':
                continue
            elif data.is_node_attr(key):
                cat_dim = data.__cat_dim__(key, value)
                out[key] = select(value, partition.node_perm, dim=cat_dim)
            elif data.is_edge_attr(key):
                cat_dim = data.__cat_dim__(key, value)
                out[key] = select(value, partition.edge_perm, dim=cat_dim)
        out.edge_index = None

        return out

    def __len__(self) -> int:
        # 返回划分的子图数量
        return self.partition.partptr.numel() - 1

    def __getitem__(self, idx: int) -> Data:
        # 获取第 idx 个子图的数据
        node_start = int(self.partition.partptr[idx])  
        node_end = int(self.partition.partptr[idx + 1]) 
        node_length = node_end - node_start  # 计算当前子图包含的节点数
    
        # 获取当前子图的边指针（indptr），用于获取子图的边
        indptr = self.partition.indptr[node_start:node_end + 1]
        edge_start = int(indptr[0])  
        edge_end = int(indptr[-1]) 
        edge_length = edge_end - edge_start  # 计算当前子图包含的边数
        indptr = indptr - edge_start  
    
        # 根据稀疏矩阵格式（CSR 或 CSC）处理边信息
        if self.sparse_format == 'csr':
            # 如果是 CSR 格式，将 indptr 转换为行索引
            row = ptr2index(indptr)
            col = self.partition.index[edge_start:edge_end]  
            if not self.keep_inter_cluster_edges:
                # 如果不保留集群间的边，过滤掉不属于当前子图的边
                edge_mask = (col >= node_start) & (col < node_end) # 找出当前子图的边
                row = row[edge_mask]  # 保留有效的行索引
                col = col[edge_mask] - node_start  # 保留有效的列索引并重新映射节点编号
        else:
            # 如果是 CSC 格式，将 indptr 转换为列索引
            col = ptr2index(indptr)
            row = self.partition.index[edge_start:edge_end]  
            if not self.keep_inter_cluster_edges:
                # 如果不保留集群间的边，过滤掉不属于当前子图的边
                edge_mask = (row >= node_start) & (row < node_end) 
                col = col[edge_mask]  # 保留有效的列索引
                row = row[edge_mask] - node_start  # 保留有效的行索引并重新映射节点编号
    
        out = copy.copy(self.data)
    
        # 对节点和边的属性进行相应的切片处理
        for key, value in self.data.items():
            if key == 'num_nodes':
                out.num_nodes = node_length  # 设置子图的节点数
            elif self.data.is_node_attr(key):
                # 如果是节点属性，进行节点维度的切片操作
                cat_dim = self.data.__cat_dim__(key, value)  # 获取该属性的维度
                out[key] = narrow(value, cat_dim, node_start, node_length)  # 截取节点数据
            elif self.data.is_edge_attr(key):
                # 如果是边属性，进行边维度的切片操作
                cat_dim = self.data.__cat_dim__(key, value)  # 获取该属性的维度
                out[key] = narrow(value, cat_dim, edge_start, edge_length)  # 截取边数据
                if not self.keep_inter_cluster_edges:
                    # 如果不保留集群间的边，过滤掉不属于当前子图的边
                    out[key] = out[key][edge_mask]
    
        # 构造并更新边的索引
        out.edge_index = torch.stack([row, col], dim=0) 
    
        return out  # 返回当前子图的数据对象


def __repr__(self) -> str:
    # 返回类的字符串表示，便于打印和调试
    return f'{self.__class__.__name__}({self.num_parts})'  # 打印该对象的类名和划分数



class ClusterLoader(torch.utils.data.DataLoader):
    """
    基于 'Cluster-GCN' 论文提出的图划分数据加载器，将多个划分后的子图及其集群间的边连接组合成一个小批量。
    请配合 `ClusterData` 和 `ClusterLoader` 一起使用，以形成小批量。
    """
    
    def __init__(self, cluster_data, **kwargs):
        """
        初始化 ClusterLoader。
        
        参数:
        - cluster_data (ClusterData): 已经划分好的数据对象。
        - **kwargs: 传递给 DataLoader 的其他参数（如 batch_size、shuffle、drop_last 等）。
        """
        self.cluster_data = cluster_data  # 存储传入的 ClusterData 对象
        
        # 创建迭代器，方便从 cluster_data 中获取数据
        iterator = range(len(cluster_data))  
        
        # 调用父类（DataLoader）的初始化方法，设置迭代器、批处理函数等
        super().__init__(iterator, collate_fn=self._collate, **kwargs)

    def _collate(self, batch: List[int]) -> Data:
        """
        处理批次数据，将当前批次的子图数据拼接起来，并计算新的边连接。
        
        参数:
        - batch (List[int]): 当前小批量中包含的子图索引。
        
        返回:
        - Data: 组合成的新图数据对象，包含拼接后的节点、边信息。
        """
        
        # 如果 batch 不是 Tensor 类型，则转换为 Tensor 类型
        if not isinstance(batch, torch.Tensor):
            batch = torch.tensor(batch)

        # 从 cluster_data 中获取全局的节点和边信息
        global_indptr = self.cluster_data.partition.indptr  # 每个节点的边的起始位置
        global_index = self.cluster_data.partition.index  # 所有边的目标节点索引
        
        # 获取当前批次中每个子图的节点和边的起始与结束位置
        node_start = self.cluster_data.partition.partptr[batch]  # 节点的起始位置
        node_end = self.cluster_data.partition.partptr[batch + 1]  # 节点的结束位置
        edge_start = global_indptr[node_start]  # 边的起始位置
        edge_end = global_indptr[node_end]  # 边的结束位置
        
        # 存储拼接后的行和列索引，以及节点信息
        rows, cols, nodes, cumsum = [], [], [], 0

        # 遍历当前批次中的每个子图
        for i in range(batch.numel()):
            nodes.append(torch.arange(node_start[i], node_end[i]))  # 获取当前子图的节点索引
            # 获取当前子图的边的偏移量
            indptr = global_indptr[node_start[i]:node_end[i] + 1]
            indptr = indptr - edge_start[i]  # 归一化偏移量
            
            # 根据稀疏格式（CSR 或 CSC）处理边的索引
            if self.cluster_data.partition.sparse_format == 'csr':
                row = ptr2index(indptr) + cumsum  # 获取行索引，并加上累计的边数偏移
                col = global_index[edge_start[i]:edge_end[i]]  # 获取列索引
            else:
                col = ptr2index(indptr) + cumsum  # 获取列索引，并加上累计的边数偏移
                row = global_index[edge_start[i]:edge_end[i]]  # 获取行索引

            rows.append(row)  # 添加行索引
            cols.append(col)  # 添加列索引
            cumsum += indptr.numel() - 1  # 更新累计的边数（用于计算新的边索引）

        # 拼接所有节点、行和列索引
        node = torch.cat(nodes, dim=0)  # 拼接节点
        row = torch.cat(rows, dim=0)  # 拼接行索引
        col = torch.cat(cols, dim=0)  # 拼接列索引

        # 仅保留连接同一子图的边
        if self.cluster_data.partition.sparse_format == 'csr':
            col, edge_mask = map_index(col, node)  # 获取同一子图内的边
            row = row[edge_mask]  # 过滤出有效的行索引
        else:
            row, edge_mask = map_index(row, node)  # 获取同一子图内的边
            col = col[edge_mask]  # 过滤出有效的列索引

        # 创建一个新的图数据对象，包含子图的数据
        out = copy.copy(self.cluster_data.data)

        # 根据子图的偏移量对节点和边属性进行切片
        for key, value in self.cluster_data.data.items():
            if key == 'num_nodes':
                out.num_nodes = cumsum  # 设置节点数
            elif self.cluster_data.data.is_node_attr(key):
                # 如果是节点属性，进行节点维度的切片操作
                cat_dim = self.cluster_data.data.__cat_dim__(key, value)
                out[key] = torch.cat([narrow(out[key], cat_dim, s, e - s) for s, e in zip(node_start, node_end)], dim=cat_dim)
            elif self.cluster_data.data.is_edge_attr(key):
                # 如果是边属性，进行边维度的切片操作
                cat_dim = self.cluster_data.data.__cat_dim__(key, value)
                value = torch.cat([narrow(out[key], cat_dim, s, e - s) for s, e in zip(edge_start, edge_end)], dim=cat_dim)
                out[key] = select(value, edge_mask, dim=cat_dim)  # 只保留有效的边属性

        # 更新边的索引，并返回最终的数据对象
        out.edge_index = torch.stack([row, col], dim=0)  # 拼接成边的索引
        return out  # 返回当前批次的图数据对象
