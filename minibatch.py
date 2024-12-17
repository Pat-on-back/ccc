from typing import Any, Callable, Iterator, List, Optional, Tuple, Union
import torch
from torch import Tensor
from torch_geometric.data import Data, FeatureStore, GraphStore, HeteroData
from torch_geometric.loader.base import DataLoaderIterator
from torch_geometric.loader.mixin import (
    AffinityMixin,
    LogMemoryMixin,
    MultithreadingMixin,
)
from torch_geometric.loader.utils import (
    filter_custom_hetero_store,
    filter_custom_store,
    filter_data,
    filter_hetero_data,
    get_input_nodes,
    infer_filter_per_worker,
)
from torch_geometric.sampler import (
    BaseSampler,
    HeteroSamplerOutput,
    NodeSamplerInput,
    SamplerOutput,
)
from torch_geometric.typing import InputNodes, OptTensor


class NodeLoader(
        torch.utils.data.DataLoader,  
        AffinityMixin,  
        MultithreadingMixin,  
        LogMemoryMixin,  
):
    def __init__(
        self,
        data: Union[Data, HeteroData, Tuple[FeatureStore, GraphStore]],  # 输入数据
        node_sampler: BaseSampler,  # 采样器，必须实现`sample_from_nodes`方法
        input_nodes: InputNodes = None,  # 采样的起始节点，默认情况下所有节点都参与
        input_time: OptTensor = None,  # 可选，输入节点的时间戳（如果有）
        transform: Optional[Callable] = None,  # 采样后数据的转换函数
        transform_sampler_output: Optional[Callable] = None,  # 采样器输出的转换函数
        filter_per_worker: Optional[bool] = None,  # 是否在每个工作进程中过滤数据
        custom_cls: Optional[HeteroData] = None,  # 在分布式环境下，返回的自定义HeteroData类
        input_id: OptTensor = None,  # 可选，输入节点的ID
        **kwargs,  # 其他DataLoader参数，如batch_size，shuffle等
    ):
        # 根据输入数据自动推断是否需要在每个工作进程中过滤数据
        if filter_per_worker is None:
            filter_per_worker = infer_filter_per_worker(data)

        self.data = data  # 保存输入数据
        self.node_sampler = node_sampler  # 保存节点采样器
        self.input_nodes = input_nodes  # 保存输入节点
        self.input_time = input_time  # 保存输入时间
        self.transform = transform  # 保存数据转换函数
        self.transform_sampler_output = transform_sampler_output  # 保存采样器输出转换函数
        self.filter_per_worker = filter_per_worker  # 保存是否在每个工作进程中过滤数据
        self.custom_cls = custom_cls  # 保存自定义HeteroData类
        self.input_id = input_id  # 保存输入节点ID

        kwargs.pop('dataset', None)  # 移除dataset参数
        kwargs.pop('collate_fn', None)  # 移除collate_fn参数

        # 获取输入节点类型（对于同质图，返回None）
        input_type, input_nodes, input_id = get_input_nodes(
            data, input_nodes, input_id)

        # 创建NodeSamplerInput实例
        self.input_data = NodeSamplerInput(
            input_id=input_id,
            node=input_nodes,
            time=input_time,
            input_type=input_type,
        )

        # 创建迭代器
        iterator = range(input_nodes.size(0))
        super().__init__(iterator, collate_fn=self.collate_fn, **kwargs)

    def __call__(self, index: Union[Tensor, List[int]]) -> Union[Data, HeteroData]:
        r"""从输入节点批次中采样子图。"""
        out = self.collate_fn(index)  # 通过collate_fn采样子图
        if not self.filter_per_worker:
            out = self.filter_fn(out)  # 如果不在工作进程中过滤数据，
                                       # 则在主进程中执行过滤
        return out

    def collate_fn(self, index: Union[Tensor, List[int]]) -> Any:
        r"""根据输入索引采样子图。"""
        input_data: NodeSamplerInput = self.input_data[index]  # 获取输入数据

        # 使用采样器从输入节点中采样
        out = self.node_sampler.sample_from_nodes(input_data)

        if self.filter_per_worker:  # 如果在工作进程中过滤数据，执行过滤
            out = self.filter_fn(out)

        return out

    def filter_fn(self, out: Union[SamplerOutput, HeteroSamplerOutput]) -> Union[Data, HeteroData]:
        r"""将采样到的节点与其特征进行结合，返回Data或HeteroData对象。"""
        if self.transform_sampler_output:
            out = self.transform_sampler_output(out)  # 如果有自定义的采样器输出转换函数，进行转换

        if isinstance(out, SamplerOutput):  # 如果输出是SamplerOutput类型
            if isinstance(self.data, Data):
                data = filter_data(  # 过滤数据并返回Data对象
                    self.data, out.node, out.row, out.col, out.edge,
                    self.node_sampler.edge_permutation)

            else:  # 如果数据是FeatureStore和GraphStore的元组
                # 检测是否在分布式环境下
                if (self.node_sampler.__class__.__name__ == 'DistNeighborSampler'):
                    edge_index = torch.stack([out.row, out.col])
                    data = Data(edge_index=edge_index)
                    data.x = out.metadata[-3]  # 设置节点特征
                    data.y = out.metadata[-2]  # 设置节点标签
                    data.edge_attr = out.metadata[-1]  # 设置边属性
                else:
                    data = filter_custom_store(  # 过滤并返回自定义的Store对象
                        *self.data, out.node, out.row, out.col, out.edge,
                        self.custom_cls)

            if 'n_id' not in data:
                data.n_id = out.node
            if out.edge is not None and 'e_id' not in data:
                edge = out.edge.to(torch.long)
                perm = self.node_sampler.edge_permutation
                data.e_id = perm[edge] if perm is not None else edge

            data.batch = out.batch  # 设置批次信息
            data.num_sampled_nodes = out.num_sampled_nodes  # 设置采样节点数
            data.num_sampled_edges = out.num_sampled_edges  # 设置采样边数

            if out.orig_row is not None and out.orig_col is not None:
                data._orig_edge_index = torch.stack([out.orig_row, out.orig_col], dim=0) 

            data.input_id = out.metadata[0]  # 设置输入ID
            data.seed_time = out.metadata[1]  # 设置时间戳
            data.batch_size = out.metadata[0].size(0)  # 设置批次大小

        elif isinstance(out, HeteroSamplerOutput):  
            if isinstance(self.data, HeteroData):
                data = filter_hetero_data(  # 过滤并返回HeteroData对象
                    self.data, out.node, out.row, out.col, out.edge,
                    self.node_sampler.edge_permutation)

            else:  # 如果数据是FeatureStore和GraphStore的元组
                if (self.node_sampler.__class__.__name__ == 'DistNeighborSampler'):
                    import torch_geometric.distributed as dist

                    data = dist.utils.filter_dist_store(  # 过滤分布式Store对象
                        *self.data, out.node, out.row, out.col, out.edge,
                        self.custom_cls, out.metadata,
                        self.input_data.input_type)
                else:
                    data = filter_custom_hetero_store(  # 过滤并返回自定义的HeteroStore对象
                        *self.data, out.node, out.row, out.col, out.edge,
                        self.custom_cls)

            # 如果采样结果中的节点 (out.node) 不包含 'n_id' 字段，则为每个节点添加 'n_id'
            for key, node in out.node.items():
                if 'n_id' not in data[key]:  
                    data[key].n_id = node  # 设置 'n_id' 为采样节点的 ID
            
            # 对采样结果中的边 (out.edge) 进行处理，确保边数据包含 'e_id'
            for key, edge in (out.edge or {}).items():  
                if edge is not None and 'e_id' not in data[key]:  
                    edge = edge.to(torch.long)  
            
                    # 如果采样器提供了边的重新排列信息（edge_permutation），则应用该重新排列
                    perm = self.node_sampler.edge_permutation
                    if perm is not None and perm.get(key, None) is not None:
                        edge = perm[key][edge]  
            
                    data[key].e_id = edge  # 设置 'e_id' 为采样边的 ID
            
            # 设置 'batch'、'num_sampled_nodes' 和 'num_sampled_edges' 信息
            data.set_value_dict('batch', out.batch)  # 设置批次信息
            data.set_value_dict('num_sampled_nodes', out.num_sampled_nodes) 
            data.set_value_dict('num_sampled_edges', out.num_sampled_edges)  
            
            # 如果采样结果包含原始的边索引 (orig_row 和 orig_col)，则将其添加到数据中
            if out.orig_row is not None and out.orig_col is not None:
                for key in out.orig_row.keys():  #
                    data[key]._orig_edge_index = torch.stack([out.orig_row[key], out.orig_col[key]], dim=0)  
    
            
            # 获取输入数据类型
            input_type = self.input_data.input_type  
            
            # 将采样结果的元数据（如输入ID和时间）添加到数据中
            data[input_type].input_id = out.metadata[0]  # 设置输入节点 ID
            data[input_type].seed_time = out.metadata[1]  # 设置种子时间
            data[input_type].batch_size = out.metadata[0].size(0)  # 设置批次大小
            
        # 如果数据类型不符合预期，抛出类型错误
        else:
            raise TypeError(f"'{self.__class__.__name__}' found invalid type: '{type(out)}'")
        
        # 如果有自定义的转换函数（transform），则返回转换后的数据
        # 否则返回原始的数据
        return data if self.transform is None else self.transform(data)


    def __repr__(self) -> str:
        return f'{self.__class__.__name__}()'
