from typing import Callable, Dict, List, Optional, Tuple, Union
# 导入类型定义

from torch_geometric.data import Data, FeatureStore, GraphStore, HeteroData
from torch_geometric.loader.node_loader import NodeLoader
from torch_geometric.sampler import NeighborSampler
from torch_geometric.sampler.base import SubgraphType
from torch_geometric.typing import EdgeType, InputNodes, OptTensor
# 导入 PyTorch Geometric 中的相关模块和类型

class NeighborLoader(NodeLoader):
    r"""邻居采样器，用于大规模图的图神经网络 (GNN) 训练，进行小批量训练
    基于论文 `"Inductive Representation Learning on Large Graphs"
    <https://arxiv.org/abs/1706.02216>`_ 中提出的邻居采样方法。

    这个采样器逐步为每个节点采样邻居，以避免内存消耗过大，适用于大规模图的训练。

    :param data: 输入的数据，支持 `Data`、`HeteroData` 或 (`FeatureStore`, `GraphStore`) 类型。
    :param num_neighbors: 每一轮采样时，每个节点的邻居数量
    :param input_nodes: 初始的输入节点，指定从哪些节点开始进行邻居采样。
    :param input_time: 可选，指定输入节点的时间戳。
    :param replace: 是否允许重复采样，默认为 `False`。
    :param subgraph_type: 选择返回子图的类型，默认为 `'directional'`。
    :param disjoint: 是否使用不相交的子图，默认为 `False`。
    :param temporal_strategy: 时间采样策略，默认为 `'uniform'`。
    :param time_attr: 可选，指定时间属性，用于时间约束的采样。
    :param weight_attr: 可选，指定图中边的权重属性。
    :param transform: 可选，数据转换函数。
    :param transform_sampler_output: 可选，采样输出的转换函数。
    :param is_sorted: 是否假设边按列排序，默认为 `False`。
    :param filter_per_worker: 是否在每个工作线程中过滤数据，默认为 `None`。
    :param neighbor_sampler: 可选，指定自定义的邻居采样器。
    :param directed: 是否为有向图，已废弃。
    :param kwargs: 其他参数，例如 `batch_size`、`shuffle`、`drop_last` 等。
    """

    def __init__(
        self,
        data: Union[Data, HeteroData, Tuple[FeatureStore, GraphStore]],  
        num_neighbors: Union[List[int], Dict[EdgeType, List[int]]],  
        input_nodes: InputNodes = None,  
        input_time: OptTensor = None, 
        replace: bool = False,  
        subgraph_type: Union[SubgraphType, str] = 'directional',  
        disjoint: bool = False, 
        temporal_strategy: str = 'uniform',  
        time_attr: Optional[str] = None,  
        weight_attr: Optional[str] = None,  
        transform: Optional[Callable] = None,  
        transform_sampler_output: Optional[Callable] = None, 
        is_sorted: bool = False,  # 是否假设边已按列排序
        filter_per_worker: Optional[bool] = None,  
        neighbor_sampler: Optional[NeighborSampler] = None,  
        directed: bool = True,  # 是否为有向图，已废弃
        **kwargs,  
    ):
        
        if input_time is not None and time_attr is None:
            raise ValueError("Received conflicting 'input_time' and "
                             "'time_attr' arguments: 'input_time' is set "
                             "while 'time_attr' is not set.")

        # 如果没有提供自定义的邻居采样器，则使用默认的 NeighborSampler
        if neighbor_sampler is None:
            neighbor_sampler = NeighborSampler(
                data,
                num_neighbors=num_neighbors,
                replace=replace,
                subgraph_type=subgraph_type,
                disjoint=disjoint,
                temporal_strategy=temporal_strategy,
                time_attr=time_attr,
                weight_attr=weight_attr,
                is_sorted=is_sorted,
                share_memory=kwargs.get('num_workers', 0) > 0,
                directed=directed,
            )

        # 调用父类构造函数，初始化 NodeLoader
        super().__init__(
            data=data,  # 输入的数据
            node_sampler=neighbor_sampler,  # 使用的邻居采样器
            input_nodes=input_nodes,  # 初始节点
            input_time=input_time,  # 时间戳
            transform=transform,  # 数据转换函数
            transform_sampler_output=transform_sampler_output,  # 采样器输出转换函数
            filter_per_worker=filter_per_worker,  # 是否每个工作线程过滤数据
            **kwargs,  # 其他额外参数
        )
