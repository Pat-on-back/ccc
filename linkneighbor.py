from typing import Callable, Dict, List, Optional, Tuple, Union
from torch_geometric.data import Data, FeatureStore, GraphStore, HeteroData
from torch_geometric.loader.link_loader import LinkLoader
from torch_geometric.sampler import NegativeSampling, NeighborSampler
from torch_geometric.sampler.base import SubgraphType
from torch_geometric.typing import EdgeType, InputEdges, OptTensor


class LinkNeighborLoader(LinkLoader):
    def __init__(
      self,
      data: Union[Data, HeteroData, Tuple[FeatureStore, GraphStore]],  # 输入的图数据
      num_neighbors: Union[List[int], Dict[EdgeType, List[int]]],  # 每个节点在每次迭代中采样的邻居数量。
      edge_label_index: InputEdges = None,  # 用于训练的边的索引。
      edge_label: OptTensor = None,  # 边的标签
      edge_label_time: OptTensor = None,  # 边的时间戳，用于时序数据采样
      replace: bool = False,  # 是否允许重复采样同一个邻居
      subgraph_type: Union[SubgraphType, str] = 'directional',  # 'directional' 表示仅包含有向边
                                                                # 'bidirectional' 表示双向边
                                                                # 'induced' 表示包含所有采样节点的诱导子图
      disjoint: bool = False,  # 是否每个节点使用独立的子图进行采样
      temporal_strategy: str = 'uniform',  # 时序采样的策略，'uniform' 表示在符合时间约束的邻居中均匀采样
                                           # 'last' 表示采样最后符合条件的邻居
      neg_sampling: Optional[NegativeSampling] = None,  # 用于生成负样本
      neg_sampling_ratio: Optional[Union[int, float]] = None,  # 负采样的比例，已废弃
                                                               # 建议使用 neg_sampling 参数进行配置
      time_attr: Optional[str] = None,  # 图中节点或边的时间戳属性
      weight_attr: Optional[str] = None,  # 在采样时会根据边权重的大小决定采样概率
      transform: Optional[Callable] = None,  # 数据变换函数，输入采样的批次数据，输出变换后的数据
      transform_sampler_output: Optional[Callable] = None,  # 对采样输出的变换函数
      is_sorted: bool = False,  
      filter_per_worker: Optional[bool] = None,  
      neighbor_sampler: Optional[NeighborSampler] = None, 
      directed: bool = True, 
      **kwargs, 
  ):

        # 检查时间戳相关的参数是否冲突
        if (edge_label_time is not None) != (time_attr is not None):
            raise ValueError(
                f"Received conflicting 'edge_label_time' and 'time_attr' "
                f"arguments: 'edge_label_time' is "
                f"{'set' if edge_label_time is not None else 'not set'} "
                f"while 'time_attr' is "
                f"{'set' if time_attr is not None else 'not set'}. "
                f"Both arguments must be provided for temporal sampling.")

        # 如果没有传入 neighbor_sampler，则创建一个新的
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

        # 调用父类LinkLoader的构造函数进行初始化
        super().__init__(
            data=data,
            link_sampler=neighbor_sampler,  # 使用上述的neighbor_sampler进行链接采样
            edge_label_index=edge_label_index,  # 边标签索引
            edge_label=edge_label,  # 边标签
            edge_label_time=edge_label_time,  # 边标签时间
            neg_sampling=neg_sampling,  # 负采样策略
            neg_sampling_ratio=neg_sampling_ratio,  # 负采样比例
            transform=transform,  # 数据变换函数
            transform_sampler_output=transform_sampler_output,  # 变换采样输出
            filter_per_worker=filter_per_worker, 
            **kwargs,
        )
