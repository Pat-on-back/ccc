from abc import abstractmethod
from typing import Any, Callable, Dict, Optional, Protocol, Tuple, Union

from torch_geometric.data import Data, FeatureStore, HeteroData
from torch_geometric.sampler import HeteroSamplerOutput, SamplerOutput
from torch_geometric.typing import InputEdges, InputNodes


# 定义 RAGFeatureStore 协议（接口），用于远程 GNN RAG 后端的特征存储
class RAGFeatureStore(Protocol):
    """远程 GNN RAG 后端的特征存储模板。"""
    
    @abstractmethod
    def retrieve_seed_nodes(self, query: Any, **kwargs) -> InputNodes:
        """根据查询获取所有最接近的节点，返回作为 RAG 采样器种子节点的索引。"""
        ...

    @abstractmethod
    def retrieve_seed_edges(self, query: Any, **kwargs) -> InputEdges:
        """根据查询获取所有最接近的边，返回作为 RAG 采样器种子边的索引。"""
        ...

    @abstractmethod
    def load_subgraph(self, sample: Union[SamplerOutput, HeteroSamplerOutput]) -> Union[Data, HeteroData]:
        """将采样的子图输出与特征结合，生成一个 Data 或 HeteroData 对象。"""
        ...


# 定义 RAGGraphStore 协议（接口），用于远程 GNN RAG 后端的图存储
class RAGGraphStore(Protocol):
    """远程 GNN RAG 后端的图存储模板。"""
    
    @abstractmethod
    def sample_subgraph(self, seed_nodes: InputNodes, seed_edges: InputEdges, **kwargs) -> Union[SamplerOutput, HeteroSamplerOutput]:
        """使用种子节点和种子边采样一个子图。"""
        ...

    @abstractmethod
    def register_feature_store(self, feature_store: FeatureStore):
        """注册一个特征存储以供采样器使用，采样器需要从特征存储中获取信息以便在异构图上正常工作。"""
        ...


# RAGQueryLoader 类：用于从远程后端加载并处理 RAG 查询
class RAGQueryLoader:
    """用于从远程后端发起 RAG 查询的加载器。"""
    
    def __init__(self, data: Tuple[RAGFeatureStore, RAGGraphStore],
                 local_filter: Optional[Callable[[Data, Any], Data]] = None,
                 seed_nodes_kwargs: Optional[Dict[str, Any]] = None,
                 seed_edges_kwargs: Optional[Dict[str, Any]] = None,
                 sampler_kwargs: Optional[Dict[str, Any]] = None,
                 loader_kwargs: Optional[Dict[str, Any]] = None):
        """
        初始化 RAG 查询加载器。

        参数:
            data (Tuple[RAGFeatureStore, RAGGraphStore]): 
                远程特征存储和图存储对象。
            local_filter (Optional[Callable[[Data, Any], Data]], optional): 
                用于在数据加载后应用的本地转换函数，默认不应用。
            seed_nodes_kwargs (Optional[Dict[str, Any]], optional): 
                用于获取种子节点的参数，默认为 None。
            seed_edges_kwargs (Optional[Dict[str, Any]], optional): 
                用于获取种子边的参数，默认为 None。
            sampler_kwargs (Optional[Dict[str, Any]], optional): 
                用于采样图的参数，默认为 None。
            loader_kwargs (Optional[Dict[str, Any]], optional): 
                用于加载图特征的参数，默认为 None。
        """
        fstore, gstore = data
        self.feature_store = fstore  # 远程特征存储
        self.graph_store = gstore  # 远程图存储
        self.graph_store.register_feature_store(self.feature_store)  # 注册特征存储
        self.local_filter = local_filter  # 本地过滤器（可选）
        self.seed_nodes_kwargs = seed_nodes_kwargs or {}  # 获取种子节点的参数
        self.seed_edges_kwargs = seed_edges_kwargs or {}  # 获取种子边的参数
        self.sampler_kwargs = sampler_kwargs or {}  # 采样参数
        self.loader_kwargs = loader_kwargs or {}  # 加载图特征的参数

    def query(self, query: Any) -> Data:
        """根据查询获取相关子图及其所有特征属性。"""
        # 获取种子节点和种子边
        seed_nodes = self.feature_store.retrieve_seed_nodes(query, **self.seed_nodes_kwargs)
        seed_edges = self.feature_store.retrieve_seed_edges(query, **self.seed_edges_kwargs)

        # 使用种子节点和种子边从图存储中采样子图
        subgraph_sample = self.graph_store.sample_subgraph(seed_nodes, seed_edges, **self.sampler_kwargs)

        # 加载子图并将其特征与查询结果结合
        data = self.feature_store.load_subgraph(sample=subgraph_sample, **self.loader_kwargs)

        # 如果提供了本地过滤器，则应用本地过滤器
        if self.local_filter:
            data = self.local_filter(data, query)

        return data
