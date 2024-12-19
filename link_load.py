from typing import Any, Callable, Iterator, List, Optional, Tuple, Union
import torch
from torch import Tensor
from torch_geometric.data import Data, FeatureStore, GraphStore, HeteroData
from torch_geometric.loader.base import DataLoaderIterator
from torch_geometric.loader.mixin import (
    AffinityMixin,  # 用于处理计算图的亲和性
    LogMemoryMixin,  # 用于内存日志记录
    MultithreadingMixin,  # 支持多线程处理
)
from torch_geometric.loader.utils import (
    filter_custom_hetero_store,  # 自定义过滤异质图数据
    filter_custom_store,  # 自定义过滤数据
    filter_data,  # 过滤数据
    filter_hetero_data,  # 过滤异质图数据
    get_edge_label_index,  # 获取边标签索引
    infer_filter_per_worker,  # 推断是否每个 worker 进行过滤
)
from torch_geometric.sampler import (
    BaseSampler,  # 基础采样器类
    EdgeSamplerInput,  # 边采样器输入
    HeteroSamplerOutput,  # 异质采样输出
    NegativeSampling,  # 负采样配置
    SamplerOutput,  # 采样器输出
)
from torch_geometric.typing import InputEdges, OptTensor


class LinkLoader(
        torch.utils.data.DataLoader,
        AffinityMixin,
        MultithreadingMixin,
        LogMemoryMixin,
):
    """
    一个从链接信息中进行批量采样的数据加载器，使用：`torch_geometric.sampler.BaseSampler`，
    该实现定义了一个`sample_from_edges`方法，并且在提供的输入`data`对象上得到支持。
    """
    def __init__(
        self,
        data: Union[Data, HeteroData, Tuple[FeatureStore, GraphStore]],  # 输入数据
        link_sampler: BaseSampler,  # 采样器，用于从边信息中采样
        edge_label_index: InputEdges = None,  # 边的索引，定义采样的边
        edge_label: OptTensor = None,  # 边标签，标明边的分类
        edge_label_time: OptTensor = None,  # 边的时间戳，用于时间约束采样
        neg_sampling: Optional[NegativeSampling] = None,  # 负采样配置
        neg_sampling_ratio: Optional[Union[int, float]] = None,  # 负采样比率
        transform: Optional[Callable] = None,  # 可选的转换函数
        transform_sampler_output: Optional[Callable] = None,  # 可选的转换采样输出函数
        filter_per_worker: Optional[bool] = None,  # 是否在每个 worker 中进行过滤
        custom_cls: Optional[HeteroData] = None,  # 自定义的 HeteroData 类，用于远程后端
        input_id: OptTensor = None,  # 输入 ID，用于采样过程
        **kwargs,  # 其他 DataLoader 的参数，如 batch_size, shuffle 等
    ):
        # 如果没有指定 `filter_per_worker`，则根据数据类型推断该选项
        if filter_per_worker is None:
            filter_per_worker = infer_filter_per_worker(data)

        # 移除 PyTorch Lightning 中不需要的参数
        kwargs.pop('dataset', None)
        kwargs.pop('collate_fn', None)
        # 保存数据标签索引用于 PyTorch Lightning
        self.edge_label_index = edge_label_index

        # 如果给定了负采样比率，则用其创建一个新的负采样配置
        if neg_sampling_ratio is not None and neg_sampling_ratio != 0.0:
            neg_sampling = NegativeSampling("binary", neg_sampling_ratio)

        # 获取边类型（如果是同质图则为 None）
        input_type, edge_label_index = get_edge_label_index(data, edge_label_index)

        # 初始化对象的基本参数
        self.data = data
        self.link_sampler = link_sampler
        self.neg_sampling = NegativeSampling.cast(neg_sampling)  # 转换为负采样配置
        self.transform = transform
        self.transform_sampler_output = transform_sampler_output
        self.filter_per_worker = filter_per_worker
        self.custom_cls = custom_cls

        # 处理负采样中的标签，确保标签 0 代表负样本
        if (self.neg_sampling is not None and self.neg_sampling.is_binary()
                and edge_label is not None and edge_label.min() == 0):
            edge_label = edge_label + 1  # 将标签从 0 开始改为 1 开始

        # 处理 triplet 负采样模式的错误
        if (self.neg_sampling is not None and self.neg_sampling.is_triplet()
                and edge_label is not None):
            raise ValueError("'edge_label' needs to be undefined for "
                             "'triplet'-based negative sampling. Please use "
                             "`src_index`, `dst_pos_index` and "
                             "`neg_pos_index` of the returned mini-batch "
                             "instead to differentiate between positive and "
                             "negative samples.")

        # 创建边采样器输入对象，存储采样所需的数据
        self.input_data = EdgeSamplerInput(
            input_id=input_id,
            row=edge_label_index[0],
            col=edge_label_index[1],
            label=edge_label,
            time=edge_label_time,
            input_type=input_type,
        )

        # 创建迭代器，用于采样的索引
        iterator = range(edge_label_index.size(1))
        super().__init__(iterator, collate_fn=self.collate_fn, **kwargs)

    def __call__(
        self,
        index: Union[Tensor, List[int]],
    ) -> Union[Data, HeteroData]:
        """
        从输入的边索引中采样一个子图。
        """
        out = self.collate_fn(index)  # 调用 collate_fn 进行采样
        if not self.filter_per_worker:
            out = self.filter_fn(out)  
        return out

    def collate_fn(self, index: Union[Tensor, List[int]]) -> Any:
        """
        从一批输入边中采样子图。
        """
        # 根据索引选择相应的采样输入数据
        input_data: EdgeSamplerInput = self.input_data[index]

        # 使用 link_sampler 进行边采样
        out = self.link_sampler.sample_from_edges(input_data, 
                                                  neg_sampling=self.neg_sampling)

        # 如果在每个 worker 中进行过滤，则执行过滤函数
        if self.filter_per_worker:
            out = self.filter_fn(out)

        return out

    def filter_fn(
        self,
        out: Union[SamplerOutput, HeteroSamplerOutput],
    ) -> Union[Data, HeteroData]:
        """
        将采样得到的节点与特征连接，返回可用于下游任务的 Data 或 HeteroData 对象。
        """
        # 如果设置了 transform_sampler_output，则先应用转换函数
        if self.transform_sampler_output:
            out = self.transform_sampler_output(out)

        # 处理 SamplerOutput 类型的输出
        if isinstance(out, SamplerOutput):
            if isinstance(self.data, Data):
                # 如果是同质图数据，调用 filter_data 进行数据过滤
                data = filter_data(self.data, out.node, out.row, out.col, out.edge,
                                   self.link_sampler.edge_permutation)
            else:
                # 否则处理异质数据
                data = filter_custom_store(*self.data, out.node, out.row, 
                                           out.col, out.edge, self.custom_cls)
                                     
            # 给数据添加必要的 ID 和标签
            if 'n_id' not in data:
                data.n_id = out.node
            if out.edge is not None and 'e_id' not in data:
                edge = out.edge.to(torch.long)
                perm = self.link_sampler.edge_permutation
                data.e_id = perm[out.edge] if perm is not None else out.edge

            # 记录批次和采样节点、边的数量
            data.batch = out.batch
            data.num_sampled_nodes = out.num_sampled_nodes
            data.num_sampled_edges = out.num_sampled_edges
            data.input_id = out.metadata[0]

            # 如果是二分类负采样，处理边标签
            if self.neg_sampling is None or self.neg_sampling.is_binary():
                data.edge_label_index = out.metadata[1]
                data.edge_label = out.metadata[2]
                data.edge_label_time = out.metadata[3]
            elif self.neg_sampling.is_triplet():
                # 如果是 triplet 负采样，则处理不同的负样本索引
                data.src_index = out.metadata[1]
                data.dst_pos_index = out.metadata[2]
                data.dst_neg_index = out.metadata[3]
                data.seed_time = out.metadata[4]
                # 清理掉不需要的属性
                del data.edge_label_index
                del data.edge_label_time

        # 处理异质采样输出
        elif isinstance(out, HeteroSamplerOutput):
            if isinstance(self.data, HeteroData):
                # 如果是异质图数据，调用 filter_hetero_data 进行过滤
                data = filter_hetero_data(self.data, out, self.link_sampler.edge_permutation)
            else:
                # 否则处理 FeatureStore 和 GraphStore 数据
                data = filter_custom_hetero_store(*self.data, out.node, out.row, out.col, out.edge,
                                                  self.custom_cls)

            # 处理每种类型的节点和边
            for key, node in out.node.items():
                if 'n_id' not in data[key]:
                    data[key].n_id = node

            for key, edge in (out.edge or {}).items():
                if edge is not None and 'e_id' not in data[key]:
                    edge = edge.to(torch.long)
                    perm = self.link_sampler.edge_permutation
                    if perm is not None and perm.get(key, None) is not None:
                        edge = perm[key][edge]
                    data[key].e_id = edge

            # 添加批次信息和采样的节点、边数目
            data.set_value_dict('batch', out.batch)
            data.set_value_dict('num_sampled_nodes', out.num_sampled_nodes)
            data.set_value_dict('num_sampled_edges', out.num_sampled_edges)

            input_type = self.input_data.input_type
            data[input_type].input_id = out.metadata[0]

            # 处理负采样标签
            if self.neg_sampling is None or self.neg_sampling.is_binary():
                data[input_type].edge_label_index = out.metadata[1]
                data[input_type].edge_label = out.metadata[2]
                data[input_type].edge_label_time = out.metadata[3]
            elif self.neg_sampling.is_triplet():
                data[input_type[0]].src_index = out.metadata[1]
                data[input_type[-1]].dst_pos_index = out.metadata[2]
                data[input_type[-1]].dst_neg_index = out.metadata[3]
                data[input_type[0]].seed_time = out.metadata[4]
                data[input_type[-1]].seed_time = out.metadata[4]

                # 清理掉不需要的属性
                if input_type in data.edge_types:
                    del data[input_type].edge_label_index
                    del data[input_type].edge_label_time

        else:
            raise TypeError(f"'{self.__class__.__name__}' found invalid type: '{type(out)}'")

        # 如果定义了 transform 函数，则应用转换
        return data if self.transform is None else self.transform(data)

    def _get_iterator(self) -> Iterator:
        """
        获取迭代器，用于遍历数据集。
        """
        if self.filter_per_worker:
            return super()._get_iterator()  # 如果在每个 worker 中过滤，则使用默认迭代器

        # 否则，在主进程中过滤数据
        return DataLoaderIterator(super()._get_iterator(), self.filter_fn)

    def __repr__(self) -> str:
        """
        返回类的字符串表示。
        """
        return f'{self.__class__.__name__}()'
