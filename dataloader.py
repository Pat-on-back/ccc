from collections.abc import Mapping
from typing import Any, List, Optional, Sequence, Union

import torch.utils.data
from torch.utils.data.dataloader import default_collate

from torch_geometric.data import Batch, Dataset
from torch_geometric.data.data import BaseData
from torch_geometric.data.datapipes import DatasetAdapter
from torch_geometric.typing import TensorFrame, torch_frame


class Collater:
    """
    Collater类负责将不同类型的数据合并成一个小批量（mini-batch）。
    它通过检查每个数据元素的类型来决定如何处理该元素。
    """
    def __init__(
        self,
        dataset: Union[Dataset, Sequence[BaseData], DatasetAdapter],
        follow_batch: Optional[List[str]] = None,
        exclude_keys: Optional[List[str]] = None,
    ):
        """
        初始化Collater实例。

        参数：
        - dataset: 数据集，可以是Dataset类型，BaseData类型序列，或者DatasetAdapter。
        - follow_batch: 可选的字段列表，指定哪些字段在批处理中要“跟随”。
        - exclude_keys: 可选的字段列表，指定哪些字段在批处理中需要排除。
        """
        self.dataset = dataset
        self.follow_batch = follow_batch
        self.exclude_keys = exclude_keys

    def __call__(self, batch: List[Any]) -> Any:
        """
        根据batch中数据的类型选择合适的合并方式。
        参数：
        - batch: 一个包含多个样本的数据列表。
        返回：
        - 返回合并后的批数据。
        """
        elem = batch[0]  # 获取batch中的第一个元素，检查其类型
        if isinstance(elem, BaseData):
            # 如果元素是BaseData类型，使用Batch.from_data_list进行合并
            return Batch.from_data_list(
                batch,
                follow_batch=self.follow_batch,
                exclude_keys=self.exclude_keys,
            )
        elif isinstance(elem, torch.Tensor):
            # 如果元素是torch.Tensor类型，使用默认的合并方式
            return default_collate(batch)
        elif isinstance(elem, TensorFrame):
            # 如果元素是TensorFrame类型，使用torch_frame.cat进行合并
            return torch_frame.cat(batch, dim=0)
        elif isinstance(elem, float):
            # 如果元素是float类型，转为torch.float类型的tensor
            return torch.tensor(batch, dtype=torch.float)
        elif isinstance(elem, int):
            # 如果元素是int类型，转为torch.int类型的tensor
            return torch.tensor(batch)
        elif isinstance(elem, str):
            # 如果元素是字符串，直接返回batch列表
            return batch
        elif isinstance(elem, Mapping):
            # 如果元素是字典类型，递归处理每个键值对
            return {key: self([data[key] for data in batch]) for key in elem}
        elif isinstance(elem, tuple) and hasattr(elem, '_fields'):
            # 如果元素是命名元组，递归处理每个字段
            return type(elem)(*(self(s) for s in zip(*batch)))
        elif isinstance(elem, Sequence) and not isinstance(elem, str):
            # 如果元素是序列（但不是字符串），递归处理每个元素
            return [self(s) for s in zip(*batch)]

        # 如果数据类型不符合预期，抛出错误
        raise TypeError(f"DataLoader found invalid type: '{type(elem)}'")


class DataLoader(torch.utils.data.DataLoader):
    """
    一个自定义的数据加载器，用于将图神经网络的数据对象（如Data或HeteroData）合并成小批量。
    
    参数：
    - dataset: 数据集，类型为Dataset。
    - batch_size: 每个小批量的样本数，默认是1。
    - shuffle: 是否在每个epoch时重新打乱数据，默认是False。
    - follow_batch: 指定哪些字段要在批处理中“跟随”。
    - exclude_keys: 指定哪些字段在批处理中要排除。
    - **kwargs: 其他可选的DataLoader参数。
    """
    def __init__(
        self,
        dataset: Union[Dataset, Sequence[BaseData], DatasetAdapter],
        batch_size: int = 1,
        shuffle: bool = False,
        follow_batch: Optional[List[str]] = None,
        exclude_keys: Optional[List[str]] = None,
        **kwargs,
    ):
        # 为了与PyTorch Lightning兼容，移除collate_fn参数
        kwargs.pop('collate_fn', None)

        # 保存follow_batch和exclude_keys参数
        self.follow_batch = follow_batch
        self.exclude_keys = exclude_keys

        # 初始化父类DataLoader，并指定自定义的collate_fn（即Collater类）
        super().__init__(
            dataset,
            batch_size,
            shuffle,
            collate_fn=Collater(dataset, follow_batch, exclude_keys),
            **kwargs,
        )
