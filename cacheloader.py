from collections.abc import Mapping
from typing import Any, Callable, List, Optional, Sequence

import torch
from torch.utils.data import DataLoader


# 将输入的数据迁移到指定的设备（如 GPU 或 CPU）
def to_device(inputs: Any, device: Optional[torch.device] = None) -> Any:
    # 如果输入的数据对象具有 'to' 方法，表示它是一个可以被迁移到指定设备的对象（如 Tensor）
    if hasattr(inputs, 'to'):
        return inputs.to(device)  # 将数据迁移到指定设备
    # 如果输入是字典类型（Mapping），递归地将每个键值对中的数据迁移到指定设备
    elif isinstance(inputs, Mapping):
        return {key: to_device(value, device) for key, value in inputs.items()}
    # 如果输入是命名元组（tuple，且具有 _fields 属性），递归地处理其中的元素
    elif isinstance(inputs, tuple) and hasattr(inputs, '_fields'):
        return type(inputs)(*(to_device(s, device) for s in zip(*inputs)))
    # 如果输入是序列（如列表），递归地处理其中的元素
    elif isinstance(inputs, Sequence) and not isinstance(inputs, str):
        return [to_device(s, device) for s in zip(*inputs)]

    # 如果以上条件都不满足，返回原始输入
    return inputs


# 创建一个缓存的数据加载器，用于存储小批量输出（如：在 NeighborLoader 迭代期间获取的结果）
class CachedLoader:
    r"""一个缓存小批量输出的加载器，例如在 :class:`NeighborLoader` 迭代期间获得的输出。

    参数:
        loader (torch.utils.data.DataLoader): 数据加载器。
        device (torch.device, 可选): 将数据加载到指定设备（默认值为：None）。
        transform (可调用对象, 可选): 一个函数/转换，它接受一个样本小批量并返回转换后的版本。
    """
    def __init__(
        self,
        loader: DataLoader,  # 传入的数据加载器（如训练集的 DataLoader）
        device: Optional[torch.device] = None,  # 可选的设备参数，默认None，表示不指定设备
        transform: Optional[Callable] = None,  # 可选的转换函数，默认None
    ):
        self.loader = loader  # 数据加载器
        self.device = device  # 设备
        self.transform = transform  # 转换函数
        self._cache: List[Any] = []  # 用于缓存加载的小批量数据

    def clear(self):
        r"""清空缓存。"""
        self._cache = []  # 清空缓存列表

    def __iter__(self) -> Any:
        """实现迭代器，返回每个小批量数据。"""
        # 如果缓存中有数据，则直接从缓存中返回
        if len(self._cache):
            for batch in self._cache:
                yield batch
            return

        # 如果缓存为空，则从原始数据加载器中加载数据
        for batch in self.loader:
            # 如果存在转换函数，则应用该转换
            if self.transform is not None:
                batch = self.transform(batch)

            # 将批量数据迁移到指定设备
            batch = to_device(batch, self.device)

            # 将批量数据缓存到列表中
            self._cache.append(batch)

            # 返回当前的小批量数据
            yield batch

    def __len__(self) -> int:
        """返回数据加载器中的数据总数。"""
        return len(self.loader)  # 返回加载器的长度（即小批量的数量）

    def __repr__(self) -> str:
        """返回类的字符串表示。"""
        return f'{self.__class__.__name__}({self.loader})'
