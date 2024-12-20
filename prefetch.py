import warnings  
from contextlib import nullcontext  
from functools import partial  
from typing import Any, Optional  
import torch 
from torch.utils.data import DataLoader  
from torch_geometric.typing import WITH_IPEX 

# DeviceHelper 类用于处理设备相关的操作，简化 GPU 或 CPU 的设置
class DeviceHelper:
    def __init__(self, device: Optional[torch.device] = None):
        # 判断当前系统是否支持 CUDA 和 XPU（如果启用了 IPEX）
        with_cuda = torch.cuda.is_available()  # 检查是否有 CUDA 支持的 GPU
        with_xpu = torch.xpu.is_available() if WITH_IPEX else False 

        # 如果没有指定设备，默认选择一个可用的设备
        if device is None:
            if with_cuda:
                device = 'cuda'  # 优先选择 GPU（CUDA）
            elif with_xpu:
                device = 'xpu'  # 如果有 XPU 支持，则选择 XPU
            else:
                device = 'cpu'  # 如果没有 GPU 或 XPU，选择 CPU

        # 设置设备对象
        self.device = torch.device(device)
        self.is_gpu = self.device.type in ['cuda', 'xpu']  

        # 如果请求的设备不可用，发出警告并回退到 CPU
        if ((self.device.type == 'cuda' and not with_cuda)
                or (self.device.type == 'xpu' and not with_xpu)):
            warnings.warn(f"Requested device '{self.device.type}' is not "
                          f"available, falling back to CPU")
            self.device = torch.device('cpu')  # 如果设备不可用，则使用 CPU

        self.stream = None  # 初始化流对象
        self.stream_context = nullcontext  # 初始化上下文管理器
        self.module = getattr(torch, self.device.type) if self.is_gpu else None  

    def maybe_init_stream(self) -> None:
        """如果是 GPU，则初始化一个 CUDA 流（用于异步操作）。"""
        if self.is_gpu:
            self.stream = self.module.Stream()  # 创建一个新的流对象
            self.stream_context = partial(
                self.module.stream,
                stream=self.stream,
            )  # 使用 partial 创建一个流上下文管理器

    def maybe_wait_stream(self) -> None:
        """等待 GPU 上的操作完成。"""
        if self.stream is not None:
            self.module.current_stream().wait_stream(self.stream)  # 等待流操作完成


# PrefetchLoader 类用于在异步地将数据从主机内存传输到设备内存
class PrefetchLoader:
    r"""一个 GPU 预取器类，用于将数据从主机内存异步地传输到设备内存
    :class:`torch.utils.data.DataLoader` 类的数据加载器。
    参数：
        loader (torch.utils.data.DataLoader): 用于加载数据的 DataLoader。
        device (torch.device, 可选): 指定数据加载目标设备。默认为 None。
    """
    def __init__(
        self,
        loader: DataLoader,  # 输入的 DataLoader，用于从数据集加载数据
        device: Optional[torch.device] = None,  # 目标设备（CPU 或 GPU）
    ):
        self.loader = loader  # 保存传入的 DataLoader
        self.device_helper = DeviceHelper(device)  # 创建 DeviceHelper 实例

    def non_blocking_transfer(self, batch: Any) -> Any:
        """异步将数据从主机内存传输到设备内存。"""
        if not self.device_helper.is_gpu:
            return batch  # 如果不是 GPU，则直接返回数据

        # 如果 batch 是一个列表或元组，递归地对每个元素进行转换
        if isinstance(batch, (list, tuple)):
            return [self.non_blocking_transfer(v) for v in batch]
        # 如果 batch 是字典，递归地对每个键值对进行转换
        if isinstance(batch, dict):
            return {k: self.non_blocking_transfer(v) for k, v in batch.items()}

        batch = batch.pin_memory()  # 将数据从主机内存固定到物理内存中
        return batch.to(self.device_helper.device, non_blocking=True)  # 异步将数据传输到目标设备

    def __iter__(self) -> Any:
        """实现迭代器，用于异步加载数据并传输到设备。"""
        first = True  # 标记是否是第一个 batch
        self.device_helper.maybe_init_stream()  # 初始化流（如果是 GPU）

        batch = None  # 初始化 batch 变量
        for next_batch in self.loader:  # 遍历 DataLoader 中的每个 batch
            with self.device_helper.stream_context():  # 使用流上下文管理器进行数据传输
                next_batch = self.non_blocking_transfer(next_batch)  # 异步传输当前 batch 到设备
            if not first:
                yield batch  # 如果不是第一个 batch，则返回前一个 batch
            else:
                first = False  # 标记为第一个 batch
            self.device_helper.maybe_wait_stream()  # 等待流中的操作完成
            batch = next_batch  # 更新当前 batch
        yield batch  # 最后返回最后一个 batch

    def __len__(self) -> int:
        """返回 DataLoader 中批次的数量。"""
        return len(self.loader)

    def __repr__(self) -> str:
        """返回 PrefetchLoader 的字符串表示。"""
        return f'{self.__class__.__name__}({self.loader})'
