import os.path as osp
import warnings
from abc import abstractmethod
from inspect import Parameter
from typing import (
    Any,
    Callable,
    Dict,
    Final,
    List,
    Optional,
    OrderedDict,
    Set,
    Tuple,
    Union,
)

import torch
from torch import Tensor
from torch.utils.hooks import RemovableHandle

from torch_geometric import EdgeIndex, is_compiling
from torch_geometric.index import ptr2index
from torch_geometric.inspector import Inspector, Signature
from torch_geometric.nn.aggr import Aggregation
from torch_geometric.nn.resolver import aggregation_resolver as aggr_resolver
from torch_geometric.template import module_from_template
from torch_geometric.typing import Adj, Size, SparseTensor
from torch_geometric.utils import (
    is_sparse,
    is_torch_sparse_tensor,
    to_edge_index,
)

FUSE_AGGRS = {'add', 'sum', 'mean', 'min', 'max'}
HookDict = OrderedDict[int, Callable]


class MessagePassing(torch.nn.Module):
    # 预定义一些特殊的参数名称，这些名称将在消息传递过程中使用
    special_args: Set[str] = {
        'edge_index', 'adj_t', 'edge_index_i', 'edge_index_j', 'size',
        'size_i', 'size_j', 'ptr', 'index', 'dim_size'
    }

    # 该标志表示是否支持通过 `EdgeIndex` 进行“消息和聚合”操作
    # TODO：一旦迁移完成，应删除此功能标志
    SUPPORTS_FUSED_EDGE_INDEX: Final[bool] = False

    def __init__(
        self,
        aggr: Optional[Union[str, List[str], Aggregation]] = 'sum',
        *,
        aggr_kwargs: Optional[Dict[str, Any]] = None,
        flow: str = "source_to_target",
        node_dim: int = -2,
        decomposed_layers: int = 1,
    ) -> None:
        # 调用父类构造函数
        super().__init__()

        # 校验 `flow` 参数的有效性，确保其值为 'source_to_target' 或 'target_to_source'
        if flow not in ['source_to_target', 'target_to_source']:
            raise ValueError(f"Expected 'flow' to be either 'source_to_target'"
                             f" or 'target_to_source' (got '{flow}')")

        # 如果 `aggr` 参数为 None，则不进行聚合
        # 如果 `aggr` 是字符串或 Aggregation 类型，将其转为字符串
        # 如果 `aggr` 是列表或元组，将列表中的每个元素转为字符串
        self.aggr: Optional[Union[str, List[str]]]
        if aggr is None:
            self.aggr = None
        elif isinstance(aggr, (str, Aggregation)):
            self.aggr = str(aggr)
        elif isinstance(aggr, (tuple, list)):
            self.aggr = [str(x) for x in aggr]

        # 通过 `aggr_resolver` 解析聚合方式，`aggr_kwargs` 为可选参数
        self.aggr_module = aggr_resolver(aggr, **(aggr_kwargs or {}))
        self.flow = flow
        self.node_dim = node_dim

        # 收集消息传递钩子中请求的属性名称
        self.inspector = Inspector(self.__class__)
        # 检查 `message` 方法的签名
        self.inspector.inspect_signature(self.message)
        # 检查 `aggregate` 方法的签名，排除第一个参数（通常是 self）和 'aggr' 参数
        self.inspector.inspect_signature(self.aggregate, exclude=[0, 'aggr'])
        # 检查 `message_and_aggregate` 方法的签名，排除第一个参数
        self.inspector.inspect_signature(self.message_and_aggregate, [0])
        # 检查 `update` 方法的签名，排除第一个参数
        self.inspector.inspect_signature(self.update, exclude=[0])
        # 检查 `edge_update` 方法的签名
        self.inspector.inspect_signature(self.edge_update)

    
        self._user_args: List[str] = self.inspector.get_flat_param_names(
            ['message', 'aggregate', 'update'], exclude=self.special_args)
        self._fused_user_args: List[str] = self.inspector.get_flat_param_names(
            ['message_and_aggregate', 'update'], exclude=self.special_args)
        self._edge_user_args: List[str] = self.inspector.get_param_names(
            'edge_update', exclude=self.special_args)

        # 支持“融合”消息传递：即在一个操作中同时进行消息和聚合
        self.fuse = self.inspector.implements('message_and_aggregate')
        if self.aggr is not None:
            self.fuse &= isinstance(self.aggr, str) and self.aggr in FUSE_AGGRS

        # 定义消息传递相关的钩子：这些钩子用于在消息传递的不同阶段插入自定义操作
        self._propagate_forward_pre_hooks: HookDict = OrderedDict()
        self._propagate_forward_hooks: HookDict = OrderedDict()
        self._message_forward_pre_hooks: HookDict = OrderedDict()
        self._message_forward_hooks: HookDict = OrderedDict()
        self._aggregate_forward_pre_hooks: HookDict = OrderedDict()
        self._aggregate_forward_hooks: HookDict = OrderedDict()
        self._message_and_aggregate_forward_pre_hooks: HookDict = OrderedDict()
        self._message_and_aggregate_forward_hooks: HookDict = OrderedDict()
        self._edge_update_forward_pre_hooks: HookDict = OrderedDict()
        self._edge_update_forward_hooks: HookDict = OrderedDict()

        self._set_jittable_templates()

        # 解释性：用于启用模型的解释功能
        self._explain: Optional[bool] = None
        # 用于边缘的掩码（例如，边缘选择性掩码）
        self._edge_mask: Optional[Tensor] = None
        # 用于循环的掩码（例如，节点之间的循环消息）
        self._loop_mask: Optional[Tensor] = None
        # 是否应用 Sigmoid 激活函数
        self._apply_sigmoid: bool = True

        # 推理时的特征分解层数
        self._decomposed_layers = 1
        # 设置分解层数，默认为 1
        self.decomposed_layers = decomposed_layers

    def reset_parameters(self) -> None:
        if self.aggr_module is not None:
            self.aggr_module.reset_parameters()

    def __setstate__(self, data: Dict[str, Any]) -> None:
        self.inspector = data['inspector']
        self.fuse = data['fuse']
        self._set_jittable_templates()
        super().__setstate__(data)

    def __repr__(self) -> str:
        channels_repr = ''
        if hasattr(self, 'in_channels') and hasattr(self, 'out_channels'):
            channels_repr = f'{self.in_channels}, {self.out_channels}'
        elif hasattr(self, 'channels'):
            channels_repr = f'{self.channels}'
        return f'{self.__class__.__name__}({channels_repr})'

    # Utilities ###############################################################

    def _check_input(
        self,
        edge_index: Union[Tensor, SparseTensor],
        size: Optional[Tuple[Optional[int], Optional[int]]],
    ) -> List[Optional[int]]:
    # 如果不是 JIT 脚本模式，并且 edge_index 是 EdgeIndex 类型
        if not torch.jit.is_scripting() and isinstance(edge_index, EdgeIndex):
            # 返回 edge_index 的行数和列数
            return [edge_index.num_rows, edge_index.num_cols]
    
        # 判断 edge_index 是否为稀疏张量（torch_sparse.SparseTensor 或 torch.sparse.Tensor）
        if is_sparse(edge_index):
            # 如果流向是 'target_to_source'，则报错，因为不支持这种流向的稀疏张量消息传递
            if self.flow == 'target_to_source':
                raise ValueError(
                    'Flow direction "target_to_source" is invalid for '
                    'message propagation via `torch_sparse.SparseTensor` '
                    'or `torch.sparse.Tensor`. If you really want to make '
                    'use of a reverse message passing flow, pass in the '
                    'transposed sparse tensor to the message passing module, '
                    'e.g., `adj_t.t()`.')
    
            # 如果 edge_index 是 SparseTensor，返回其大小（边的数量和节点的数量）
            if isinstance(edge_index, SparseTensor):
                return [edge_index.size(1), edge_index.size(0)]
            # 对于其他类型的稀疏张量（如 torch.sparse.Tensor），也返回类似的大小
            return [edge_index.size(1), edge_index.size(0)]
    
        # 如果 edge_index 是常规的张量（Tensor）
        elif isinstance(edge_index, Tensor):
            # 定义一个整数类型的元组，表示支持的整数数据类型
            int_dtypes = (torch.uint8, torch.int8, torch.int16, torch.int32,
                          torch.int64)
    
            # 如果 edge_index 的数据类型不是整数类型，抛出异常
            if edge_index.dtype not in int_dtypes:
                raise ValueError(f"Expected 'edge_index' to be of integer "
                                 f"type (got '{edge_index.dtype}')")
            
            # 如果 edge_index 的维度不是 2，抛出异常
            if edge_index.dim() != 2:
                raise ValueError(f"Expected 'edge_index' to be two-dimensional"
                                 f" (got {edge_index.dim()} dimensions)")
            
            # 如果不是 JIT 跟踪模式，并且 edge_index 的第一维大小不是 2，抛出异常
            if not torch.jit.is_tracing() and edge_index.size(0) != 2:
                raise ValueError(f"Expected 'edge_index' to have size '2' in "
                                 f"the first dimension (got "
                                 f"'{edge_index.size(0)}')")
    
            # 如果 size 参数不为 None，返回 size，否则返回 [None, None]
            return list(size) if size is not None else [None, None]
    
        # 如果 edge_index 既不是 Tensor、SparseTensor，也不是 torch.sparse.Tensor，抛出异常
        raise ValueError(
            '`MessagePassing.propagate` only supports integer tensors of '
            'shape `[2, num_messages]`, `torch_sparse.SparseTensor` or '
            '`torch.sparse.Tensor` for argument `edge_index`.')

    def _set_size(
        self,
        size: List[Optional[int]],
        dim: int,
        src: Tensor,
    ) -> None:
        the_size = size[dim]
        if the_size is None:
            size[dim] = src.size(self.node_dim)
        elif the_size != src.size(self.node_dim):
            raise ValueError(
                f'Encountered tensor with size {src.size(self.node_dim)} in '
                f'dimension {self.node_dim}, but expected size {the_size}.')

    def _index_select(self, src: Tensor, index) -> Tensor:
        if torch.jit.is_scripting() or is_compiling():
            return src.index_select(self.node_dim, index)
        else:
            return self._index_select_safe(src, index)

    def _index_select_safe(self, src: Tensor, index: Tensor) -> Tensor:
        try:
            return src.index_select(self.node_dim, index)
        except (IndexError, RuntimeError) as e:
            if index.numel() > 0 and index.min() < 0:
                raise IndexError(
                    f"Found negative indices in 'edge_index' (got "
                    f"{index.min().item()}). Please ensure that all "
                    f"indices in 'edge_index' point to valid indices "
                    f"in the interval [0, {src.size(self.node_dim)}) in "
                    f"your node feature matrix and try again.")

            if (index.numel() > 0 and index.max() >= src.size(self.node_dim)):
                raise IndexError(
                    f"Found indices in 'edge_index' that are larger "
                    f"than {src.size(self.node_dim) - 1} (got "
                    f"{index.max().item()}). Please ensure that all "
                    f"indices in 'edge_index' point to valid indices "
                    f"in the interval [0, {src.size(self.node_dim)}) in "
                    f"your node feature matrix and try again.")

            raise e

    def _lift(
        self,
        src: Tensor,
        edge_index: Union[Tensor, SparseTensor],
        dim: int,
    ) -> Tensor:
        if not torch.jit.is_scripting() and is_torch_sparse_tensor(edge_index):
            assert dim == 0 or dim == 1
            if edge_index.layout == torch.sparse_coo:
                index = edge_index._indices()[1 - dim]
            elif edge_index.layout == torch.sparse_csr:
                if dim == 0:
                    index = edge_index.col_indices()
                else:
                    index = ptr2index(edge_index.crow_indices())
            elif edge_index.layout == torch.sparse_csc:
                if dim == 0:
                    index = ptr2index(edge_index.ccol_indices())
                else:
                    index = edge_index.row_indices()
            else:
                raise ValueError(f"Unsupported sparse tensor layout "
                                 f"(got '{edge_index.layout}')")
            return src.index_select(self.node_dim, index)

        elif isinstance(edge_index, Tensor):
            if torch.jit.is_scripting():  # Try/catch blocks are not supported.
                index = edge_index[dim]
                return src.index_select(self.node_dim, index)
            return self._index_select(src, edge_index[dim])

        elif isinstance(edge_index, SparseTensor):
            row, col, _ = edge_index.coo()
            if dim == 0:
                return src.index_select(self.node_dim, col)
            elif dim == 1:
                return src.index_select(self.node_dim, row)

        raise ValueError(
            '`MessagePassing.propagate` only supports integer tensors of '
            'shape `[2, num_messages]`, `torch_sparse.SparseTensor` '
            'or `torch.sparse.Tensor` for argument `edge_index`.')

    def _collect(
        self,
        args: Set[str],
        edge_index: Union[Tensor, SparseTensor],
        size: List[Optional[int]],
        kwargs: Dict[str, Any],
    ) -> Dict[str, Any]:

        i, j = (1, 0) if self.flow == 'source_to_target' else (0, 1)

        out = {}
        for arg in args:
            if arg[-2:] not in ['_i', '_j']:
                out[arg] = kwargs.get(arg, Parameter.empty)
            else:
                dim = j if arg[-2:] == '_j' else i
                data = kwargs.get(arg[:-2], Parameter.empty)

                if isinstance(data, (tuple, list)):
                    assert len(data) == 2
                    if isinstance(data[1 - dim], Tensor):
                        self._set_size(size, 1 - dim, data[1 - dim])
                    data = data[dim]

                if isinstance(data, Tensor):
                    self._set_size(size, dim, data)
                    data = self._lift(data, edge_index, dim)

                out[arg] = data

        if is_torch_sparse_tensor(edge_index):
            indices, values = to_edge_index(edge_index)
            out['adj_t'] = edge_index
            out['edge_index'] = None
            out['edge_index_i'] = indices[0]
            out['edge_index_j'] = indices[1]
            out['ptr'] = None  # TODO Get `rowptr` from CSR representation.
            if out.get('edge_weight', None) is None:
                out['edge_weight'] = values
            if out.get('edge_attr', None) is None:
                out['edge_attr'] = None if values.dim() == 1 else values
            if out.get('edge_type', None) is None:
                out['edge_type'] = values

        elif isinstance(edge_index, Tensor):
            out['adj_t'] = None
            out['edge_index'] = edge_index
            out['edge_index_i'] = edge_index[i]
            out['edge_index_j'] = edge_index[j]

            out['ptr'] = None
            if isinstance(edge_index, EdgeIndex):
                if i == 0 and edge_index.is_sorted_by_row:
                    (out['ptr'], _), _ = edge_index.get_csr()
                elif i == 1 and edge_index.is_sorted_by_col:
                    (out['ptr'], _), _ = edge_index.get_csc()

        elif isinstance(edge_index, SparseTensor):
            row, col, value = edge_index.coo()
            rowptr, _, _ = edge_index.csr()

            out['adj_t'] = edge_index
            out['edge_index'] = None
            out['edge_index_i'] = row
            out['edge_index_j'] = col
            out['ptr'] = rowptr
            if out.get('edge_weight', None) is None:
                out['edge_weight'] = value
            if out.get('edge_attr', None) is None:
                out['edge_attr'] = value
            if out.get('edge_type', None) is None:
                out['edge_type'] = value

        out['index'] = out['edge_index_i']
        out['size'] = size
        out['size_i'] = size[i] if size[i] is not None else size[j]
        out['size_j'] = size[j] if size[j] is not None else size[i]
        out['dim_size'] = out['size_i']

        return out

    # Message Passing #########################################################

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        r"""Runs the forward pass of the module."""

   def propagate(
        self,
        edge_index: Adj,
        size: Size = None,
        **kwargs: Any,
    ) -> Tensor:
        decomposed_layers = 1 if self.explain else self.decomposed_layers
        # 如果是解释模式，则分解层数为 1，否则使用 decomposed_layers 的值
    
        # 处理传播前的钩子函数（Pre-hooks）
        for hook in self._propagate_forward_pre_hooks.values():
            res = hook(self, (edge_index, size, kwargs))
            if res is not None:
                edge_index, size, kwargs = res
        # 检查输入数据的合法性并得到 mutable_size
        mutable_size = self._check_input(edge_index, size)
    
        # 判断是否采用融合消息传递和聚合操作
        fuse = False
        if self.fuse and not self.explain:
            if is_sparse(edge_index):  # 如果 edge_index 是稀疏格式
                fuse = True
            elif (not torch.jit.is_scripting()
                  and isinstance(edge_index, EdgeIndex)):
                # 如果当前不是 TorchScript 模式并且 edge_index 是 EdgeIndex 类型
                if (self.SUPPORTS_FUSED_EDGE_INDEX
                        and edge_index.is_sorted_by_col):
                    fuse = True
        # 如果启用消息和聚合操作的融合，执行以下流程
        if fuse:
            # 收集必要的参数并执行消息和聚合操作
            coll_dict = self._collect(self._fused_user_args, edge_index,
                                      mutable_size, kwargs)
            
            # 收集“消息和聚合”所需的参数
            msg_aggr_kwargs = self.inspector.collect_param_data(
                'message_and_aggregate', coll_dict)
            
            # 执行传播前的钩子函数
            for hook in self._message_and_aggregate_forward_pre_hooks.values():
                res = hook(self, (edge_index, msg_aggr_kwargs))
                if res is not None:
                    edge_index, msg_aggr_kwargs = res
    
            # 执行消息和聚合操作
            out = self.message_and_aggregate(edge_index, **msg_aggr_kwargs)
    
            # 执行消息和聚合后的钩子函数
            for hook in self._message_and_aggregate_forward_hooks.values():
                res = hook(self, (edge_index, msg_aggr_kwargs), out)
                if res is not None:
                    out = res
    
            # 执行节点更新
            update_kwargs = self.inspector.collect_param_data(
                'update', coll_dict)
            out = self.update(out, **update_kwargs)
    
        else:  # 如果不启用融合操作，则分别执行消息传播和聚合操作
            if decomposed_layers > 1:
                user_args = self._user_args
                # 获取与分解相关的参数
                decomp_args = {a[:-2] for a in user_args if a[-2:] == '_j'}
                decomp_kwargs = {
                    a: kwargs[a].chunk(decomposed_layers, -1)
                    for a in decomp_args
                }
                decomp_out = []
    
            for i in range(decomposed_layers):
                if decomposed_layers > 1:
                    # 每层使用不同的参数
                    for arg in decomp_args:
                        kwargs[arg] = decomp_kwargs[arg][i]
    
                # 收集消息传播的参数
                coll_dict = self._collect(self._user_args, edge_index,
                                          mutable_size, kwargs)
                
                # 收集“消息”操作所需的参数
                msg_kwargs = self.inspector.collect_param_data(
                    'message', coll_dict)
    
                # 执行消息传播前的钩子函数
                for hook in self._message_forward_pre_hooks.values():
                    res = hook(self, (msg_kwargs, ))
                    if res is not None:
                        msg_kwargs = res[0] if isinstance(res, tuple) else res
                # 执行消息传播操作
                out = self.message(**msg_kwargs)
    
                # 执行消息传播后的钩子函数
                for hook in self._message_forward_hooks.values():
                    res = hook(self, (msg_kwargs, ), out)
                    if res is not None:
                        out = res
                # 如果是解释模式，则执行解释相关的操作
                if self.explain:
                    explain_msg_kwargs = self.inspector.collect_param_data(
                        'explain_message', coll_dict)
                    out = self.explain_message(out, **explain_msg_kwargs)
                # 聚合消息
                aggr_kwargs = self.inspector.collect_param_data(
                    'aggregate', coll_dict)
                for hook in self._aggregate_forward_pre_hooks.values():
                    res = hook(self, (aggr_kwargs, ))
                    if res is not None:
                        aggr_kwargs = res[0] if isinstance(res, tuple) else res
    
                # 执行聚合操作
                out = self.aggregate(out, **aggr_kwargs)
    
                # 执行聚合后的钩子函数
                for hook in self._aggregate_forward_hooks.values():
                    res = hook(self, (aggr_kwargs, ), out)
                    if res is not None:
                        out = res
                # 执行节点更新操作
                update_kwargs = self.inspector.collect_param_data(
                    'update', coll_dict)
                out = self.update(out, **update_kwargs)
    
                if decomposed_layers > 1:
                    decomp_out.append(out)
            if decomposed_layers > 1:
                out = torch.cat(decomp_out, dim=-1)
    
        # 执行传播后的钩子函数
        for hook in self._propagate_forward_hooks.values():
            res = hook(self, (edge_index, mutable_size, kwargs), out)
            if res is not None:
                out = res
        return out


    def message(self, x_j: Tensor) -> Tensor: 
        r"""构建从节点 :math:`j` 到节点 :math:`i` 的消息
        该函数可以接收任何作为参数传递给 :meth:`propagate` 方法的输入。
        此外，通过在变量名后添加 :obj:`_i` 或 :obj:`_j`，例如 :obj:`x_i` 和 :obj:`x_j`，
        可以将传递给 :meth:`propagate` 的张量映射到相应的节点 :math:`i` 或 :math:`j`。
        
        该函数的作用是从邻居节点 `j` 向当前节点 `i` 传递信息（消息）。
        目前在这个实现中，它只是简单地返回 `x_j`，即直接传递 `j` 节点的特征。
        """
        return x_j


    def aggregate(
    self,
    inputs: Tensor,
    index: Tensor,
    ptr: Optional[Tensor] = None,
    dim_size: Optional[int] = None,
) -> Tensor:
        r"""聚合来自邻居节点的消息
        该函数接收消息计算的输出作为第一个参数，并且可以接收
        任何传递给 :meth:`propagate` 的参数。
        默认情况下，该函数会将调用委托给底层的 
        :class:`~torch_geometric.nn.aggr.Aggregation` 模块，
        以根据 :meth:`__init__` 中通过 :obj:`aggr` 参数指定的聚合方式来减少消息。
        """
        return self.aggr_module(inputs, index, ptr=ptr, dim_size=dim_size,
                                dim=self.node_dim)
    

    @abstractmethod
    def message_and_aggregate(self, edge_index: Adj) -> Tensor:
        r"""将 :func:`message` 和 :func:`aggregate` 的计算融合为一个函数。
    
        这样做可以节省时间和内存，因为消息不需要显式地被构建（materialized）。
        如果适用，消息的计算会直接在传递过程中进行，从而避免中间步骤。
    
        该函数只有在实现时才会被调用，且传播操作必须基于 :obj:`torch_sparse.SparseTensor`
        或 :obj:`torch.sparse.Tensor`（即稀疏张量）。
        """
        raise NotImplementedError


    def update(self, inputs: Tensor) -> Tensor:
        r"""更新每个节点的嵌入，类似于
        :math:`\gamma_{\mathbf{\Theta}}` 对于每个节点
        :math:`i \in \mathcal{V}`。
        该函数接收聚合操作后的输出作为第一个参数，并且可以接收
        任何传递给 :meth:`propagate` 的参数。
    
        默认情况下，这个函数直接返回聚合后的输入，
        但在实际应用中可以通过对输入进行非线性变换等操作来更新节点的嵌入。
        """
        return inputs


    # Edge-level Updates ######################################################

    def edge_updater(
        self,
        edge_index: Adj,
        size: Size = None,
        **kwargs: Any,
    ) -> Tensor:
        r"""计算或更新图中每条边的特征（嵌入）。
        参数：
            edge_index (torch.Tensor 或 SparseTensor): 一个表示图的连接性或消息传递流的边索引，
            size ((int, int), 可选): 当 `edge_index` 是一个 :obj:`torch.Tensor` 时，
                这是分配矩阵的大小 :obj:`(N, M)`。
                如果设置为 :obj:`None`，则会自动推断大小并假设其为方阵。
                如果 `edge_index` 是稀疏张量类型，则此参数会被忽略。
                (默认值: :obj:`None`)
            **kwargs: 计算或更新图中每条边特征所需的其他数据。
        """
        # 执行边更新前的钩子函数
        for hook in self._edge_update_forward_pre_hooks.values():
            res = hook(self, (edge_index, size, kwargs))
            if res is not None:
                edge_index, size, kwargs = res
    
        # 检查输入的有效性
        mutable_size = self._check_input(edge_index, size=None)
    
        # 收集用于边更新的用户参数
        coll_dict = self._collect(self._edge_user_args, edge_index,
                                  mutable_size, kwargs)
    
        # 从收集的参数中获取边更新所需的数据
        edge_kwargs = self.inspector.collect_param_data(
            'edge_update', coll_dict)
    
        # 调用边更新方法，计算或更新边特征
        out = self.edge_update(**edge_kwargs)
    
        # 执行边更新后的钩子函数
        for hook in self._edge_update_forward_hooks.values():
            res = hook(self, (edge_index, size, kwargs), out)
            if res is not None:
                out = res
    
        return out

    @abstractmethod
    def edge_update(self) -> Tensor:
        r"""计算或更新图中每条边的特征。
        该函数可以接受传递给 :meth:`edge_updater` 的任何参数作为输入。
        此外，传递给 :meth:`edge_updater` 的张量可以通过将变量名后缀加上 
        :obj:`_i` 或 :obj:`_j` 来映射到相应的节点 :math:`i` 和 :math:`j`
        """
        raise NotImplementedError


    # Inference Decomposition #################################################

    @property
    def decomposed_layers(self) -> int:
        # 这是属性方法，返回当前的分解层次值
        return self._decomposed_layers
    
    @decomposed_layers.setter
    def decomposed_layers(self, decomposed_layers: int) -> None:
        # 设置分解层次时的操作
    
        # 如果当前是在 JIT 脚本模式下，抛出异常
        # 因为推理分解功能不支持 JIT 脚本模式
        if torch.jit.is_scripting():
            raise ValueError("Inference decomposition of message passing "
                             "modules is only supported on the Python module")
    
        # 如果传入的分解层次与当前值相同，直接返回，避免无用的计算
        if decomposed_layers == self._decomposed_layers:
            return  # Abort early if nothing to do.
    
        # 更新 _decomposed_layers 为新的值
        self._decomposed_layers = decomposed_layers
    
        # 如果分解层次不等于 1，恢复原始的 propagate 方法
        # 这意味着使用原始的消息传递实现
        if decomposed_layers != 1:
            if hasattr(self.__class__, '_orig_propagate'):
                # 恢复 propagate 方法为原始的 _orig_propagate 方法
                self.propagate = self.__class__._orig_propagate.__get__(
                    self, MessagePassing)
    
        # 如果分解层次为 1，并且 `explain` 为 None 或 False，选择使用 jinja_propagate
        # 这可能是用于调试或解释的替代实现
        elif self.explain is None or self.explain is False:
            if hasattr(self.__class__, '_jinja_propagate'):
                # 设置 propagate 为 _jinja_propagate 方法
                self.propagate = self.__class__._jinja_propagate.__get__(
                    self, MessagePassing)


    # Explainability ##########################################################

    @property
    def explain(self) -> Optional[bool]:
        # 这是一个属性方法，用于获取 `_explain` 的值。
        # 它返回一个布尔值，指示当前是否启用了可解释性。
        return self._explain
    
    @explain.setter
    def explain(self, explain: Optional[bool]) -> None:
        # 这是一个 setter 方法，用于设置 `_explain` 的值。
        # 它根据传入的 `explain` 参数决定是否启用可解释性。
        
        # 检查当前是否是 TorchScript 编译模式。如果是，抛出异常。
        # 说明：PyTorch 的 TorchScript 模式不支持启用消息传递模块的可解释性。
        if torch.jit.is_scripting():
            raise ValueError("Explainability of message passing modules "
                             "is only supported on the Python module")
    
        # 如果新设置的 `explain` 和当前的 `_explain` 相同，直接返回，无需进行进一步操作
        if explain == self._explain:
            return  # Abort early if nothing to do.
    
        # 更新 `_explain` 为传入的值
        self._explain = explain
    
        # 如果启用了可解释性 (explain == True)
        if explain is True:
            # 确保当前的分解层数为 1。因为可解释性需要层数为 1，否则会有不一致。
            assert self.decomposed_layers == 1
            
            # 移除先前的签名并开始新的签名检查
            self.inspector.remove_signature(self.explain_message)
            self.inspector.inspect_signature(self.explain_message, exclude=[0])
            
            # 获取 "message"、"explain_message"、"aggregate" 和 "update" 方法中的参数名，
            # 并将这些参数名平展以便后续分析。
            self._user_args = self.inspector.get_flat_param_names(
                funcs=['message', 'explain_message', 'aggregate', 'update'],
                exclude=self.special_args,
            )
    
            # 如果类中有 `_orig_propagate`，则恢复 `propagate` 方法为原始方法
            if hasattr(self.__class__, '_orig_propagate'):
                self.propagate = self.__class__._orig_propagate.__get__(
                    self, MessagePassing)
        else:
            # 如果禁用可解释性 (explain == False)
            # 重新获取 "message"、"aggregate" 和 "update" 方法的参数名，并展平这些参数
            self._user_args = self.inspector.get_flat_param_names(
                funcs=['message', 'aggregate', 'update'],
                exclude=self.special_args,
            )
            
            # 如果分解层数为 1，恢复 `propagate` 为 `jinja_propagate` 方法
            if self.decomposed_layers == 1:
                if hasattr(self.__class__, '_jinja_propagate'):
                    self.propagate = self.__class__._jinja_propagate.__get__(
                        self, MessagePassing)

    def explain_message(
        self,
        inputs: Tensor,
        dim_size: Optional[int],
    ) -> Tensor:
        # NOTE Replace this method in custom explainers per message-passing
        # layer to customize how messages shall be explained, e.g., via:
        # conv.explain_message = explain_message.__get__(conv, MessagePassing)
        # see stackoverflow.com: 394770/override-a-method-at-instance-level
        edge_mask = self._edge_mask

        if edge_mask is None:
            raise ValueError("Could not find a pre-defined 'edge_mask' "
                             "to explain. Did you forget to initialize it?")

        if self._apply_sigmoid:
            edge_mask = edge_mask.sigmoid()

        # Some ops add self-loops to `edge_index`. We need to do the same for
        # `edge_mask` (but do not train these entries).
        if inputs.size(self.node_dim) != edge_mask.size(0):
            assert dim_size is not None
            edge_mask = edge_mask[self._loop_mask]
            loop = edge_mask.new_ones(dim_size)
            edge_mask = torch.cat([edge_mask, loop], dim=0)
        assert inputs.size(self.node_dim) == edge_mask.size(0)

        size = [1] * inputs.dim()
        size[self.node_dim] = -1
        return inputs * edge_mask.view(size)

    # Hooks ###################################################################

    def register_propagate_forward_pre_hook(
        self,
        hook: Callable,
    ) -> RemovableHandle:
        r"""Registers a forward pre-hook on the module.

        The hook will be called every time before :meth:`propagate` is invoked.
        It should have the following signature:

        .. code-block:: python

            hook(module, inputs) -> None or modified input

        The hook can modify the input.
        Input keyword arguments are passed to the hook as a dictionary in
        :obj:`inputs[-1]`.

        Returns a :class:`torch.utils.hooks.RemovableHandle` that can be used
        to remove the added hook by calling :obj:`handle.remove()`.
        """
        handle = RemovableHandle(self._propagate_forward_pre_hooks)
        self._propagate_forward_pre_hooks[handle.id] = hook
        return handle

    def register_propagate_forward_hook(
        self,
        hook: Callable,
    ) -> RemovableHandle:
        r"""Registers a forward hook on the module.

        The hook will be called every time after :meth:`propagate` has computed
        an output.
        It should have the following signature:

        .. code-block:: python

            hook(module, inputs, output) -> None or modified output

        The hook can modify the output.
        Input keyword arguments are passed to the hook as a dictionary in
        :obj:`inputs[-1]`.

        Returns a :class:`torch.utils.hooks.RemovableHandle` that can be used
        to remove the added hook by calling :obj:`handle.remove()`.
        """
        handle = RemovableHandle(self._propagate_forward_hooks)
        self._propagate_forward_hooks[handle.id] = hook
        return handle

    def register_message_forward_pre_hook(
        self,
        hook: Callable,
    ) -> RemovableHandle:
        r"""Registers a forward pre-hook on the module.
        The hook will be called every time before :meth:`message` is invoked.
        See :meth:`register_propagate_forward_pre_hook` for more information.
        """
        handle = RemovableHandle(self._message_forward_pre_hooks)
        self._message_forward_pre_hooks[handle.id] = hook
        return handle

    def register_message_forward_hook(self, hook: Callable) -> RemovableHandle:
        r"""Registers a forward hook on the module.
        The hook will be called every time after :meth:`message` has computed
        an output.
        See :meth:`register_propagate_forward_hook` for more information.
        """
        handle = RemovableHandle(self._message_forward_hooks)
        self._message_forward_hooks[handle.id] = hook
        return handle

    def register_aggregate_forward_pre_hook(
        self,
        hook: Callable,
    ) -> RemovableHandle:
        r"""Registers a forward pre-hook on the module.
        The hook will be called every time before :meth:`aggregate` is invoked.
        See :meth:`register_propagate_forward_pre_hook` for more information.
        """
        handle = RemovableHandle(self._aggregate_forward_pre_hooks)
        self._aggregate_forward_pre_hooks[handle.id] = hook
        return handle

    def register_aggregate_forward_hook(
        self,
        hook: Callable,
    ) -> RemovableHandle:
        r"""Registers a forward hook on the module.
        The hook will be called every time after :meth:`aggregate` has computed
        an output.
        See :meth:`register_propagate_forward_hook` for more information.
        """
        handle = RemovableHandle(self._aggregate_forward_hooks)
        self._aggregate_forward_hooks[handle.id] = hook
        return handle

    def register_message_and_aggregate_forward_pre_hook(
        self,
        hook: Callable,
    ) -> RemovableHandle:
        r"""Registers a forward pre-hook on the module.
        The hook will be called every time before :meth:`message_and_aggregate`
        is invoked.
        See :meth:`register_propagate_forward_pre_hook` for more information.
        """
        handle = RemovableHandle(self._message_and_aggregate_forward_pre_hooks)
        self._message_and_aggregate_forward_pre_hooks[handle.id] = hook
        return handle

    def register_message_and_aggregate_forward_hook(
        self,
        hook: Callable,
    ) -> RemovableHandle:
        r"""Registers a forward hook on the module.
        The hook will be called every time after :meth:`message_and_aggregate`
        has computed an output.
        See :meth:`register_propagate_forward_hook` for more information.
        """
        handle = RemovableHandle(self._message_and_aggregate_forward_hooks)
        self._message_and_aggregate_forward_hooks[handle.id] = hook
        return handle

    def register_edge_update_forward_pre_hook(
        self,
        hook: Callable,
    ) -> RemovableHandle:
        r"""Registers a forward pre-hook on the module.
        The hook will be called every time before :meth:`edge_update` is
        invoked. See :meth:`register_propagate_forward_pre_hook` for more
        information.
        """
        handle = RemovableHandle(self._edge_update_forward_pre_hooks)
        self._edge_update_forward_pre_hooks[handle.id] = hook
        return handle

    def register_edge_update_forward_hook(
        self,
        hook: Callable,
    ) -> RemovableHandle:
        r"""Registers a forward hook on the module.
        The hook will be called every time after :meth:`edge_update` has
        computed an output.
        See :meth:`register_propagate_forward_hook` for more information.
        """
        handle = RemovableHandle(self._edge_update_forward_hooks)
        self._edge_update_forward_hooks[handle.id] = hook
        return handle

    # TorchScript Support #####################################################

    def _set_jittable_templates(self, raise_on_error: bool = False) -> None:
        root_dir = osp.dirname(osp.realpath(__file__))
        jinja_prefix = f'{self.__module__}_{self.__class__.__name__}'
        # Optimize `propagate()` via `*.jinja` templates:
        if not self.propagate.__module__.startswith(jinja_prefix):
            try:
                if ('propagate' in self.__class__.__dict__
                        and self.__class__.__dict__['propagate']
                        != MessagePassing.propagate):
                    raise ValueError("Cannot compile custom 'propagate' "
                                     "method")

                module = module_from_template(
                    module_name=f'{jinja_prefix}_propagate',
                    template_path=osp.join(root_dir, 'propagate.jinja'),
                    tmp_dirname='message_passing',
                    # Keyword arguments:
                    modules=self.inspector._modules,
                    collect_name='collect',
                    signature=self._get_propagate_signature(),
                    collect_param_dict=self.inspector.get_flat_param_dict(
                        ['message', 'aggregate', 'update']),
                    message_args=self.inspector.get_param_names('message'),
                    aggregate_args=self.inspector.get_param_names('aggregate'),
                    message_and_aggregate_args=self.inspector.get_param_names(
                        'message_and_aggregate'),
                    update_args=self.inspector.get_param_names('update'),
                    fuse=self.fuse,
                )

                self.__class__._orig_propagate = self.__class__.propagate
                self.__class__._jinja_propagate = module.propagate

                self.__class__.propagate = module.propagate
                self.__class__.collect = module.collect
            except Exception as e:  # pragma: no cover
                if raise_on_error:
                    raise e
                self.__class__._orig_propagate = self.__class__.propagate
                self.__class__._jinja_propagate = self.__class__.propagate

        # Optimize `edge_updater()` via `*.jinja` templates (if implemented):
        if (self.inspector.implements('edge_update')
                and not self.edge_updater.__module__.startswith(jinja_prefix)):
            try:
                if ('edge_updater' in self.__class__.__dict__
                        and self.__class__.__dict__['edge_updater']
                        != MessagePassing.edge_updater):
                    raise ValueError("Cannot compile custom 'edge_updater' "
                                     "method")

                module = module_from_template(
                    module_name=f'{jinja_prefix}_edge_updater',
                    template_path=osp.join(root_dir, 'edge_updater.jinja'),
                    tmp_dirname='message_passing',
                    # Keyword arguments:
                    modules=self.inspector._modules,
                    collect_name='edge_collect',
                    signature=self._get_edge_updater_signature(),
                    collect_param_dict=self.inspector.get_param_dict(
                        'edge_update'),
                )

                self.__class__._orig_edge_updater = self.__class__.edge_updater
                self.__class__._jinja_edge_updater = module.edge_updater

                self.__class__.edge_updater = module.edge_updater
                self.__class__.edge_collect = module.edge_collect
            except Exception as e:  # pragma: no cover
                if raise_on_error:
                    raise e
                self.__class__._orig_edge_updater = self.__class__.edge_updater
                self.__class__._jinja_edge_updater = (
                    self.__class__.edge_updater)

    def _get_propagate_signature(self) -> Signature:
        param_dict = self.inspector.get_params_from_method_call(
            'propagate', exclude=[0, 'edge_index', 'size'])
        update_signature = self.inspector.get_signature('update')

        return Signature(
            param_dict=param_dict,
            return_type=update_signature.return_type,
            return_type_repr=update_signature.return_type_repr,
        )

    def _get_edge_updater_signature(self) -> Signature:
        param_dict = self.inspector.get_params_from_method_call(
            'edge_updater', exclude=[0, 'edge_index', 'size'])
        edge_update_signature = self.inspector.get_signature('edge_update')

        return Signature(
            param_dict=param_dict,
            return_type=edge_update_signature.return_type,
            return_type_repr=edge_update_signature.return_type_repr,
        )

    def jittable(self, typing: Optional[str] = None) -> 'MessagePassing':
        r"""Analyzes the :class:`MessagePassing` instance and produces a new
        jittable module that can be used in combination with
        :meth:`torch.jit.script`.

        .. note::
            :meth:`jittable` is deprecated and a no-op from :pyg:`PyG` 2.5
            onwards.
        """
        warnings.warn(f"'{self.__class__.__name__}.jittable' is deprecated "
                      f"and a no-op. Please remove its usage.")
        return self
