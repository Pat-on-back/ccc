import torch
import torch.nn.functional as F
from ogb.nodeproppred import Evaluator, PygNodePropPredDataset
from torch.nn import LayerNorm, Linear, ReLU
from tqdm import tqdm

from torch_geometric.loader import RandomNodeLoader
from torch_geometric.nn import DeepGCNLayer, GENConv
from torch_geometric.utils import scatter

# 加载 OGB 数据集 'ogbn-proteins'
dataset = PygNodePropPredDataset('ogbn-proteins', root='../data')
splitted_idx = dataset.get_idx_split()  # 获取数据集的训练、验证、测试分割索引
data = dataset[0]  # 获取图数据
data.node_species = None  # 删除节点种类信息（不需要的特征）
data.y = data.y.to(torch.float)  # 将标签转换为浮点型

# 初始化节点特征，使用边特征的求和聚合
row, col = data.edge_index
data.x = scatter(data.edge_attr, col, dim_size=data.num_nodes, reduce='sum')  

# 设置训练、验证、测试集的掩码
for split in ['train', 'valid', 'test']:
    mask = torch.zeros(data.num_nodes, dtype=torch.bool)  # 初始化为全False
    mask[splitted_idx[split]] = True  # 将对应的数据集索引位置设为True
    data[f'{split}_mask'] = mask  # 将掩码添加到数据中

# 初始化训练和测试数据加载器，使用 RandomNodeLoader 随机采样节点并返回诱导子图
train_loader = RandomNodeLoader(data, num_parts=40, shuffle=True, num_workers=5)
test_loader = RandomNodeLoader(data, num_parts=5, num_workers=5)


class DeeperGCN(torch.nn.Module):
    """
    一个深度图神经网络模型，使用多层 GCN 层和残差连接。
    """

    def __init__(self, hidden_channels, num_layers):
        super().__init__()

        # 节点特征编码器，将节点特征转换为 hidden_channels 维度
        self.node_encoder = Linear(data.x.size(-1), hidden_channels)
        # 边特征编码器，将边特征转换为 hidden_channels 维度
        self.edge_encoder = Linear(data.edge_attr.size(-1), hidden_channels)

        # 定义多个 GCN 层
        self.layers = torch.nn.ModuleList()
        for i in range(1, num_layers + 1):
            # 使用 GENConv 作为图卷积层
            conv = GENConv(hidden_channels, hidden_channels, aggr='softmax', t=1.0, learn_t=True, num_layers=2, norm='layer')
            norm = LayerNorm(hidden_channels, elementwise_affine=True)  # 层归一化
            act = ReLU(inplace=True)  # 激活函数 ReLU

            # DeepGCNLayer 是一个封装了卷积、激活和归一化的模块，包含残差连接和 dropout
            layer = DeepGCNLayer(conv, norm, act, block='res+', dropout=0.1, ckpt_grad=i % 3)
            self.layers.append(layer)

        # 最后的线性层，将输出维度转换为标签的维度
        self.lin = Linear(hidden_channels, data.y.size(-1))

    def forward(self, x, edge_index, edge_attr):
        """
        前向传播函数，将节点特征和边特征输入，计算图的输出。
        """

        # 编码节点特征和边特征
        x = self.node_encoder(x)
        edge_attr = self.edge_encoder(edge_attr)

        # 第一层卷积
        x = self.layers[0].conv(x, edge_index, edge_attr)

        # 其他层的卷积
        for layer in self.layers[1:]:
            x = layer(x, edge_index, edge_attr)

        # 应用激活函数和层归一化
        x = self.layers[0].act(self.layers[0].norm(x))
        # Dropout 操作，避免过拟合
        x = F.dropout(x, p=0.1, training=self.training)

        # 输出节点的最终预测值
        return self.lin(x)


# 设置设备（GPU 或 CPU）
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 初始化模型
model = DeeperGCN(hidden_channels=64, num_layers=28).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)  # 使用 Adam 优化器
criterion = torch.nn.BCEWithLogitsLoss()  # 使用二元交叉熵损失
evaluator = Evaluator('ogbn-proteins')  # 使用 OGB 提供的评估器

def train(epoch):
    """
    训练函数，计算每个 epoch 的损失，并更新模型参数。
    """
    model.train()  # 设置模型为训练模式

    pbar = tqdm(total=len(train_loader))  # 使用 tqdm 显示训练进度
    pbar.set_description(f'Training epoch: {epoch:04d}')

    total_loss = total_examples = 0
    for data in train_loader:  # 遍历训练数据加载器
        optimizer.zero_grad()  # 清空梯度
        data = data.to(device)  # 将数据迁移到指定设备
        out = model(data.x, data.edge_index, data.edge_attr)  # 模型输出
        loss = criterion(out[data.train_mask], data.y[data.train_mask])  # 计算损失
        loss.backward()  # 反向传播
        optimizer.step()  # 更新模型参数

        total_loss += float(loss) * int(data.train_mask.sum())  # 累加损失
        total_examples += int(data.train_mask.sum())  # 累加样本数量

        pbar.update(1)  # 更新进度条

    pbar.close()  # 关闭进度条

    # 返回平均损失
    return total_loss / total_examples


@torch.no_grad()  # 在评估时不需要计算梯度
def test():
    """
    测试函数，评估模型在训练集、验证集和测试集上的性能。
    """
    model.eval()  # 设置模型为评估模式

    y_true = {'train': [], 'valid': [], 'test': []}  # 真实标签
    y_pred = {'train': [], 'valid': [], 'test': []}  # 预测标签

    pbar = tqdm(total=len(test_loader))  # 使用 tqdm 显示评估进度
    pbar.set_description(f'Evaluating epoch: {epoch:04d}')

    for data in test_loader:  # 遍历测试数据加载器
        data = data.to(device)  # 将数据迁移到设备
        out = model(data.x, data.edge_index, data.edge_attr)  # 获取模型输出

        # 将预测值和真实标签分别添加到对应的列表中
        for split in y_true.keys():
            mask = data[f'{split}_mask']
            y_true[split].append(data.y[mask].cpu())
            y_pred[split].append(out[mask].cpu())

        pbar.update(1)  # 更新进度条

    pbar.close()  # 关闭进度条

    # 计算训练集、验证集和测试集上的 ROC-AUC 分数
    train_rocauc = evaluator.eval({
        'y_true': torch.cat(y_true['train'], dim=0),
        'y_pred': torch.cat(y_pred['train'], dim=0),
    })['rocauc']

    valid_rocauc = evaluator.eval({
        'y_true': torch.cat(y_true['valid'], dim=0),
        'y_pred': torch.cat(y_pred['valid'], dim=0),
    })['rocauc']

    test_rocauc = evaluator.eval({
        'y_true': torch.cat(y_true['test'], dim=0),
        'y_pred': torch.cat(y_pred['test'], dim=0),
    })['rocauc']

    return train_rocauc, valid_rocauc, test_rocauc


# 训练并评估模型，进行 1000 个 epoch 的训练
for epoch in range(1, 1001):
    loss = train(epoch)  # 训练模型
    train_rocauc, valid_rocauc, test_rocauc = test()  # 测试模型
    print(f'Loss: {loss:.4f}, Train: {train_rocauc:.4f}, '
          f'Val: {valid_rocauc:.4f}, Test: {test_rocauc:.4f}')
