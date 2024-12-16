import time
import torch
import torch.nn.functional as F
from torch.nn import ModuleList
from tqdm import tqdm

from torch_geometric.datasets import Reddit
from torch_geometric.loader import ClusterData, ClusterLoader, NeighborLoader
from torch_geometric.nn import SAGEConv

dataset = Reddit('../data/Reddit')
data = dataset[0]

cluster_data = ClusterData(data, num_parts=1500, recursive=False, 
                           save_dir=dataset.processed_dir)
train_loader = ClusterLoader(cluster_data, batch_size=20, 
                             shuffle=True, num_workers=12)

subgraph_loader = NeighborLoader(data, num_neighbors=[-1], 
                                 atch_size=1024, shuffle=False, num_workers=12)

class Net(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.convs = ModuleList(
            [SAGEConv(in_channels, 128),
             SAGEConv(128, out_channels)])

    def forward(self, x, edge_index):
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)  # 通过每一层 GCN 操作
            if i != len(self.convs) - 1:
                x = F.relu(x)  # 激活函数
                x = F.dropout(x, p=0.5, training=self.training)  # Dropout
        return F.log_softmax(x, dim=-1)  # 使用 softmax 进行分类

    def inference(self, x_all):
        pbar = tqdm(total=x_all.size(0) * len(self.convs))
        pbar.set_description('Evaluating')

        for i, conv in enumerate(self.convs):
            xs = []
            for batch in subgraph_loader:
                edge_index = batch.edge_index.to(device)
                x = x_all[batch.n_id].to(device)
                x_target = x[:batch.batch_size]
                x = conv((x, x_target), edge_index)  # 图卷积计算
                if i != len(self.convs) - 1:
                    x = F.relu(x)  # 激活函数
                xs.append(x.cpu())  # 记录输出

                pbar.update(batch.batch_size)

            x_all = torch.cat(xs, dim=0)  # 拼接所有子图的输出

        pbar.close()

        return x_all

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = Net(dataset.num_features, dataset.num_classes).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.005)

def train():
    model.train()

    total_loss = total_nodes = 0
    for batch in train_loader:
        batch = batch.to(device)
        optimizer.zero_grad()  # 清空梯度
        out = model(batch.x, batch.edge_index)  # 前向传播
        loss = F.nll_loss(out[batch.train_mask], batch.y[batch.train_mask])  # 计算损失
        loss.backward()  # 反向传播
        optimizer.step()  # 更新参数

        nodes = batch.train_mask.sum().item()
        total_loss += loss.item() * nodes  # 累积损失
        total_nodes += nodes  # 累积训练节点数

    return total_loss / total_nodes  # 返回平均损失
  
@torch.no_grad()
def test():  # 推理时应该对全图进行评估
    model.eval()  # 设置模型为评估模式

    out = model.inference(data.x)  # 计算全图节点表示
    y_pred = out.argmax(dim=-1)  # 获取预测的类别

    accs = []
    for mask in [data.train_mask, data.val_mask, data.test_mask]:
        correct = y_pred[mask].eq(data.y[mask]).sum().item()  # 计算正确预测的数量
        accs.append(correct / mask.sum().item())  # 计算准确率
    return accs

times = []
for epoch in range(1, 31):
    start = time.time()
    loss = train()  # 训练模型
    if epoch % 5 == 0:  # 每隔 5 个 epoch 进行评估
        train_acc, val_acc, test_acc = test()  # 测试模型准确率
        print(f'Epoch: {epoch:02d}, Loss: {loss:.4f}, Train: {train_acc:.4f}, '
              f'Val: {val_acc:.4f}, test: {test_acc:.4f}')
    else:
        print(f'Epoch: {epoch:02d}, Loss: {loss:.4f}')
    times.append(time.time() - start)  # 记录每个 epoch 的时间
print(f"Median time per epoch: {torch.tensor(times).median():.4f}s")

