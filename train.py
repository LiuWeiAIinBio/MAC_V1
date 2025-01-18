from net import MACNet
import torch
import torch.nn as nn
from dataset import MACDataset
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

# 1. 数据
path_fasta_seq = "./data/acrA_mcrA_AA_2.txt"
path_label_file = "./data/acrA_mcrA_label_2.txt"
BATCH_SIZE = 16
train_data = MACDataset("./data/acrA_mcrA_AA_2.txt", "./data/acrA_mcrA_label_2.txt")
train_loader = DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=True)

# 2. 模型
net = MACNet(d_prob=0.5)
net.initialize_weights()

# 3. 损失函数
criterion = nn.BCELoss()

# 4. 优化器
LR = 0.01
optimizer = torch.optim.SGD(net.parameters(), lr=LR, momentum=0.9)
scheduler_lr = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.1, patience=10)

# 5. 训练
MAX_EPOCH = 50
iter_count = 0
total = 0
correct = 0

# 构建 SummaryWriter
writer = SummaryWriter(comment='MAC_train')

for epoch in range(MAX_EPOCH):
    net.train()
    for i, data in enumerate(train_loader):
        iter_count += 1
        # forward
        inputs, labels = data
        outputs = net(inputs)
        # backward
        optimizer.zero_grad()
        loss = criterion(outputs, labels)
        loss.backward()
        # update weights
        optimizer.step()

        # # 边训练，边统计分类情况
        # total += labels.shape[0]
        # outputs = [1 if i > 0.5 else 0 for i in outputs]
        # correct += sum([1 if a == b else 0 for (a, b) in zip(outputs, labels)])
        # accuracy = correct / total  # 各个 epoch 的样本的累加结果

        # 边训练，边统计分类情况
        total = labels.shape[0]
        outputs = [1 if i > 0.5 else 0 for i in outputs]
        correct = sum([1 if a == b else 0 for (a, b) in zip(outputs, labels)])
        accuracy = correct / total  # 单一 batch 的样本的结果

        # 记录 loss 和 accuracy，保存于 event file
        writer.add_scalar("loss", loss.item(), iter_count)  # loss.item() 将张量转换为标量
        writer.add_scalar("accuracy", accuracy, iter_count)

    # 在 epoch 循环下，记录在每个 epoch 中，每个网络层的参数（权值）和参数（权值）的梯度
    for layer_name, layer_param in net.named_parameters():
        writer.add_histogram(layer_name + '_grad', layer_param.grad, epoch)  # 记录参数的梯度
        writer.add_histogram(layer_name + '_data', layer_param.data, epoch)  # 记录参数

    scheduler_lr.step(loss)  # 更新学习率

writer.close()

# 保存模型
torch.save(net.state_dict(), "./model_state_dict.pkl")


# 检验训练后的模型在训练集的表现
if __name__ == "__main__":
    train_data = MACDataset("./data/acrA_mcrA_AA_2.txt", "./data/acrA_mcrA_label_2.txt")
    train_loader = DataLoader(dataset=train_data, batch_size=16, shuffle=True)

    total = 0
    correct = 0

    for i, data in enumerate(train_loader):
        inputs, labels = data
        net.eval()
        outputs = net(inputs)
        total += labels.shape[0]
        outputs = [1 if i > 0.5 else 0 for i in outputs]
        correct += sum([1 if a == b else 0 for (a, b) in zip(outputs, labels)])
        accuracy = correct / total
        print(f"对前 {i} 个 batch 样本的预测准确率为：{accuracy}")
