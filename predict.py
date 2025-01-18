from net import MACNet
import torch
from dataset import MACDataset
from torch.utils.data import DataLoader
import argparse


def predict(path_AAfasta_seq, path_label_file):
    # 加载模型
    net = MACNet(d_prob=0.5)
    state_dict_load = torch.load("./model_state_dict.pkl")
    net.load_state_dict(state_dict_load)


    # 推断
    sample_data = MACDataset(path_AAfasta_seq, path_label_file)
    sample_loader = DataLoader(dataset=sample_data, batch_size=1, shuffle=False)

    total = 0
    correct = 0

    for i, data in enumerate(sample_loader):
        inputs, labels = data
        total += labels.shape[0]

        if int(labels.item()) == 0:
            labels = "MCR"
        else:
            labels = "ACR"

        net.eval()
        outputs = net(inputs)
        outputs = 1 if outputs > 0.5 else 0

        if outputs == 0:
            outputs = "MCR"
        else:
            outputs = "ACR"

        predict = "True" if outputs == labels else "False"

        print(f"第 {i} 个样本为：{labels}, predict 为 {outputs}, 预测 {predict}")
        correct += 1 if outputs == labels else 0

    accuracy = correct / total
    print(f"对全部样本的预测准确率为：{accuracy}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--path_AAfasta_seq", type=str)
    parser.add_argument("--path_label_file", type=str)
    args = parser.parse_args()

    predict(path_AAfasta_seq=args.path_AAfasta_seq, path_label_file=args.path_label_file)
