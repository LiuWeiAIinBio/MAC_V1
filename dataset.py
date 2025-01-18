"""
@file name  : dataset.py
@author     : LiuWei https://github.com/LiuWeiAIinBio
@date       : 2024-01-11
@brief      : 数据集的 Dataset 定义
"""
import torch
from torch.utils.data import Dataset
import embedding


class MACDataset(Dataset):
    def __init__(self, path_AAfasta_seq, path_label_file):
        """
        :param path_AAfasta_seq: 氨基酸序列文本（fasta 格式）的路径
        :param path_label_file: label 文本路径
        """
        self.data = self.get_data(path_AAfasta_seq, path_label_file)

    def __getitem__(self, index):
        x, y = self.data[index]
        x = torch.tensor(x, dtype=torch.float32)
        y = torch.tensor(y, dtype=torch.float32)
        return x, y

    def __len__(self):
        return len(self.data)

    @staticmethod
    def get_data(path_AAfasta_seq, path_label_file):
        data = embedding.prepare_data(path_AAfasta_seq, path_label_file)
        blosumMat = embedding.creat_blosumMat(embedding.blosum62, columns=embedding.amino, index=embedding.amino)
        features, labels = embedding.data2blosumMat(data, blosumMat)

        data = []
        for i in range(features.shape[0]):
            x = features[i]
            y = labels[i]
            data.append((x, y))

        return data


if __name__ == "__main__":
    path_AAfasta_seq = "./data/acrA_mcrA_AA_2.txt"
    path_label_file = "./data/acrA_mcrA_label_2.txt"

    my_dataset = MACDataset(path_AAfasta_seq, path_label_file)
    print(my_dataset.data)
