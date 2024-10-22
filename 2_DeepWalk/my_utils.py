import torch
from torch.utils.data import Dataset
import pandas as pd
from tqdm import tqdm
import numpy as np
import random


def one_node_random_walk(
    adj_list: list,
    begin_node: int,
    t: int,
):
    """一次随机游走

    Args:
        adj_list (list): 邻接表
        begin_node (int): 最开始的节点
        t (int): 游走的最长长度

    Returns:
        list: 一次随机游走得到的序列
    """
    sequence = []
    current_node = begin_node
    sequence.append(current_node)
    while len(sequence) < t:
        adj_nodes = adj_list[current_node]
        next_index = np.random.choice(np.arange(len(adj_nodes)))
        current_node = adj_nodes[next_index]
        sequence.append(current_node)
    return sequence


class SkipGramHierarchicalSoftmaxDataset(Dataset):
    def __init__(
        self,
        nodes_file: str,
        edges_file: str,
        window_size: int,
        walks_num: int,
        t: int,
        bias: int,
    ) -> None:
        super().__init__()
        self.nodes = pd.read_csv(nodes_file, names=["node"])
        self.edges = pd.read_csv(edges_file, names=["node1", "node2"])
        self.walks_num = walks_num
        self.window_size = window_size
        self.t = t
        self.adj_list = self._get_adj_list()
        self.tree_height = int(np.ceil(np.log2(self.nodes.shape[0])))
        self.data = self._data_process()
        self.bias = bias

    def _data_process(
        self,
    ):
        data = []
        for gamma in tqdm(range(self.walks_num), desc="Random Walking"):
            nodes = self.nodes["node"].tolist().copy()
            random.shuffle(nodes)
            for node in nodes:
                sequence = one_node_random_walk(
                    self.adj_list, begin_node=node, t=self.t
                )
                data += self._gain_data_from_seq(sequence=sequence)
        return data

    def _gain_data_from_seq(self, sequence):

        # 保证两个窗口加上中间节点的长度小于序列的长度
        assert (self.window_size * 2 + 1) < len(sequence)
        data = []
        for i in range(self.window_size, len(sequence) - self.window_size):
            data.append(sequence[i - self.window_size : i + self.window_size + 1])
        return data

    def _get_adj_list(self):
        nodes = self.nodes["node"].tolist()
        adj_list = {node: [] for node in nodes}
        for i in tqdm(range(len(self.edges)), desc="Formating adjacency list"):
            node1 = self.edges["node1"].iloc[i]
            node2 = self.edges["node2"].iloc[i]
            adj_list[node1].append(node2)
            adj_list[node2].append(node1)
        return adj_list

    def _get_tree_info(self, target_idx, tree_height):
        binary_tree_code = format(target_idx, f"0{tree_height}b")
        binary_tree_code = [int(c) for c in binary_tree_code]
        path_nodes = [0]
        c = 0
        for char in binary_tree_code:
            if char == 0:
                c = c * 2 + 1
            else:
                c = c * 2 + 2
            path_nodes.append(c)
        path_nodes.pop(-1)
        assert len(path_nodes) == len(binary_tree_code)
        return binary_tree_code, path_nodes

    def __len__(
        self,
    ):
        return len(self.data)

    def __getitem__(self, index):
        sample = self.data[index]
        sample = [node - 1 for node in sample]
        input_node = sample[self.window_size]
        context_nodes = sample[: self.window_size] + sample[self.window_size + 1 :]
        bin_tree_codes = []  # [s,h]
        path_nodes = []  # [s,h]
        for node in context_nodes:
            b_t_code, p_nodes = self._get_tree_info(node, self.tree_height)
            bin_tree_codes.append(b_t_code)
            path_nodes.append(p_nodes)
        return {
            "input_node": input_node,
            "bin_tree_codes": bin_tree_codes,
            "path_nodes": path_nodes,
        }

    def collate_fn(batch):
        input_nodes = [f["input_node"] for f in batch]
        bin_tree_codes = [f["bin_tree_codes"] for f in batch]
        path_nodes = [f["path_nodes"] for f in batch]
        input_nodes = torch.LongTensor(input_nodes)  # [b]
        bin_tree_codes = torch.tensor(bin_tree_codes)  # [b, s, h]
        path_nodes = torch.tensor(path_nodes)  # [b, s, h]
        return input_nodes, bin_tree_codes, path_nodes
