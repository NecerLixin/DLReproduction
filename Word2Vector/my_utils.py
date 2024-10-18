import heapq
import json
from typing import Optional
from torch.utils.data import Dataset
import json
import torch


class HuffmanNode:
    def __init__(
        self,
        char: Optional[str],
        freq: int,
    ) -> None:
        self.char = char
        self.freq = freq
        self.right = None
        self.left = None

    # 最小堆比较规则
    def __lt__(self, other):
        return self.freq < other.freq


def construct_Huffman_tree(
    words_freq: dict,
):
    huffman_heap = [HuffmanNode(key, val) for key, val in words_freq.items()]
    heapq.heapify(huffman_heap)

    while len(huffman_heap) > 1:
        # 取出两个频率最小的节点
        node1 = heapq.heappop(huffman_heap)
        node2 = heapq.heappop(huffman_heap)
        # 构造新的节点，频率为两个节点之和
        node_comb = HuffmanNode(None, node1.freq + node2.freq)
        node_comb.left = node1
        node_comb.right = node2
        heapq.heappush(huffman_heap, node_comb)
    return huffman_heap[0]


def get_Huffman_codes(
    huffman_node: HuffmanNode,
    current_code: str,
    codes: dict,
    no_leaf_codes: set,
):
    # 非叶子节点
    if huffman_node.char is not None:
        codes[huffman_node.char] = current_code
        # return
    else:
        no_leaf_codes.add(current_code)
    # 左0
    if huffman_node.left:
        get_Huffman_codes(huffman_node.left, current_code + "0", codes, no_leaf_codes)
    # 右1
    if huffman_node.right:
        get_Huffman_codes(huffman_node.right, current_code + "1", codes, no_leaf_codes)

    return codes, no_leaf_codes


def get_path_node_ids(huffman_code: str, no_leaf_code2index: dict) -> list:
    """根据哈夫曼编码返回经过的节点的编号

    Args:
        huffman_code (list): 哈夫曼编码

    Returns:
        list: 经过的节点的编号

    Explanation:
        哈夫曼是前缀编码，可以依次根据前缀确定经过了哪些节点
    """
    node_ids = []
    for i in range(len(huffman_code)):
        node_ids.append(no_leaf_code2index[huffman_code[:i]])
    return node_ids


class CBOWDataset(Dataset):
    def __init__(
        self,
        file_path: str,
        word2id_file: str,
        huffman_code_file: str,
        no_leaf_code2index_file: str,
        window_size: int = 5,
    ) -> None:
        super().__init__()
        self.window_size = 5
        with open(file_path, "r") as f:
            self.text = f.read().split()
        self.len = len(self.text) - window_size + 1
        self.word2id = json.load(open(word2id_file))
        self.id2word = {val: key for key, val in self.word2id.items()}
        self.huffman_code = json.load(open(huffman_code_file))
        self.huffman_code = {
            self.word2id[key]: val for key, val in self.huffman_code.items()
        }
        self.no_leaf_code2index = json.load(open(no_leaf_code2index_file))
        self.words_num = len(self.word2id)

        ## 根据滑动窗口构造数据
        self.data = []
        for i in range(self.len):
            t = self.text[i : i + self.window_size]
            t = [self.word2id[s] for s in t]
            self.data.append(t)

    def __getitem__(self, index):
        item = self.data[index].copy()
        target_id = item.pop(self.window_size // 2)  # item 将中心点去掉了
        input_vector = torch.zeros(self.words_num)
        input_vector[item] = 1
        path_nodes_index = get_path_node_ids(
            self.huffman_code[target_id], self.no_leaf_code2index
        )
        huffman_code = self.huffman_code[target_id]
        return {
            "input_vector": input_vector,
            "path_nodes_index": path_nodes_index,
            "huffman_code": huffman_code,
        }

    def __len__(self):
        return len(self.text) - self.window_size + 1

    def collate_fn(batch):
        inputs_vector = [f["input_vector"] for f in batch]
        path_nodes_indices = [f["path_nodes_index"] for f in batch]
        huffman_codes = [[int(t) for t in f["huffman_code"]] for f in batch]
        inputs_vector = torch.stack(inputs_vector, dim=0)
        return inputs_vector, path_nodes_indices, huffman_codes


if __name__ == "__main__":
    words_freq = json.load(open("Word2Vector/data/words_freq.json", "r"))
    huffman_root = construct_Huffman_tree(words_freq)
    huffman_codes = dict()
    no_leaf_codes = set()
    huffman_codes, no_leaf_codes = get_Huffman_codes(
        huffman_root, "", huffman_codes, no_leaf_codes
    )
    no_leaf_codes = {code: i for i, code in enumerate(no_leaf_codes)}
    with open("Word2Vector/data/huffman_codes.json", "w") as f:
        json.dump(
            huffman_codes,
            f,
            ensure_ascii=False,
        )
    with open("Word2Vector/data/no_leaf_code2index.json", "w") as f:
        json.dump(no_leaf_codes, f, ensure_ascii=False)
    print(len(words_freq))
    print(len(huffman_codes))
    if len(huffman_codes) + len(no_leaf_codes) == 2 * len(huffman_codes) - 1:
        print(True)
    else:
        print(False)
