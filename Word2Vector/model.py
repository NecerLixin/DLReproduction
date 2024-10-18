import torch
import torch.nn as nn
import torch.nn.functional as F


class Word2VectorModelHierarchicalSoftmax(nn.Module):
    def __init__(
        self, words_num: int, embedding_dim: int, no_leaf_nodes_num: int
    ) -> None:
        super().__init__()
        self.words_num = words_num
        self.embedding_dim = embedding_dim
        self.no_leaf_nodes_num = no_leaf_nodes_num

        self.embedding = nn.Linear(
            in_features=words_num,
            out_features=embedding_dim,
            bias=False,
        )
        self.cls = nn.Parameter(torch.randn([no_leaf_nodes_num, embedding_dim]))
        self.criterion = nn.BCEWithLogitsLoss()

    def forward(self, inputs_vector, path_nodes_indices, huffman_codes):
        # inputs_vector [b, n]
        # labels [[node index,],...]
        x = self.embedding(inputs_vector)
        batch_size = inputs_vector.shape[0]
        loss = torch.zeros(1).to(x)
        for i in range(batch_size):
            path_node_index = torch.LongTensor(path_nodes_indices[i]).to(x)
            path_vector = self.cls[path_nodes_indices[i]]
            logits = x[i] @ path_vector.T
            # prob = torch.sigmoid(logits)
            """
                $$
                \mathcal{L}(w,j)=(1-d_{j}^{w})\cdot\log[\sigma(\mathbf{x}_{w}^{\top}\theta_{j-1}^{w})]+d_{j}^{w}\cdot\log[1-\sigma(\mathbf{x}_{w}^{\top}\theta_{j-1}^{w})]
                $$
            """
            huffman_code = torch.tensor(huffman_codes[i]).to(x)
            loss = loss + F.binary_cross_entropy_with_logits(logits, huffman_code).to(x)
        loss = loss / batch_size
        return loss
