import torch
import torch.nn as nn
import torch.nn.functional as F


class SkipGramHierarchicalSoftmaxModel(nn.Module):
    def __init__(self, nodes_num: int, embedding_dim: int) -> None:
        super().__init__()
        self.nodes_num = nodes_num
        self.embedding_size = embedding_dim
        self.embedding = nn.Embedding(
            num_embeddings=nodes_num,
            embedding_dim=embedding_dim,
        )
        self.cls = nn.Linear(
            in_features=embedding_dim,
            out_features=nodes_num * 2,
        )

    def forward(self, input_nodes, path_nodes: torch.Tensor, tree_codes: torch.Tensor):

        ############################################
        # 由于没有使用哈夫曼树，所以就没使用MASK了 #
        ############################################
        x = self.embedding(input_nodes)  # [b,e]
        B = x.shape[0]
        logits = self.cls(x)  # [b,n]
        # path_nodes [b, s, h]
        # tree_nodes [b, s, h]
        logits_selected = logits[
            torch.arange(0, B).unsqueeze(-1).unsqueeze(-1), path_nodes
        ]  # [b,h]
        loss = F.binary_cross_entropy_with_logits(
            logits_selected.view(B, -1), tree_codes.view(B, -1).float()
        )
        return loss
