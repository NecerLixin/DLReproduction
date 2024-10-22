from my_utils import SkipGramHierarchicalSoftmaxDataset
import torch
from torch.utils.data import DataLoader
from model import SkipGramHierarchicalSoftmaxModel
from tqdm import tqdm
import argparse
import pandas as pd

device = None


def train(
    args,
    model: SkipGramHierarchicalSoftmaxModel,
    train_dataset: SkipGramHierarchicalSoftmaxDataset,
):
    dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        collate_fn=SkipGramHierarchicalSoftmaxDataset.collate_fn,
    )
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr)
    for epoch in range(args.epochs):
        total_loss = 0
        for batch in tqdm(dataloader, desc="Training"):
            inputs = {
                "input_nodes": batch[0].to(device),
                "path_nodes": batch[2].to(device),
                "tree_codes": batch[1].to(device),
            }
            optimizer.zero_grad()
            loss = model(**inputs)
            loss.backward()
            optimizer.step()
            total_loss += loss
        print(f"Epoch {epoch+1}, Loss: {total_loss/len(dataloader)}")
        torch.save(model.state_dict(), args.save_path)


def main():
    parser = argparse.ArgumentParser(description="Training a bert model.")
    parser.add_argument(
        "--epochs",
        type=int,
        default=30,
        help="Number of training epochs.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=50,
        help="Training batch size.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Device used to training model",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=0.025,
        help="Learning rate.",
    )
    parser.add_argument(
        "--save_path",
        type=str,
        default="2_DeepWalk/model_save/model.pth",
        help="Path to save model",
    )
    parser.add_argument(
        "--nodes_file",
        type=str,
        default="2_DeepWalk/BlogCatalog-dataset/nodes.csv",
        help="Nodes file path.",
    )
    parser.add_argument(
        "--edges_file",
        type=str,
        default="2_DeepWalk/BlogCatalog-dataset/edges.csv",
        help="Edges file path.",
    )
    parser.add_argument(
        "--groups_file",
        type=str,
        default="2_DeepWalk/BlogCatalog-dataset/groups.csv",
        help="Groups file path.",
    )
    parser.add_argument(
        "--group_edges_file",
        type=str,
        default="2_DeepWalk/BlogCatalog-dataset/group-edges.csv",
        help="Nodes labels file path.",
    )
    parser.add_argument(
        "--window_size",
        type=int,
        default=10,
        help="Nodes labels file path.",
    )
    parser.add_argument(
        "--walks_num",
        type=int,
        default=1,
        help="Num of random walks",
    )
    parser.add_argument(
        "--t",
        type=int,
        default=40,
        help="Length of random walk",
    )
    parser.add_argument(
        "--embedding_dim",
        type=int,
        default=128,
        help="Length of random walk",
    )

    args = parser.parse_args()
    global device
    device = torch.device(args.device)
    dataset = SkipGramHierarchicalSoftmaxDataset(
        nodes_file=args.nodes_file,
        edges_file=args.edges_file,
        window_size=args.window_size,
        walks_num=args.walks_num,
        t=args.t,
        bias=1,
    )

    nodes = pd.read_csv(args.nodes_file, names=["node"])
    nodes_num = nodes.shape[0]
    model = SkipGramHierarchicalSoftmaxModel(
        nodes_num=nodes_num,
        embedding_dim=args.embedding_dim,
    )
    train(args=args, model=model, train_dataset=dataset)


if __name__ == "__main__":
    main()
