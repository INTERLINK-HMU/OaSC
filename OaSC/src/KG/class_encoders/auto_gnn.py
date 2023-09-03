import torch
import torch.nn as nn
from KG.class_encoders.gat_model import GATConv
from KG.class_encoders.gcn_model import GCNConv
from KG.class_encoders.lstm_model import LSTMConv
from KG.class_encoders.rgcn_model import RGCNConv
from KG.class_encoders.trgcn_model import TrGCNConv
from KG.knowledge_graph.kg import KG

GNN_CONV = {
    "gcn": GCNConv,
    "gat": GATConv,
    "rgcn": RGCNConv,
    "trgcn": TrGCNConv,
    "lstm": LSTMConv,
     "rgcn_yolo": RGCNConv,
      "rgcn_yolo_m": RGCNConv,
       "rgcn_yolo_s": RGCNConv,
}


class AutoGNN(nn.Module):
    def __init__(self, config: dict):
        """Auto class to create gnn convolutional layers. This class
        takes in a config dictionary which contains all the arugments
        of the combine and aggregators modules in their gnn convolutional
        layers.

        Args:
            config (dict): the arguments for the gnn type.
        """
        super().__init__()

        self.input_dim = config["input_dim"]
        self.output_dim = config["output_dim"]
        self.gnn_type = config["type"]

        gnn_conv = GNN_CONV[self.gnn_type]

        gnn_modules = []
        current_conv = None
        for params in config["gnn"]:
            params["features"] = current_conv

            conv = gnn_conv(**params)

            gnn_modules.append(conv)
            current_conv = conv

            # TODO: check if the output dim of this layer
            # is same as the input dim of the next layer

        self.add_module("conv", gnn_modules[-1])

    def forward(self, node_idx: torch.tensor, kg: KG):
        """Forward function for the attention aggregator.

        Args:
            nodes (torch.tensor): nodes in the knowledge graph.
            kg (KG): knowledge graph (ConceptNet or WordNet).

        Returns:
            torch.Tensor: features/embeddings for nodes.
        """
        return self.conv(node_idx, kg)
