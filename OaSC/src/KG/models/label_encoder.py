import os
import json
import sys
import KG.models
import KG.models.templates
from KG.models.templates import gcn, gat, rgcn, trgcn, lstm,rgcn_yolo,rgcn_yolo_s,rgcn_yolo_m,rgcn_faster
from KG.class_encoders.auto_gnn import AutoGNN
from KG.class_encoder.dgp import DGP
from KG.class_encoder.sgcn import SGCN
from KG.class_encoder.gcnz import GCNZ

GNN_CONFIGS = {
    "gcn": gcn,
    "gat": gat,
    "rgcn": rgcn,
    "rgcn_yolo": rgcn_yolo,
     "rgcn_yolo_s": rgcn_yolo_s,
      "rgcn_yolo_m": rgcn_yolo_m,
       "rgcn_faster": rgcn_faster,
    "trgcn": trgcn,
    "lstm": lstm,
}


DIR_PATH = os.path.dirname(os.path.realpath(__file__))


def dgp(options):
    graph = json.load(
        open(os.path.join(DIR_PATH, "../data/dense_graph.json"), "r")
    )
    wnids = graph["wnids"]
    n = len(wnids)

    edges_set = graph["edges_set"]

    print("edges_set", [len(l) for l in edges_set])

    lim = 4
    for i in range(lim + 1, len(edges_set)):
        edges_set[lim].extend(edges_set[i])
    edges_set = edges_set[: lim + 1]
    print("edges_set", [len(l) for l in edges_set])

    hidden_layers = "d2048,d"
    gcn = DGP(n, edges_set, 300, 2049, hidden_layers)

    save_path = os.path.join(
        DIR_PATH, "../save/dgp_seed_{}".format(options["seed"])
    )
    return gcn, save_path


def sgcn(options):
    graph = json.load(
        open(os.path.join(DIR_PATH, "../data/induced_graph.json"), "r")
    )
    wnids = graph["wnids"]
    n = len(wnids)
    edges = graph["edges"]

    edges = edges + [(v, u) for (u, v) in edges]
    edges = edges + [(u, u) for u in range(n)]

    hidden_layers = "d2048,d"
    gcn = SGCN(n, edges, 300, 2049, hidden_layers, options["device"])

    save_path = os.path.join(
        DIR_PATH, "../save/sgcn_seed_{}".format(options["seed"])
    )

    return gcn, save_path


def gcnz(options):
    graph = json.load(
        open(os.path.join(DIR_PATH, "../data/induced_graph.json"), "r")
    )
    wnids = graph["wnids"]
    n = len(wnids)
    edges = graph["edges"]

    edges = edges + [(v, u) for (u, v) in edges]
    edges = edges + [(u, u) for u in range(n)]

    hidden_layers = "d2048,d"
    gcn = GCNZ(n, edges, 300, 2049, hidden_layers, options["device"])

    save_path = os.path.join(
        DIR_PATH, "../save/resnet_gcnz_seed_{}".format(options["seed"])
    )

    return gcn, save_path


def get_label_encoder(label_encoder_type, options):
    if label_encoder_type == "trgcn":
        pass

    if label_encoder_type in ["gcn", "gat", "rgcn", "lstm", "trgcn","rgcn_yolo","rgcn_yolo_m","rgcn_yolo_s","rgcn_faster"]:
        
        if( label_encoder_type not in ['rgcn_yolo']):
            model = AutoGNN(GNN_CONFIGS[label_encoder_type])
            
        else:
            model=[None,None,None]
            model[0] = AutoGNN(GNN_CONFIGS["rgcn_yolo_s"])
            model[1] = AutoGNN(GNN_CONFIGS["rgcn_yolo_m"])   
            model[2] = AutoGNN(GNN_CONFIGS["rgcn_yolo"]) 
        save_path = os.path.join(
            DIR_PATH, f'../save/{label_encoder_type}_seed_{options["seed"]}'
        )
        return model, save_path

    if label_encoder_type == "sgcn":
        return sgcn(options)

    if label_encoder_type == "gcnz":
        return gcnz(options)

    if label_encoder_type == "dgp":
        return dgp(options)
