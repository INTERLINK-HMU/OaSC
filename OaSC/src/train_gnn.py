import argparse
import copy
import json
import os
import random
import torch
import torch.nn.functional as F

import sys


from torchvision.models import resnet101
from KG.knowledge_graph.conceptnet import ConceptNetKG

from KG.models.label_encoder import get_label_encoder
from KG.utils.common import l2_loss, mask_l2_loss, pick_vectors, set_seed

from torchvision.models.detection.faster_rcnn import fasterrcnn_resnet50_fpn

DIR_PATH = os.path.dirname(os.path.realpath(__file__))


def predict(model, graph_path, dataset):
    print("generating graph embeddings for {}".format(dataset))
    kg = ConceptNetKG.load_from_disk(graph_path)
    concepts_path = os.path.join(DIR_PATH, f"data/kgs_concepts/{dataset}_concepts.txt")
    concepts = [line.strip() for line in open(concepts_path)]
    print(graph_path,concepts)
    concept_idx = torch.tensor(kg.get_node_ids(concepts)).to(options["device"])
    print(concept_idx)
    kg.to(options["device"])

    model.eval()
    pred_vectors = model(concept_idx, kg)

    return pred_vectors


def get_apy_preds(pred_obj):
    pred_wnids = pred_obj["wnids"]
    pred_vectors = pred_obj["pred"].cpu()
    pred_dic = dict(zip(pred_wnids, pred_vectors))
    with open(os.path.join(DIR_PATH, "materials/apy_wnid.json")) as fp:
        apy_wnid = json.load(fp)

    train_wnids = apy_wnid["train"]
    test_wnids = apy_wnid["test"]

    pred_vectors = pick_vectors(
        pred_dic, train_wnids + test_wnids, is_tensor=True
    )

    return pred_vectors


def get_awa_preds(pred_obj):
    awa2_split = json.load(
        open(os.path.join(DIR_PATH, "materials/awa2-split.json"), "r")
    )
    train_wnids = awa2_split["train"]
    print(train_wnids)
    test_wnids = awa2_split["test"]

    pred_wnids = pred_obj["wnids"]
    pred_vectors = pred_obj["pred"].cpu()
    pred_dic = dict(zip(pred_wnids, pred_vectors))

    pred_vectors = pick_vectors(
        pred_dic, train_wnids + test_wnids, is_tensor=True
    )
    return pred_vectors



def get_osdd_preds(pred_obj):
   
    #train_wnids = awa2_split["train"]
    #test_wnids = awa2_split["test"]
    import pandas as pd
    import os.path as osp
    image_data = pd.read_csv(
        osp.join(DIR_PATH, "datasets/osdd/image_label_train.csv")
    )
    
    image_data_test = pd.read_csv(
        osp.join(DIR_PATH, "datasets/osdd/image_label_train.csv")
    )
    train_wnids =image_data.iloc[:, 0]
   # print(train_wnids)
    test_wnids =image_data_test.iloc[:, 0]
    pred_wnids = pred_obj["wnids"]
    pred_vectors = pred_obj["pred"].cpu()
    pred_dic = dict(zip(pred_wnids, pred_vectors))

    pred_vectors = pick_vectors(
        pred_dic, train_wnids + test_wnids, is_tensor=True
    )
    return pred_vectors



def get_osdd__ir_preds(pred_obj):
       
    #train_wnids = awa2_split["train"]
    #test_wnids = awa2_split["test"]
    import pandas as pd
    import os.path as osp
    image_data = pd.read_csv(
        osp.join(DIR_PATH, "datasets/osdd_ir/image_label_train.csv")
    )
    
    image_data_test = pd.read_csv(
        osp.join(DIR_PATH, "datasets/osdd_ir/image_label_train.csv")
    )
    train_wnids =image_data.iloc[:, 0]
    
    test_wnids =image_data_test.iloc[:, 0]
    pred_wnids = pred_obj["wnids"]
    pred_vectors = pred_obj["pred"].cpu()
    pred_dic = dict(zip(pred_wnids, pred_vectors))

    pred_vectors = pick_vectors(
        pred_dic, train_wnids + test_wnids, is_tensor=True
    )
    return pred_vectors


def train_baseline_model(model, fc_vectors, device):
    model.to(device)

    graph = json.load(
        open(os.path.join(DIR_PATH, "data/dense_graph.json"), "r")
    )
    wnids = graph["wnids"]
    word_vectors = torch.tensor(graph["vectors"]).to(device)
    word_vectors = F.normalize(word_vectors)

    print("word vectors:", word_vectors.shape)
    print("fc vectors:", fc_vectors.shape)

    optimizer = torch.optim.Adam(
        model.parameters(), lr=0.001, weight_decay=0.0005
    )

    v_train, v_val = 0.95, 0.05
    n_trainval = len(fc_vectors)
    n_train = round(n_trainval * (v_train / (v_train + v_val)))
    print("num train: {}, num val: {}".format(n_train, n_trainval - n_train))

    tlist = list(range(len(fc_vectors)))
    random.shuffle(tlist)

    trlog = {}
    trlog["train_loss"] = []
    trlog["val_loss"] = []
    trlog["min_loss"] = 0
    best_model = None

    for epoch in range(1, args.max_epoch + 1):
        model.train()
        output_vectors = model(word_vectors)
        loss = mask_l2_loss(output_vectors, fc_vectors, tlist[:n_train])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        model.eval()
        output_vectors = model(word_vectors)
        train_loss = mask_l2_loss(
            output_vectors, fc_vectors, tlist[:n_train]
        ).item()

        if v_val > float(0):
            val_loss = mask_l2_loss(
                output_vectors, fc_vectors, tlist[n_train:]
            ).item()
            loss = val_loss
        else:
            val_loss = 0
            loss = train_loss
        print(
            "epoch {}, train_loss={:.4f}, val_loss={:.4f}".format(
                epoch, train_loss, val_loss
            )
        )

        pred_obj = {"wnids": wnids, "pred": output_vectors}

        if trlog["val_loss"]:
            min_val_loss = min(trlog["val_loss"])
            if val_loss < min_val_loss:
                best_model = copy.deepcopy(model.state_dict())
        else:
            best_model = copy.deepcopy(model.state_dict())

        trlog["train_loss"].append(train_loss)
        trlog["val_loss"].append(val_loss)

    model.load_state_dict(best_model)
    return model, pred_obj, trlog


def train_gnn_model(model, fc_vectors, device, options):
    kg = ConceptNetKG.load_from_disk(options["ilsvrc_graph_path"])
    kg.to(options["device"])

    concepts_path = os.path.join(DIR_PATH, "datasets/ilsvrc_concepts.txt")
    concepts = [line.strip() for line in open(concepts_path)]
    ilsvrc_idx = torch.tensor(kg.get_node_ids(concepts)).to(options["device"])



    model.to(device)

    optimizer = torch.optim.Adam(
        model.parameters(), lr=0.001, weight_decay=0.0005
    )

    v_train, v_val = map(float, options["trainval"].split(","))
    n_trainval = len(fc_vectors)
    n_train = round(n_trainval * (v_train / (v_train + v_val)))
    print("num train: {}, num val: {}".format(n_train, n_trainval - n_train))

    tlist = list(range(len(fc_vectors)))
    random.shuffle(tlist)

    trlog = {}
    trlog["train_loss"] = []
    trlog["val_loss"] = []
    trlog["min_loss"] = 0
    num_w = fc_vectors.shape[0]
    best_model = None
   
    for epoch in range(1, options["num_epochs"] + 1):
        model.train()
        for i, start in enumerate(range(0, n_train, 100)):
            end = min(start + 100, n_train)
            indices = tlist[start:end]
            output_vectors = model(ilsvrc_idx[indices], kg)
            loss = l2_loss(output_vectors, fc_vectors[indices])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        model.eval()
        output_vectors = torch.empty(num_w, 2049, device=device)
        # if(args.label_encoder_type=='rgcn_yolo'):
        #     output_vectors = torch.empty(num_w, 1025, device=device)
        with torch.no_grad():
            for start in range(0, num_w, 100):
                end = min(start + 100, num_w)
                output_vectors[start:end] = model(ilsvrc_idx[start:end], kg)

        train_loss = mask_l2_loss(
            output_vectors, fc_vectors, tlist[:n_train]
        ).item()
       
        if v_val > float(0):
            val_loss = mask_l2_loss(
                output_vectors, fc_vectors, tlist[n_train:]
            ).item()
            loss = val_loss
        else:
            val_loss = 0
            loss = train_loss

        print(
            "epoch {}, train_loss={:.4f}, val_loss={:.4f}".format(
                epoch, train_loss, val_loss
            )
        )

        # check if I need to save the model
        if trlog["val_loss"]:
            min_val_loss = min(trlog["val_loss"])
            if val_loss < min_val_loss:
                best_model = copy.deepcopy(model.state_dict())
        else:
            best_model = copy.deepcopy(model.state_dict())

        trlog["train_loss"].append(train_loss)
        trlog["val_loss"].append(val_loss)

    model.load_state_dict(best_model)
    return model, trlog


def train_gnn_model_imgnet(model, fc_vectors, device, options):
    kg = ConceptNetKG.load_from_disk(options["ilsvrc_graph_path"])
    kg.to(options["device"])

    concepts_path ="data/kgs_concepts/ilsvrc_concepts.txt"
    concepts = [line.strip() for line in open(concepts_path)]
    ilsvrc_idx = torch.tensor(kg.get_node_ids(concepts)).to(options["device"])



    model.to(device)

    optimizer = torch.optim.Adam(
        model.parameters(), lr=0.001, weight_decay=0.0005
    )

    v_train, v_val = map(float, options["trainval"].split(","))
    n_trainval = len(fc_vectors)
    n_train = round(n_trainval * (v_train / (v_train + v_val)))
    print("num train: {}, num val: {}".format(n_train, n_trainval - n_train))

    tlist = list(range(len(fc_vectors)))
    random.shuffle(tlist)

    trlog = {}
    trlog["train_loss"] = []
    trlog["val_loss"] = []
    trlog["min_loss"] = 0
    num_w = fc_vectors.shape[0]
    best_model = None
    fc_vectors=fc_vectors.to(options["device"])
    for epoch in range(1, options["num_epochs"] + 1):
        model.train()
        for i, start in enumerate(range(0, n_train, 100)):
            end = min(start + 100, n_train)
            indices = tlist[start:end]
            output_vectors = model(ilsvrc_idx[indices], kg)
            loss = l2_loss(output_vectors, fc_vectors[indices])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        model.eval()
        output_vectors = torch.empty(num_w, 2049, device=device)
        # if(args.label_encoder_type=='rgcn_yolo'):
        #     output_vectors = torch.empty(num_w, 1025, device=device)
        with torch.no_grad():
            for start in range(0, num_w, 100):
                end = min(start + 100, num_w)
                output_vectors[start:end] = model(ilsvrc_idx[start:end], kg)

        train_loss = mask_l2_loss(
            output_vectors, fc_vectors, tlist[:n_train]
        ).item()
       
        if v_val > float(0):
            val_loss = mask_l2_loss(
                output_vectors, fc_vectors, tlist[n_train:]
            ).item()
            loss = val_loss
        else:
            val_loss = 0
            loss = train_loss

        print(
            "epoch {}, train_loss={:.4f}, val_loss={:.4f}".format(
                epoch, train_loss, val_loss
            )
        )

        # check if I need to save the model
        if trlog["val_loss"]:
            min_val_loss = min(trlog["val_loss"])
            if val_loss < min_val_loss:
                best_model = copy.deepcopy(model.state_dict())
        else:
            best_model = copy.deepcopy(model.state_dict())

        trlog["train_loss"].append(train_loss)
        trlog["val_loss"].append(val_loss)

    model.load_state_dict(best_model)
    return model, trlog




def get_fc():
    resnet = resnet101(pretrained=True)
    with torch.no_grad():
        b = resnet.fc.bias.detach()
        w = resnet.fc.weight.detach()
        print(b.shape)
        print(w.shape)
        fc_vectors = torch.cat((w, b.unsqueeze(-1)), dim=1)
        print
        
    return F.normalize(fc_vectors)


def get_fc_faster():
    #fasterrcnn_resnet50_fpn
    
    import numpy as np
    #config_file_path='/home/philippos/PycharmProjects/51/yolov4.yml'
   # print(yolov4.info())
    with torch.no_grad():
       
        faster = fasterrcnn_resnet50_fpn(num_classes=9)
        l = faster.state_dict()

       # print(l.keys())
       
        # for layer in yolov4.children():
        #     if isinstance(layer, nn.Linear):
        #         print(layer.state_dict()['weight'])
        #         print(layer.state_dict()['bias'])
        #     print(2312)
      #  b = yolov4.model[-1].detach()
       # w = yolov4[-1].weights.detach()
      #  print(b.shape)
      #  print(w.shape)
        
        
      
        b = l['roi_heads.box_predictor.cls_score.bias'].detach()
        w = l['roi_heads.box_predictor.cls_score.weight'].detach()
        w=np.squeeze(w)
        # print(b.shape)
        # print(w.shape)
        fc_vectors = torch.cat((w, b.unsqueeze(-1)), dim=1)
        print(fc_vectors.shape)
        fc_vectors=F.normalize(fc_vectors)
    return (fc_vectors)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--label_encoder_type", help="label encoder")
    parser.add_argument("--trainval", default="0.95,0.05")
    parser.add_argument("--batch-size", default=100, type=int)
    parser.add_argument("--max-epoch", type=int, default=1000)
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--dataset", default='osdd')
    parser.add_argument("--mode", default='train')
    args = parser.parse_args()
    
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")

    print("device : ", device)

    set_seed(int(args.seed))

    ilsvrc_graph = os.path.join(DIR_PATH, "data/subgraphs/ilsvrc_graph")
    apy_graph = os.path.join(DIR_PATH, "data/subgraphs/apy_graph")
    awa2_graph = os.path.join(DIR_PATH, "data/subgraphs/awa2_graph")
    osdd_graph = os.path.join(DIR_PATH, "data/subgraphs/osdd_graph")
    osdd_graph2 = os.path.join(DIR_PATH, "data/subgraphs/conceptnet_2hops_9states")
    osdd_graph8 = os.path.join(DIR_PATH, "data/subgraphs/osdd_graph2")
    osdd_ir_graph = os.path.join(DIR_PATH, "data/subgraphs/osdd_ir_graph")
    osdd_wordnet_states = os.path.join(DIR_PATH, "data/subgraphs/osdd_wordnet_states_graph")
    osdd_ir_9_graph = os.path.join(DIR_PATH, "data/subgraphs/osdd_ir_9_graph")
    
    
    
    conceptnet_hop2_thresh_10 = os.path.join(DIR_PATH, "data/subgraphs/conceptnet-hop2-thresh-10")
    conceptnet_hop2_thresh_10_excl = os.path.join(DIR_PATH, "data/subgraphs/conceptnet-hop2-thresh-10-excl")
    wordnet_states_hop2 = os.path.join(DIR_PATH, "data/subgraphs/wordnet-states-hop2")
    
    
    conceptnet_hop1_thresh_0 = os.path.join(DIR_PATH, "data/subgraphs/conceptnet-hop1-thresh-0")
    conceptnet_hop1_thresh_10 = os.path.join(DIR_PATH, "data/subgraphs/conceptnet-hop1-thresh-10")
    conceptnet_wordnet_hop2_thresh_10 = os.path.join(DIR_PATH, "data/subgraphs/conceptnet-wordnet-hop2-thresh-10")
    
       
    conceptnet_wordnet_hop1_thresh_10 = os.path.join(DIR_PATH, "data/subgraphs/conceptnet-wordnet-hop1-thresh-10")
    conceptnet_wordnet_hop1_thresh_0 = os.path.join(DIR_PATH, "data/subgraphs/conceptnet-wordnet-hop1-thresh-0")
    visualgenome_states_hops0 = os.path.join(DIR_PATH, "data/subgraphs/visualgenome-states-hops0")
    visualgenome_states_hops1 = os.path.join(DIR_PATH, "data/subgraphs/visualgenome-states-hops1")
    visualgenome_hops1 = os.path.join(DIR_PATH, "data/subgraphs/visualgenome-hops1")
    
    
    vg_from_cskg_states_hops0 = os.path.join(DIR_PATH, "data/subgraphs/vg-from-cskg-states-hops0")
    vg_from_cskg_states_hops1 = os.path.join(DIR_PATH, "data/subgraphs/vg-from-cskg-states-hops1")
    vg_from_cskg_states_hops2 = os.path.join(DIR_PATH, "data/subgraphs/vg-from-cskg-states-hops2")
    
    
    vg_from_cskg_hops0 = os.path.join(DIR_PATH, "data/subgraphs/vg-from-cskg-hops0")
    vg_from_cskg_hops1 = os.path.join(DIR_PATH, "data/subgraphs/vg-from-cskg-hops1")
    vg_from_cskg_hops2 = os.path.join(DIR_PATH, "data/subgraphs/vg-from-cskg-hops2")
    
    
    vg_cn_from_cskg_states_hops0 = os.path.join(DIR_PATH, "data/subgraphs/vg-cn-from-cskg-states-hops0")
    vg_cn_wn_from_cskg_states_hops0 = os.path.join(DIR_PATH, "data/subgraphs/vg-cn-wn-from-cskg-states-hops0")
    vg_wn_from_cskg_states_hops0 = os.path.join(DIR_PATH, "data/subgraphs/vg-wn-from-cskg-states-hops0")
    
    
    
    vg_cn_from_cskg_states_hops1 = os.path.join(DIR_PATH, "data/subgraphs/vg-cn-from-cskg-states-hops1")
    vg_cn_wn_from_cskg_states_hops1 = os.path.join(DIR_PATH, "data/subgraphs/vg-cn-wn-from-cskg-states-hops1")
    vg_wn_from_cskg_states_hops1 = os.path.join(DIR_PATH, "data/subgraphs/vg-wn-from-cskg-states-hops1")
    
    
    vg_wk_from_cskg_states_hops0 = os.path.join(DIR_PATH, "data/subgraphs/vg-wk-from-cskg-states-hops0")
    vg_wk_from_cskg_states_hops1 = os.path.join(DIR_PATH, "data/subgraphs/vg-wk-from-cskg-states-hops1")
    vg_wk_from_cskg_states_hops2 = os.path.join(DIR_PATH, "data/subgraphs/vg-wk-from-cskg-states-hops2")
    
 
    
    
    vg_cn_wk_from_cskg_states_hops0 = os.path.join(DIR_PATH, "data/subgraphs/vg-cn-wk-from-cskg-states-hops0")
    vg_cn_wn_wk_from_cskg_states_hops0 = os.path.join(DIR_PATH, "data/subgraphs/vg-cn-wn-wk-from-cskg-states-hops0")
    vg_wn_wk_from_cskg_states_hops0 = os.path.join(DIR_PATH, "data/subgraphs/vg-wn-wk-from-cskg-states-hops0")
    
    
    
    vg_cn_wk_from_cskg_states_hops1 = os.path.join(DIR_PATH, "data/subgraphs/vg-cn-wk-from-cskg-states-hops1")
    vg_cn_wn_wk_from_cskg_states_hops1 = os.path.join(DIR_PATH, "data/subgraphs/vg-cn-wn-wk-from-cskg-states-hops1")
    vg_wn_wk_from_cskg_states_hops1 = os.path.join(DIR_PATH, "data/subgraphs/vg-wn-wk-from-cskg-states-hops1")
    
    
    
    cn_from_cskg_states_hops0 = os.path.join(DIR_PATH, "data/subgraphs/cn-from-cskg-states-hops0")
    cn_wn_from_cskg_states_hops0 = os.path.join(DIR_PATH, "data/subgraphs/cn-wn-from-cskg-states-hops0")
    wn_from_cskg_states_hops0 = os.path.join(DIR_PATH, "data/subgraphs/wn-from-cskg-states-hops0")
    
    
    cn_from_cskg_states_hops1 = os.path.join(DIR_PATH, "data/subgraphs/cn-from-cskg-states-hops1")
    cn_wn_from_cskg_states_hops1 = os.path.join(DIR_PATH, "data/subgraphs/cn-wn-from-cskg-states-hops1")
    wn_from_cskg_states_hops1 = os.path.join(DIR_PATH, "data/subgraphs/wn-from-cskg-states-hops1")
    
    
    cn_from_cskg_states_hops2 = os.path.join(DIR_PATH, "data/subgraphs/cn-from-cskg-states-hops2")
    cn_wn_from_cskg_states_hops2 = os.path.join(DIR_PATH, "data/subgraphs/cn-wn-from-cskg-states-hops2")
    wn_from_cskg_states_hops2 = os.path.join(DIR_PATH, "data/subgraphs/wn-from-cskg-states-hops2")
    
    cn_wk_from_cskg_states_hops0 = os.path.join(DIR_PATH, "data/subgraphs/cn-wk-from-cskg-states-hops0")
    cn_wn_wk_from_cskg_states_hops0 = os.path.join(DIR_PATH, "data/subgraphs/cn-wn-wk-from-cskg-states-hops0")
    wn_wk_from_cskg_states_hops0 = os.path.join(DIR_PATH, "data/subgraphs/wn-wk-from-cskg-states-hops0")
    
    
    cn_wk_from_cskg_states_hops1 = os.path.join(DIR_PATH, "data/subgraphs/cn-wk-from-cskg-states-hops1")
    cn_wn_wk_from_cskg_states_hops1 = os.path.join(DIR_PATH, "data/subgraphs/cn-wn-wk-from-cskg-states-hops1")
    wn_wk_from_cskg_states_hops1 = os.path.join(DIR_PATH, "data/subgraphs/wn-wk-from-cskg-states-hops1")
    
    
    cn_wk_from_cskg_states_hops2 = os.path.join(DIR_PATH, "data/subgraphs/cn-wk-from-cskg-states-hops2")
    cn_wn_wk_from_cskg_states_hops2 = os.path.join(DIR_PATH, "data/subgraphs/cn-wn-wk-from-cskg-states-hops2")
    wn_wk_from_cskg_states_hops2 = os.path.join(DIR_PATH, "data/subgraphs/wn-wk-from-cskg-states-hops2")
    
    
    cskg_states_hops0 = os.path.join(DIR_PATH, "data/subgraphs/cskg-states-hops0")
    cskg_states_hops1 = os.path.join(DIR_PATH, "data/subgraphs/cskg-states-hops1")
    cskg_states_hops2 = os.path.join(DIR_PATH, "data/subgraphs/cskg-states-hops2")
    
    
    cskg_hops0 = os.path.join(DIR_PATH, "data/subgraphs/cskg-hops0")
    cskg_hops1 = os.path.join(DIR_PATH, "data/subgraphs/cskg-hops1")
    cskg_hops2 = os.path.join(DIR_PATH, "data/subgraphs/cskg-hops2")
    
    
    
    options = {
        "label_encoder_type": args.label_encoder_type,
        "trainval": args.trainval,
        "num_epochs": args.max_epoch,
        "batch_size": args.batch_size,
        "device": device,
        "seed": args.seed,
        "ilsvrc_graph_path": ilsvrc_graph,
        "apy_graph_path": apy_graph,
        "awa2_graph_path": awa2_graph,
         "osdd_graph_path": osdd_graph,
          "osdd_graph_path2": osdd_graph2,
          "osdd_ir_graph": osdd_ir_graph,
           "osdd_ir_9_graph": osdd_ir_9_graph,
            "osdd_wordnet_states": osdd_wordnet_states,
            
            "conceptnet_hop2_thresh_10":   conceptnet_hop2_thresh_10 ,
            "conceptnet_hop2_thresh_10_excl":conceptnet_hop2_thresh_10_excl,
            "wordnet_states_hop2":wordnet_states_hop2,
            
              "conceptnet_hop1_thresh_10":   conceptnet_hop1_thresh_10 ,
            "conceptnet_hop1_thresh_0":conceptnet_hop1_thresh_0,
            "conceptnet_wordnet_hop2_thresh_10":   conceptnet_wordnet_hop2_thresh_10 ,
  "conceptnet_wordnet_hop1_thresh_10":   conceptnet_wordnet_hop1_thresh_10 ,
       "conceptnet_wordnet_hop1_thresh_0":   conceptnet_wordnet_hop1_thresh_0 ,       
        "visualgenome_states_hops0":   visualgenome_states_hops0 ,     
         "visualgenome_states_hops1":   visualgenome_states_hops1 ,
          "visualgenome_hops1":   visualgenome_hops1 , 
           "vg_from_cskg_states_hops0":   vg_from_cskg_states_hops0 , 
            "vg_from_cskg_states_hops1":   vg_from_cskg_states_hops1 , 
            "vg_from_cskg_states_hops2":   vg_from_cskg_states_hops2 , 
          
             "vg_from_cskg_hops0":   vg_from_cskg_hops0 , 
            "vg_from_cskg_hops1":   vg_from_cskg_hops1 , 
            "vg_from_cskg_hops2":   vg_from_cskg_hops2 , 
            
            "vg_cn_from_cskg_states_hops0":   vg_cn_from_cskg_states_hops0 ,
            "vg_cn_wn_from_cskg_states_hops0":   vg_cn_wn_from_cskg_states_hops0 ,
             "vg_wn_from_cskg_states_hops0":   vg_wn_from_cskg_states_hops0 ,
             
             
              "vg_cn_from_cskg_states_hops1":   vg_cn_from_cskg_states_hops1 ,
            "vg_cn_wn_from_cskg_states_hops1":   vg_cn_wn_from_cskg_states_hops1 ,
             "vg_wn_from_cskg_states_hops1":   vg_wn_from_cskg_states_hops1 ,
             
             
                "vg_wk_from_cskg_states_hops0":   vg_wk_from_cskg_states_hops0 , 
            "vg_wk_from_cskg_states_hops1":   vg_wk_from_cskg_states_hops1 , 
            "vg_wk_from_cskg_states_hops2":   vg_wk_from_cskg_states_hops2 ,
             
             
                "vg_cn_wk_from_cskg_states_hops0":   vg_cn_wk_from_cskg_states_hops0 ,
            "vg_cn_wn_wk_from_cskg_states_hops0":   vg_cn_wn_wk_from_cskg_states_hops0 ,
             "vg_wn_wk_from_cskg_states_hops0":   vg_wn_wk_from_cskg_states_hops0 ,
             
             
              "vg_cn_wk_from_cskg_states_hops1":   vg_cn_wk_from_cskg_states_hops1 ,
            "vg_cn_wn_wk_from_cskg_states_hops1":   vg_cn_wn_wk_from_cskg_states_hops1 ,
             "vg_wn_wk_from_cskg_states_hops1":   vg_wn_wk_from_cskg_states_hops1 ,
             
             
            "cskg_states_hops0":   cskg_states_hops0 , 
            "cskg_states_hops1":   cskg_states_hops1 , 
            "cskg_states_hops2":   cskg_states_hops2 , 
          
             "cskg_hops0":   cskg_hops0 , 
            "cskg_hops1":   cskg_hops1 , 
            "cskg_hops2":   cskg_hops2 , 
            
            
            
            "cn_from_cskg_states_hops0":   cn_from_cskg_states_hops0 ,
            "cn_wn_from_cskg_states_hops0":   cn_wn_from_cskg_states_hops0 ,
             "wn_from_cskg_states_hops0":   wn_from_cskg_states_hops0 ,
           
           
            "cn_from_cskg_states_hops1":   cn_from_cskg_states_hops1 ,
            "cn_wn_from_cskg_states_hops1":   cn_wn_from_cskg_states_hops1 ,
             "wn_from_cskg_states_hops1":   wn_from_cskg_states_hops1 ,
             
              "cn_from_cskg_states_hops2":   cn_from_cskg_states_hops2,
            "cn_wn_from_cskg_states_hops2":   cn_wn_from_cskg_states_hops2,
             "wn_from_cskg_states_hops2":   wn_from_cskg_states_hops2 ,
             
             
             
             
               "cn_wk_from_cskg_states_hops0":   cn_wk_from_cskg_states_hops0 ,
            "cn_wn_wk_from_cskg_states_hops0":   cn_wn_wk_from_cskg_states_hops0 ,
             "wn_wk_from_cskg_states_hops0":   wn_wk_from_cskg_states_hops0 ,
           
           
            "cn_wk_from_cskg_states_hops1":   cn_wk_from_cskg_states_hops1 ,
            "cn_wn_wk_from_cskg_states_hops1":   cn_wn_wk_from_cskg_states_hops1 ,
             "wn_wk_from_cskg_states_hops1":   wn_wk_from_cskg_states_hops1 ,
             
              "cn_wk_from_cskg_states_hops2":   cn_wk_from_cskg_states_hops2,
            "cn_wn_wk_from_cskg_states_hops2":   cn_wn_wk_from_cskg_states_hops2,
             "wn_wk_from_cskg_states_hops2":   wn_wk_from_cskg_states_hops2 ,
             
             
          
    }

    # ensure save path
    if not os.path.exists(os.path.join(DIR_PATH, "save")):
        os.makedirs(os.path.join(DIR_PATH, "save"))

    model, save_path = get_label_encoder(
        options["label_encoder_type"], options
    )
    
    
    
    model = model.to(device) 
    fc_vectors = get_fc()

        
        # for i in range(3):
        #     model[i] = model[i].to(device)
        #     fc_vectors[i] = fc_vectors[i].to(device)
    
    
    if(args.mode=='test'):

        print("Computing only predictions for graphs")
        model_path='./data/gnn_weights/weights_trgcn.pt'
         
                
             
        model.load_state_dict(torch.load(model_path))
                
        all_preds = {}

    
        #only one KG
        graph_paths = [conceptnet_wordnet_hop1_thresh_10

                            ]
        

        with torch.no_grad():
            for i, dataset in enumerate(["conceptnet_wordnet_hop1_thresh_10"        ]):
                
                print(dataset,graph_paths[i],)
            # for i, dataset in enumerate(["awa", "apy","osdd","osdd2",'osdd_ir','osdd_ir_9_graph']):
                preds = predict(model, graph_paths[i], dataset)
                all_preds[dataset] = preds
                
        save_path='./data/pred_embeddings/'
        torch.save(model.state_dict(), save_path + "predictions_%s_%s.pt"%(options["label_encoder_type"],args.seed)) 
        
      
        print("done!")
    else:
        save_path='./data/gnn_weights/'
        model, tr_log = train_gnn_model_imgnet(model, fc_vectors, device, options)
        torch.save(model.state_dict(), save_path + "%s_%d_imagenet.pt"%(options["label_encoder_type"],args.seed)) 
        print("done!") 
      
            
            
            
                
            
            
         