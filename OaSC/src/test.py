import argparse
import json
import os
import os.path as osp


import pandas as pd
import torch
import torch.nn as nn
from scipy import io
from torch.utils.data import DataLoader
from torchvision.models import resnet50, resnet101
import numpy as np
import random
import sys
from torchvision import datasets,transforms
import pickle

from flags import parser


data_transforms = transforms.Compose([
        transforms.Resize([224,224]),
 
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

DIR_PATH = os.path.dirname(os.path.realpath(__file__))

  #  0        1       2          3      4         5   6           7     8
    #closed, containing, empty, filled, folded, open, plugged, unfolded unplugged 
      
      
      
    #  5       0        2   3       1           6           8       4       7
    
#      0        1       2           3      4         5      6           7     8
    # open, closed, empty, filled, containing, plugged, unpluged, folded, unfolded     







if __name__ == "__main__":

  
    
    args = parser.parse_args()
    
    
    print(args.dataset)
    
    if(args.dataset=='osdd'):
    
        classes_dict={
        
        'closed':0,
        'containing':1,
        'empty':2,
        'filled':3,
        
        
        'folded':4,
        'open':5,
        'plugged':6,
        'unfolded':7,
        'unplugged':8,
        
         }
        label_map_indices={
            0:5,
            1:0,
            2:2,
            3:3,
            4:1,
            5:6,
            6:8,
            7:4,
            8:7
            
            
            
        }
        
   


    else:
        classes_dict={
        
        'closed':0,
    
        'empty':1,
        'filled':2,
        
        
        'folded':3,
        'open':4,
    
        
        
    }


    train_ids=[x for x in classes_dict.values()]
# not_train_ids=[x for x in range(9) if x not in classes_dict.values()]


# print(train_ids)
# print(not_train_ids)
    not_train_ids=[]
    verbose=False
    by_class=False
    
    
    #if(by_class):
    if(args.pickle and os.path.exists(args.pickle)):

        with open(args.pickle,'rb') as file:
            res_dict=pickle.load(file)
            
            
        file.close()   
    else:
      #  print(args.log)
        res_dict={}
        res_dict[args.graph_type]={}
     # setting seed value
    seed=args.seed
    if(args.graph_type not in res_dict):
        res_dict[args.graph_type]={}


    random.seed(seed)
    
    if torch.cuda.is_available():
        device = torch.device("cuda:" + args.gpu)
        #device = torch.device("cuda:1")
    else:
        device = torch.device("cpu")
   # print("device : ", device)
   

    if(args.classes_ommited):
        
        classes_ommited=args.classes_ommited.split("_")
      #  print(args.classes_ommited)
      #  print(train_ids)
      #  print(not_train_ids)
        for cls in classes_ommited:
            # print(cls,classes_dict[cls])
            # print(train_ids)
           
           # print(cls)
            if(classes_dict[cls] in train_ids):
                train_ids.remove(classes_dict[cls])
                not_train_ids.append(classes_dict[cls])
      #  print(train_ids)
      #  print(not_train_ids)
    
    
    pred_file = torch.load(args.embs_pred, map_location="cpu")
   # a=[4,8]

    graph_type=args.graph_type
    
    
    if(graph_type not in pred_file):
        print("predcictions not found for this graph!")

        sys.exit()
    
    
        
    if(graph_type!='random'):
        
        if(args.dataset=='osdd'):
            pred_vectors = pred_file["%s"%graph_type]#[2:4,:]
        else:
            if(len(pred_file["%s"%graph_type])>5):
                pred_vectors = pred_file["%s"%graph_type][[0,2,3,4,5],:]
            else:
                pred_vectors = pred_file["%s"%graph_type]
                
        pred_vectors = pred_vectors.to(device)
    else:
        
        random_pred_vectors = np.random.uniform(0,1,(9,2049))

        pred_vectors=torch.Tensor(random_pred_vectors).cuda()
        #pass
    if args.resnet_50:
        cnn = resnet50(pretrained=True)
        cnn.fc = nn.Identity()

        if args.cnn:
            params = torch.load(args.cnn)
          
            cnn.load_state_dict(params)
    else:
    
        cnn = resnet101(pretrained=True)
       # cnn.fc = nn.Linear(2048, 9)
        cnn.fc = nn.Identity()
        if args.cnn:
            #print(cnn)
            params = torch.load(args.cnn)
            del params["fc.weight"]
            del params["fc.bias"]
            cnn.load_state_dict(params)
            #print(cnn[0])
            
            
            
    # if(args.classes_ommited):
        
    #     cnn.fc[:,train_ids]-=args.gamma
        
        
  
        
    model = cnn.to(device)
    model.eval()
    
    
    
    
    test_dataset = datasets.ImageFolder(args.test_dir,data_transforms)
            
    mapped_labels={}
    for key in test_dataset.class_to_idx:

        mapped_labels[test_dataset.class_to_idx[key]]=classes_dict[key]
   
    print(mapped_labels)
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=args.batch,
        shuffle=True,
        num_workers=0,
        pin_memory=True,
        sampler=None,
    )

    model.to(device)
    
  #  import numpy as np
    with torch.no_grad():
        
        #for gamma in range(0.125,2,0.125):#-1e18,0]:
        seen_accuracy=[]
        unseen_accuracy=[]
        hm_array=[]
        
        
        predictions=None
        gts=None
        for i, (inputs, labels) in enumerate(test_loader):
                
            #  print(inputs)
            # print(inputs.size())
            # print(device)
                inputs = inputs.to(device)
                labels = labels.to(device)
            
                outputs = model(inputs)
                feat = torch.cat(
                [outputs, torch.ones(len(outputs)).view(-1, 1).to(device)], dim=1
            )

                fcs = pred_vectors.t()

                if(predictions==None):
                    predictions=torch.matmul(feat, fcs)
                    
                else:
                    predictions=torch.cat((predictions, torch.matmul(feat, fcs)), dim=0)
                #print(predictions.size())
                
                if(mapped_labels!=[]):
                    cur_gts=[mapped_labels[x] for x in labels.tolist()]
                    cur_gts=torch.Tensor(cur_gts).cuda()
                    cur_gts=cur_gts.type(torch.LongTensor).cuda()
                else:
                    cur_gts=[x  for x in labels.tolist()]
                    cur_gts=torch.Tensor(cur_gts).cuda()
                    cur_gts=cur_gts.type(torch.LongTensor).cuda()
                 
                 
               # print(cur_gts.size())    
               # print(gts.size())   
                if(gts==None):
                        gts=cur_gts
                    
                else:
                    gts=torch.cat((gts, cur_gts), dim=0)
                
              
              #  table[:, train_ids] -= gamma
              #  print(not_train_ids)
               
        
        if(args.log):    
            sys.stdout = open('%s'%(args.log),'a+')
        print("results for graph %s object %s "%(args.graph_type,args.variable))
        for gamma   in  np.arange (-5.,15.125,0.5):
            if(verbose):
                print("gamma = %e  "%(gamma))
            
            curs_preds=predictions.clone()
            
            curs_preds[:, not_train_ids] += gamma
            total=0
            correct=0
            
            
            
            
            total_s=0
            total_un=0
            correct_s=0
            correct_un=0
            
            if(by_class):
                    
                dict_results={}
                pred_results={}
                samples_dict={}
                for i in range(len(classes_dict)):
                    
                    dict_results[i]=[0,0]
                    samples_dict[i]=[0]
                    pred_results[i]=[0,0,0,0,0,0,0,0,0]
                    
        
                
               # print(table)
            
            # table = table.detach().cpu()
              #  print(table[:10, :],gamma )
            predicted = torch.argmax(curs_preds, dim=1) 
            
        
        # outputs = model(inputs)
            #print(predicted.size())
           # print(gts)
          #  print(gts.size())
        # _, predicted = torch.max(outputs, 1)
            
            
            total = gts.shape[0]
           # print(predicted == gts)
            correct = torch.tensor(predicted == gts).cpu().sum().item()
            
            
            #print(labels)
            #print(train_ids )
            
            seen_ids=[label in train_ids for label in gts]
            unseen_ids=[label not in train_ids for label in gts]
            total_s = gts[seen_ids].shape[0]
    
            correct_s = torch.tensor(predicted[seen_ids] == gts[seen_ids]).cpu().sum().item()
            
            
            total_un = gts[unseen_ids].shape[0]
            correct_un = torch.tensor(predicted[unseen_ids] == gts[unseen_ids]).cpu().sum().item()
            
            if(by_class):
               # predicted=predicted.cpu().numpy()
               # gts = gts.cpu()
                
            
                
                
                for pred,gt in zip(predicted,gts):
                    
                    dict_results[gt.item()][0]+=1
                    pred_results[gt.item()][pred.item()]+=1
                    #  samples_dict[lbl]+=1
                    
                    if(pred.item()==gt.item()):
                        dict_results[gt.item()][1]+=1
                
                #print(labels,correct,total)
    # print(labels_)
    # print(predics_)
            
            
            
         #   with 
            
            #if(verbose):    
              
            
            
            if(args.variable not in res_dict[args.graph_type]):
                res_dict[args.graph_type][args.variable]={}
            res_dict[args.graph_type][args.variable][gamma]={}
            
            if(verbose):
                print('Accuracy of the network on the  test images: %.2f '%(100 * correct / total))
            
            if(total_s>0):
            
                ac_seen=correct_s / total_s
            else:
                ac_seen=np.nan
             
            if(total_un>0):   
                ac_unseen=correct_un / total_un
            
            else:
                ac_unseen=np.nan
                
            if(verbose):
                print('Accuracy seen images: %.3f '%(100 * ac_seen))
                print('Accuracy unseen images: %.3f '%(100 * ac_unseen))
                
            
            
            
            if(100 * ac_seen>0 and 100 * ac_unseen>0):
                 hm=2*ac_seen*ac_unseen/(ac_seen+ac_unseen)
                 
                       
            else:
                
                hm=0
            if(verbose):  
                print('HMs: %.3f '%(2*ac_seen*ac_unseen/(ac_seen+ac_unseen)))
            res_dict[args.graph_type][args.variable][gamma]['total']='%.3f '%(100 * correct / total)
            res_dict[args.graph_type][args.variable][gamma]['seen']='%.3f '%(100 * ac_seen)
            res_dict[args.graph_type][args.variable][gamma]['unseen']='%.3f '%(100 * ac_unseen)
            res_dict[args.graph_type][args.variable][gamma]['hm']='%.3f '%((2*ac_seen*ac_unseen/(ac_seen+ac_unseen)))
        
            seen_accuracy.append(ac_seen)
            unseen_accuracy.append(ac_unseen)
            
            
            hm_array.append(hm)
            
            
            
            if(by_class):
                for sta_class in classes_dict:

                    class_num=classes_dict[sta_class]
                    correct=dict_results[class_num][1]
                    total=dict_results[class_num][0]
                    if(total!=0):
                        print('Accuracy of the network on the  test images for state %s: %.2f '%(sta_class,100 * correct / total))
                        
                        res_dict[args.graph_type][args.variable][gamma][sta_class]='%.2f '%(100 * correct / total)
                    else:
                        print('Accuracy of the network on the  test images for state %s: 0 samples '%(sta_class)) 
                        res_dict[args.graph_type][args.variable][gamma][sta_class]='0 samples'
    
                    res_strings="total samples %s:%i ,"%(sta_class,total)
  
                    for sta_class2 in classes_dict:
                        class_num2=classes_dict[sta_class2]
                        res_strings+="predicted as %s:%i ,"%(sta_class2,pred_results[class_num][class_num2])

   
                    print(res_strings)
            
            
        res_dict[args.graph_type][args.variable][-1000]={}
        res_dict[args.graph_type][args.variable][-1000]['s_max']=np.max(seen_accuracy)    
        res_dict[args.graph_type][args.variable][-1000]['un_max']=np.max(unseen_accuracy)
            
        res_dict[args.graph_type][args.variable][-1000]['hm_max']=np.max(hm_array)
       
        
        
        seen_ac, unseen_ac = np.array(seen_accuracy), np.array(unseen_accuracy)
        area = np.trapz(seen_ac, unseen_ac)
        res_dict[args.graph_type][args.variable][-1000]['AUC']=area 
        print('HM s_max: %.2f '%(np.max(seen_accuracy)*100 ))
        print('HM un_max: %.2f '%(np.max(unseen_accuracy)*100))
        print('HM max: %.2f '%(np.max(hm_array)*100))
        print('AUC : %.2f '%(area*100))  
        if(args.pickle):      
            with open(args.pickle, 'wb') as handle:
                pickle.dump(res_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
