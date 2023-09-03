import argparse
import os
import os.path as osp
import distutils.util
import pandas as pd
import torch
import torch.nn as nn
from scipy import io
from torchvision.models import resnet50, resnet101
from torchvision import datasets,transforms
import sys
from distutils import util
import random
import numpy as np
from flags import parser

data_transforms = transforms.Compose([
        transforms.Resize([224,224]),
 
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
   
    




DIR_PATH = os.path.dirname(os.path.realpath(__file__))

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


def get_save_path(_path, pred_file_path, resnet_50, fold):
    save_path = os.path.join(DIR_PATH, _path)
    pred_file_name = os.path.basename(pred_file_path)
    save_path = os.path.join(save_path, pred_file_name)
    if resnet_50:
        save_path = os.path.join(save_path, "resnet_50")
    else:
        save_path = os.path.join(save_path, "resnet_101")

    save_path = os.path.join(save_path, "fold_" + str(fold))

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    return save_path


if __name__ == "__main__":
   
    args = parser.parse_args()

    # setting seed value
    seed=args.seed

    
    

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


    else:
        classes_dict={
        
        'closed':0,
    
        'empty':1,
        'filled':2,
        
        
        'folded':3,
        'open':4,
    
        
        
    }







    train_ids=[x for x in classes_dict.values()]
    
    

    device = torch.device("cuda:%s"%args.gpu if torch.cuda.is_available() else "cpu")

    random.seed(seed)

    # get the saved path for the fine-tuned folder
    graph_type=args.graph_type
  #  print(graph_type)
    # if(graph_type=='osdd_ir'):
   

    save_path = args.save_path+"/finetune_%s/%s"%(args.dataset,graph_type)
    
    
    


    if not os.path.exists(save_path):
        os.makedirs(save_path)

    print("save path {}".format(save_path))
  #  print(args.embs_pred)
    pred = torch.load(args.embs_pred)
    
    
    if(args.classes_ommited):
            
        classes_ommited=args.classes_ommited.split("_")
        
        for cls in classes_ommited:
            # print(cls,classes_dict[cls])
            # print(train_ids)
            
            if(cls in classes_dict):
                train_ids.remove(classes_dict[cls])


    # load the fold and train/val indices
     # load the fold and train/val indices
    # dataset_split = io.loadmat(
    #     osp.join(DIR_PATH, "datasets/osdd/att_splits.mat")
    # )
    
    #object='bottle/'
    #pairs='_4_8_/'
    #exps=''
    #exp=object+pairs+exps+"/train/"

    exp=args.exp_type
    # image_data = pd.read_csv(
    #     osp.join(DIR_PATH, "datasets/osdd/classification.%s/image_label_test.csv"%exp)
    # )
    
   
    train_dataset = datasets.ImageFolder(args.train_dir,data_transforms)
            
   
   
  
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch,
        shuffle=True,
        num_workers=0,
        pin_memory=True,
        sampler=None,
    )



   # train_ids=[3,2]
   
   
   
    if(graph_type!='random'):
  
        if(args.dataset=='osdd'):
            pred_vectors = pred["%s"%graph_type]#[2:4,:]
        else:
            if(len(pred["%s"%graph_type])>5):
                pred_vectors = pred["%s"%graph_type][[0,2,3,4,5],:]
            else:
                pred_vectors = pred["%s"%graph_type]
                

    else:
        random_pred_vectors = np.random.uniform(0,1,(9,2049))

        pred_vectors=torch.Tensor(random_pred_vectors).cuda()
      #  pass
    print(args.dataset)
    print((len(pred["%s"%graph_type])))
    if(args.weights_path):
        
    #  0        1       2          3      4         5   6           7     8
    #closed, containing, empty, filled, folded, open, plugged, unfolded unplugged 
      
      
      
    #  5       0        2   3       1           6           8       4       7
    
#      0        1       2           3      4         5      6           7     8
    # open, closed, empty, filled, containing, plugged, unpluged, folded, unfolded     
         
         #fcw=fcw[[1,4,2,3,7,0,5,8,6],:]
         
         
       #  pred_vectors=pred_vectors[[1,4,2,3,7,0,5,8,6],:]
        # print((pred_vectors[1,:]))
        
        # kk=[pred_vectors[x,:] for x in  range(len(pred_vectors))]
        # print(kk)
        pred_vectors=pred_vectors[[5,0,2,3,1,6,8,4,7],:]
        # kk=[pred_vectors[x,:] for x in  range(len(pred_vectors))]
        # print(kk)
        pass
    print("pred vector shape {}".format(pred_vectors.size()))

    if args.resnet_50:
        print('50')
        model = resnet50(pretrained=True)
    else:
        print('101')
        model = resnet101(pretrained=True)





    if(args.classes_ommited):
            pred_vectors=pred_vectors[train_ids,:]
    fcw = pred_vectors
    
   
   
    num_ftrs = model.fc.in_features
    
    model.fc = nn.Linear(num_ftrs, len(classes_dict))
    
    
  

    if(args.classes_ommited):
        num_ftrs = model.fc.in_features
    
        model.fc = nn.Linear(num_ftrs, len(train_ids))
        
        
    model.fc.weight = nn.Parameter(fcw[:, :-1])
    model.fc.bias = nn.Parameter(fcw[:, -1])   
    # print(  model.fc.bias)
   # print(model.fc.weight.shape)
   # print(len(train_ids))
    if(args.freeze):
        model.fc.weight.requires_grad = False
        model.fc.bias.requires_grad = False
    model = model.cuda()
    model.train()

    #
    optimizer = torch.optim.SGD(model.parameters(), lr=0.0001, momentum=0.9)
    loss_fn = nn.CrossEntropyLoss().cuda()


    for epoch in range(0, args.num_epochs):

        ave_loss = None
        ave_acc = None

        for i, (data, label) in enumerate(train_loader, 1):
            data = data.cuda()
            label = label.cuda()
           # print(label)
           # print((label))
            # if(args.weights_path is not None):
            
            #     new_label=[label_map_indices[int(x)] for x in label.tolist()]
                
            #    # print(new_label)
            #     label=torch.Tensor(new_label).cuda()
            #     label = label.type(torch.LongTensor).cuda()
                
                
           # print(label)
           # print(label)
            logits = model(data)
            loss = loss_fn(logits, label)
          
            _, pred = torch.max(logits, dim=1)
            #print(label)
            
            #print(sum(pred==label) )
            # if(args.weights_path is not None):
                 
            #     new_pred=[label_map_indices[x] for x in pred.tolist()]
            #     pred=torch.Tensor(new_pred).cuda()
                
            #print(sum(pred==label) )    
            acc = torch.eq(pred, label).type(torch.FloatTensor).mean().item()

         
            print(
                "epoch {}, {}/{}, loss={:.4f} , acc={:.4f} ".format(
                    epoch,
                    i,
                    len(train_loader),
                    loss.item(),
                  
                    acc,
                 
                )
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


        # if (epoch + 1)  == args.num_epochs:
        if (epoch + 1)%25  ==0:    
            torch.save(
                model.state_dict(),
                osp.join(save_path, "epoch-{}.pth".format(epoch)),
            )

    print("done!")
