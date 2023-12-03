
## Setup
The mehtod is implemented in PyTorch and tested with Ubuntu 18.04/20.04.

- Python 3.8
- PyTorch 1.11.0

Create a conda environment `ade` using
```
conda env create -f environment.yml
conda activate ade

```

## Preperation:
- Download the datasets. the pretrained weights for and  the embeddings for the model by openning a terminal and running the following scripts in the terminal:
- Download the datasets and  the embeddings for the model by openning a terminal and running the following script in the terminal:

```
bash download_data.sh
bash download_embeddings.sh

```


## Download saved checkpoints:
- Download the saved checkpoints running the following script in the terminal:

```
bash download_logs.sh

```



## Testing

- To reproduce the reported results run in the terminal the following.
The first command corresponds to the GO setting, the second to the OW setting
and the third to the CW setting.


**OSDD  Dataset**


```
python test.py --logpath "./logs/zero_shot_split1_tr_object_ow" --auc 

python test.py --logpath "./logs/zero_shot_split1_tr_ow" --auc 

python test.py --logpath "./logs/zero_shot_split1_tr_cw" --auc 

```

**CGQA-States  Dataset**


```
python test.py --logpath "./logs/cgqa_split2_tr_object" --auc 

python test.py --logpath "./logs/cgqa_split2_tr_ow"  --auc 

python test.py --logpath "./logs/cgqa_split2_tr_cw" --auc 

```

**MIT-States  Dataset**


```
 python test.py --logpath "./logs/mit_split2_tr_object" --auc 

 python test.py --logpath "./logs/mit_split2_tr_ow" --auc 

 python test.py --logpath "./logs/mit_split2_tr_cw" --auc  

```


**VAw  Dataset**


```
 python test.py --logpath "./logs/vaw_object" --auc 

 python test.py --logpath "./logs/vaw_ow" --auc 

 python test.py --logpath "./logs/vaw_cw" --auc  

```


## Training from scratch


- To train the model from scratch, run in the terminal the following.
The first command corresponds to the GO setting, the second to the OW setting
and the third to the CW setting.

**OSDD  Dataset**



```
python train.py --config "./configs/osdd_states_obj.yml"  

python train.py --config "./configs/osdd_states_ow.yml"

python train.py --config "./configs/osdd_states_cw.yml"

```

**CGQA-States  Dataset**


```
python train.py --config "./configs/cgqa_states_obj.yml"  

python train.py --config "./configs/cgqa_states_ow.yml"

python train.py --config "./configs/cgqa_states_cw.yml"

```

**MIT-States  Dataset**


```
python train.py --config "./configs/mit_states_obj.yml"  

python train.py --config "./configs/mit_states_ow.yml"

python train.py --config "./configs/mit_states_cw.yml"

```



**VAW  Dataset**


```
python train.py --config "./configs/vaw_obj.yml"  

python train.py --config "./configs/vaw_ow.yml"

python train.py --config "./configs/vaw_cw.yml"

```



