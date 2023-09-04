
## Setup
The mehtod is implemented in PyTorch and tested with Ubuntu 18.04/20.04.

- Python 3.8
- PyTorch 1.11.0

Create a conda environment `kg-sp` using
```
conda env create -f environment.yml
conda activate kg-sp
```

## Preperation:
- Download the datasets and  the embeddings for the model by openning a terminal and running the following script:

```
bash download_data.sh
bash download_embeddings.sh
```
<!-- 

After the  script completes, the repo must have the following folder structure.

<pre>

./
├── datasets
│   ├── cgqa
│   ├── mit_states
│   └── osdd
├── embeddings
│   ├── cgqa_emb.pred
│   ├── mit_emb.pred
│   └── osdd_emb.pred
├── environment.yml
├── Material_for_save
│   ├── datasets
│   ├── saved_checkpoints
│   └──split2
├── Readme.md
├── saved_checkpoints
│   ├── cgqa
│   ├── mit
│   └── osdd
└── src
    ├── data
    ├── download_data.sh
    ├── finetune_cgqa.sh
    ├── finetune_mit.sh
    ├── finetune_osdd.sh
    ├── finetune.py
    ├── flags.py
    ├── KG
    ├── requirements2.yml
    ├── requirements.txt
    ├── test_cgqa.sh
    ├── test_gnn.sh
    ├── test_mit.sh
    ├── test_osdd.sh
    ├── test.py
    ├── test.sh
    ├── train_gnn.py
    └── train_gnn.sh
</pre> -->


## Testing 

-To reproduce the reported results  run in a terminal the following.
The first command corresponds to the GO setting, the second to the OW setting
and the third to the CW setting.

**OSDD  Dataset**

```
python test.py --logpath "./logs/zero_shot_split1_tr_object" --auc 

python test.py --logpath "./logs/zero_shot_split1_tr_ow" --auc 

python test.py --logpath "./logs/zero_shot_split1_tr_cw" --auc 

```

**CGQA-States  Dataset**

```
python test.py --logpath "./logs/cgqa_split2_tr_object" --auc 

python test.py --logpath "./logs/cgqa_split2_tr_ow" --auc 

python test.py --logpath "./logs/cgqa_split2_tr_cw"  --auc 

```

**MIT-States  Dataset**

```
 python test.py --logpath "./logs/mit_split2_tr_object"  --auc 

 python test.py --logpath "./logs/mit_split2_tr_ow" --auc 

 python test.py --logpath "./logs/mit_split2_tr_cw" --auc  

```



## Training from scratch

- To train the model from scratch, run in the terminal the following.
The first command corresponds to the GO setting, the second to the OW setting
and the third to the CW setting.


**OSDD  Dataset**

```
python train.py --config configs/kgsp/osdd_object.yml

python train.py --config configs/kgsp/osdd_ow.yml

python train.py --config configs/kgsp/osdd_cw.yml


```

**CGQA-States  Dataset**

```
python train.py --config configs/kgsp/cgqa_states_object.yml

python train.py --config configs/kgsp/cgqa_states_ow.yml

python train.py --config configs/kgsp/cgqa_states_cw.yml
```

**MIT-States  Dataset**

```
python train.py --config configs/kgsp/mit_object.yml

python train.py --config configs/kgsp/mit_ow.yml

python train.py --config configs/kgsp/mit_cw.yml
```



