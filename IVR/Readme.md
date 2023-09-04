
## Setup
The mehtod is implemented in PyTorch and tested with Ubuntu 18.04/20.04.

- Python 3.8
- PyTorch 1.11.0

Create a conda environment `IVR` using
```
conda env create -f environment.yml
conda activate IVR
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
python test.py --logpath logs/IVR/osdd --auc 

python test.py --logpath logs/IVR/osdd_ow --auc 

python test.py --logpath logs/IVR/osdd_cw --auc 


```

**CGQA-States  Dataset**

```
python test.py --logpath logs/IVR/cgqa_obj --auc 

python test.py --logpath logs/IVR/cgqa_ow --auc 

python test.py --logpath logs/IVR/cgqa_cw --auc 
```

**MIT-States  Dataset**

```
python test.py --logpath logs/IVR/mit_obj --auc 

python test.py --logpath logs/IVR/mit_ow --auc 

python test.py --logpath logs/IVR/mit_cw --auc
```



## Training from scratch


- To train the model from scratch, run in the terminal the following.
The first command corresponds to the GO setting, the second to the OW setting
and the third to the CW setting.


**OSDD  Dataset**

```
python train.py --config configs/osdd_obj.yml

python train.py --config configs/osdd_ow.yml

python train.py --config configs/osdd_cw.yml


```

**CGQA-States  Dataset**

```
ython train.py --config configs/cgqa_obj.yml

python train.py --config configs/cgqa_ow.yml

python train.py --config configs/cgqa_cw.yml
```

**MIT-States  Dataset**

```
ython train.py --config configs/mit_obj.yml

python train.py --config configs/mit_ow.yml

python train.py --config configs/mit_cw.yml
```

