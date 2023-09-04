
## Setup
The mehtod is implemented in PyTorch and tested with Ubuntu 18.04/20.04.

- Python 3.8
- PyTorch 1.11.0

Create a conda environment `OaDis` using
```
conda env create -f environment.yml
conda activate OaDis
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

- To reproduce the reported results run in the terminal the following.
The first command corresponds to the GO setting, the second to the OW setting
and the third to the CW setting.


**OSDD  Dataset**

```
python test.py --cfg config/osdd_obj_ow.yml --load saved_models/osdd_obj/osdd_obj_ow_124/best.pth 

python test.py --cfg config/osdd_ow.yml --load saved_models/osdd_ow/osdd_ow_124/best.pth

python test.py --cfg config/osdd_cw.yml --load saved_models/osdd_cw/osdd_cw_124/best.pth 

```

**CGQA-States  Dataset**

```
python test.py --cfg config/cgqa_ow.yml --load saved_models/cgqa_ow/cgqa_cw_124/best.pth     

python test.py --cfg config/cgqa_cw.yml --load saved_models/cgqa_cw/cgqa_cw_124/best.pth     

python test.py --cfg config/cgqa_obj_ow.yml --load saved_models/cgqa_obj/cgqa_obj_ow_124/best.pth  

```

**MIT-States  Dataset**

```
python test.py --cfg config/mit_obj_ow.yml --load saved_models/mit_obj/mit_obj_ow_124/best.pth 

python test.py --cfg config/mit_ow.yml --load saved_models/mit_ow/mit_ow_124/best.pth 

python test.py --cfg config/mit_cw.yml --load saved_models/mit_cw/mit_cw_124/best.pth 

```



## Training from scratch


- To train the model from scratch, run in the terminal the following.
The first command corresponds to the GO setting, the second to the OW setting
and the third to the CW setting.


**OSDD  Dataset**

```
python train.py --cfg configs/osdd_obj_ow.yml

python train.py --cfg configs/osdd_ow.yml

python train.py --cfg configs/osdd_cw.yml


```

**CGQA-States  Dataset**

```
python train.py --cfg configs/cgqa_obj_ow.yml

python train.py --cfg configs/cgqa_ow.yml

python train.py --cfg configs/cgqa_cw.yml
```

**MIT-States  Dataset**

```
python train.py --cfg configs/mit_obj_ow.yml

python train.py --cfg configs/mit_ow.yml

python train.py --cfg configs/mit_cw.yml
```




