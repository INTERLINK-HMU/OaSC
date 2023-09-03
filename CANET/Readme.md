
## Setup
The mehtod is implemented in PyTorch and tested with Ubuntu 18.04/20.04.

- Python 3.8
- PyTorch 1.7.0

Create a conda environment `CANET` using
```
conda env create -f environment.yml
conda activate CANET
```

##  Data Preparation

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


# Testing

**OSDD  Dataset**

- To reproduce the reported results for OSDD  Dataset, adjust the weights and the embeddings paths in test_mit.sh and run:

```
python test.py --dataset osdd_obj --auc

python test.py --dataset osdd_cw  --auc

python test.py --dataset osdd_ow  --auc


```

**CGQA-States  Dataset**

- To reproduce the reported results for CGQA-States Dataset, adjust the weights and the embeddings paths in test_mit.sh and run:

```
python test.py --dataset cgqa_obj --auc

python test.py --dataset cgqa_states  --auc

python test.py --dataset cgqa_ow  --auc 

```

**MIT-States  Dataset**

- To reproduce the reported results for MIT-States Dataset, adjust the weights and the embeddings paths in test_mit.sh and run:

```
python test.py --dataset mit_obj --auc

python test.py --dataset mit_cw  --auc

python test.py --dataset mitd_ow  --auc

```


