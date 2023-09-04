

## Code Instructions:
Setup
Our work is implemented in PyTorch and tested with Ubuntu 18.04/20.04.

- Python 3.7
- PyTorch 1.6.0

Create a conda environment `AoSC` using
```
conda env create -f environment.yml
conda activate AoSC
```

## Preperation:
- Download the datasets, the KG, the embeddings and the saved weights for the model by openning a terminal and running the following script:

```
cd src
bash download_data.sh

```


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
</pre>


## Testing

**OSDD  Dataset**

- To reproduce the reported results for OSDD  Dataset, adjust the weights and the embeddings paths in test_mit.sh and run:

```
bash  test_osdd.sh

```

**CGQA-States  Dataset**

- To reproduce the reported results for CGQA-States Dataset, adjust the weights and the embeddings paths in test_mit.sh and run:

```
bash  test_cgqa.sh

```

**MIT-States  Dataset**

- To reproduce the reported results for MIT-States Dataset, adjust the weights and the embeddings paths in test_mit.sh and run:

```
bash  test_mit.sh

```


## Training from scratch


**Training of the GNN**

- Run the following script to train the GNN:

```
bash train_gnn.sh

```


**Embeddings Computation**

- Run the following script to compute the embeddings:

```
bash  test_gnn.sh

```


**OSDD  Dataset**

- Adjust the embeddings path in the finetune_osdd.sh script and then run it:



```
bash  finetune_osdd.sh

```

- Adjust the weights path in the test_osdd.sh script and then run it:


```
bash  test_osdd.sh

```

**CGQA-States  Dataset**

- Adjust the embeddings path in the finetune_osdd.sh script and then run it:



```
bash  finetune_cgqa.sh

```

- Adjust the weights path in the test_osdd.sh script and then run it:


```
bash  test_cgqa.sh

```

**MIT-States  Dataset**

- Adjust the embeddings path in the finetune_osdd.sh script and then run it:



```
bash  finetune_mit.sh

```

- Adjust the weights path in the test_osdd.sh script and then run it:


```
bash  test_mit.sh

```
