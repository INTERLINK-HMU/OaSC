
## Setup
The mehtod is implemented in PyTorch and tested with Ubuntu 18.04/20.04.

- Python 3.8
- PyTorch 1.11.0

Create a conda environment `compcos` using
```
conda env create -f environment.yml
conda activate compcos

```

## Preperation:
- Download the datasets and  the embeddings for the model by openning a terminal and running the following script:

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

- To reproduce the reported results run in the terminal the following. The script produces the results for
AoP, le+, symnet, tmn and CompCos. The results for the osdd are saved in the file results_osdd.txt,
the results for the cgqa are saved in the file results_cgqa.txt and the results for the mit are saved in 
the file results_mit.txt.

**OSDD  Dataset**

```
bash test_osdd.sh

```

**CGQA-States  Dataset**

```
bash test_cgqa.sh

```

**MIT-States  Dataset**


```
bash test_mit.sh

```

**MIT-States  Dataset**


```
bash test_vaw.sh

```


## Training from scratch


To train from scratch run in the terminal the following. The script trains
AoP, le+, symnet, tmn and CompCos. 

**OSDD  Dataset**

```
bash train_osdd.sh

```

**CGQA-States  Dataset**

```
bash train_cgqa.sh

```

**MIT-States  Dataset**


```
bash train_mit.sh

```



**VAW Dataset**


```
bash train_vaw.sh

```



