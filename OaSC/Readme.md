

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

Pre-requisites:
- Download the datasets, the embeddings and the saved weights for the model [here](https://drive.google.com/file/d/1lFr0C1aTJsufXiQ3p_nhcU2ovC7MY_EW/view?usp=sharing) and then uncompress the  three files in your machine.
- Update the paths for the datasets, embeddings and saved checkpoints paths in the sh files for testing and training.


To reproduce the reported results for OSDD  Dataset, adjust the weights and the embeddings paths in test_mit.sh and run:
```
./test_osdd.sh
```

If you want to train (finetune) the model from scratch befor testing, run:
```
./finetune_osdd.sh
```

Then adjust the path for the finetuned weights  in test_osdd.sh and run::
```
./test_osdd.sh
```

To reproduce the reported results for CGQA-States Dataset, adjust the weights and the embeddings paths in test_mit.sh and run:
```
./test_cgqa.sh
```

If you want to train (finetune) the model from scratch befor testing, run:
```
./finetune_cgqa.sh
```

Then adjust the path for the finetuned weights  in test_cgqa.sh and run::
```
./test_cgqa.sh
```

To reproduce the reported results for MIT-States Dataset, adjust the weights and the embeddings paths in test_mit.sh and run:
```
./test_mit.sh
```

If you want to train (finetune) the model from scratch befor testing, run:
```
./finetune_mit.sh
```

Then adjust the path for the finetuned weights  in test_mit.sh and run::
```
././test_mit.sh
```
