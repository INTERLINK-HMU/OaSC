
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

**VAW  Dataset**

```

python test.py --cfg config/vaw_states_ow.yml --load saved_models/vaw_states__ow/vaw_states__ow_124/best.pth  

python test.py --cfg config/vaw_states__cw.yml --load saved_models/vaw_states__cw/vaw_states__cw_124/best.pth  

python test.py --cfg config/vaw_states__obj_ow.yml --load saved_models/vaw_states__obj/vaw_states__obj_ow_124/best.pth 

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


**VAW Dataset**

```
python train.py --cfg configs/vaw_states_obj.yml

python train.py --cfg configs/vaw_states_ow.yml

python train.py --cfg configs/vaw_states_cw.yml

```





