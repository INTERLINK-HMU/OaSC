cuda_device=1


dataset='mitstates'




CUDA_VISIBLE_DEVICES=$cuda_device python test.py --logpath "./logs/mit_split2_tr" --auc --open_world &> results_mit.txt     


CUDA_VISIBLE_DEVICES=$cuda_device python test.py --logpath "./logs/mit_split2_tr_cw" --auc  &>>results_mit.txt     


CUDA_VISIBLE_DEVICES=$cuda_device python test.py --logpath "./logs/mit_split2_tr_object" --auc  &>> results_mit.txt   


CUDA_VISIBLE_DEVICES=$cuda_device python test.py --logpath "./logs/mit_split2_tr_object_ow" --auc  &>> results_mit.txt 


CUDA_VISIBLE_DEVICES=$cuda_device python test.py --logpath "./logs/mit_split2_tr_tmn" --auc --open_world &>>results_mit.txt     


CUDA_VISIBLE_DEVICES=$cuda_device python test.py --logpath "./logs/mit_split2_cw_tmn" --auc  &>>results_mit.txt     


CUDA_VISIBLE_DEVICES=$cuda_device python test.py --logpath "./logs/mit_split2_tr_object_tmn" --auc  &>>results_mit.txt     

CUDA_VISIBLE_DEVICES=$cuda_device python test.py --logpath "./logs/mit_split2_tr_object_ow_tmn" --auc  &>>results_mit.txt 


CUDA_VISIBLE_DEVICES=$cuda_device python test.py --logpath "./logs/mit_split2_tr_le+" --auc --open_world &>>results_mit.txt     


CUDA_VISIBLE_DEVICES=$cuda_device python test.py --logpath "./logs/mit_split2_cw_le+" --auc  &>>results_mit.txt     


CUDA_VISIBLE_DEVICES=$cuda_device python test.py --logpath "./logs/mit_split2_tr_object_le+" --auc  &>>results_mit.txt     

CUDA_VISIBLE_DEVICES=$cuda_device python test.py --logpath "./logs/mit_split2_tr_object_ow_le+" --auc  &>>results_mit.txt  


CUDA_VISIBLE_DEVICES=$cuda_device python test.py --logpath "./logs/mit_split2_tr_symnet" --auc --open_world &>>results_mit.txt     


CUDA_VISIBLE_DEVICES=$cuda_device python test.py --logpath "./logs/mit_split2_tr_cw_symnet" --auc  &>>results_mit.txt     


CUDA_VISIBLE_DEVICES=$cuda_device python test.py --logpath "./logs/mit_split2_tr_obj_symnet" --auc  &>>results_mit.txt     

CUDA_VISIBLE_DEVICES=$cuda_device python test.py --logpath "./logs/mit_split2_tr_obj_ow_symnet" --auc  &>>results_mit.txt  


CUDA_VISIBLE_DEVICES=$cuda_device python test.py --logpath "./logs/mit_split2_tr_aopp" --auc --open_world &>>results_mit.txt     


CUDA_VISIBLE_DEVICES=$cuda_device python test.py --logpath "./logs/mit_split2_tr_cw_aopp" --auc  &>>results_mit.txt     


CUDA_VISIBLE_DEVICES=$cuda_device python test.py --logpath "./logs/mit_split2_tr_obj_aopp" --auc &>>results_mit.txt     

CUDA_VISIBLE_DEVICES=$cuda_device python test.py --logpath "./logs/mit_split2_tr_obj_ow_aopp" --auc &>>results_mit.txt  