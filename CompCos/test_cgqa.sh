cuda_device=1





CUDA_VISIBLE_DEVICES=$cuda_device python test.py --logpath "./logs/cgqa_split2_tr" --auc  &> results_cgqa.txt     


CUDA_VISIBLE_DEVICES=$cuda_device python test.py --logpath "./logs/cgqa_split2_tr_cw" --auc  &>>results_cgqa.txt     


CUDA_VISIBLE_DEVICES=$cuda_device python test.py --logpath "./logs/cgqa_split2_tr_object_ow" --auc  &>> results_cgqa.txt   



CUDA_VISIBLE_DEVICES=$cuda_device python test.py --logpath "./logs/cgqa_split2_tr_tmn" --auc  &>>results_cgqa.txt     


CUDA_VISIBLE_DEVICES=$cuda_device python test.py --logpath "./logs/cgqa_split2_cw_tmn" --auc  &>>results_cgqa.txt     


CUDA_VISIBLE_DEVICES=$cuda_device python test.py --logpath "./logs/cgqa_split2_tr_object_tmn" --auc  &>>results_cgqa.txt     



CUDA_VISIBLE_DEVICES=$cuda_device python test.py --logpath "./logs/cgqa_split2_tr_le+" --auc  &>>results_cgqa.txt     


CUDA_VISIBLE_DEVICES=$cuda_device python test.py --logpath "./logs/cgqa_split2_cw_le+" --auc  &>>results_cgqa.txt     


CUDA_VISIBLE_DEVICES=$cuda_device python test.py --logpath "./logs/cgqa_split2_tr_object_le+" --auc  &>>results_cgqa.txt     




CUDA_VISIBLE_DEVICES=$cuda_device python test.py --logpath "./logs/cgqa_split2_tr_symnet" --auc  &>>results_cgqa.txt     


CUDA_VISIBLE_DEVICES=$cuda_device python test.py --logpath "./logs/cgqa_split2_tr_cw_symnet" --auc  &>>results_cgqa.txt     


CUDA_VISIBLE_DEVICES=$cuda_device python test.py --logpath "./logs/cgqa_split2_tr_obj_ow_symnet" --auc  &>>results_cgqa.txt   
  


CUDA_VISIBLE_DEVICES=$cuda_device python test.py --logpath "./logs/cgqa_split2_tr_aopp" --attr  &>>results_cgqa.txt     


CUDA_VISIBLE_DEVICES=$cuda_device python test.py --logpath "./logs/cgqa_split2_tr_cw_aopp" --auc  &>>results_cgqa.txt     


CUDA_VISIBLE_DEVICES=$cuda_device python test.py --logpath "./logs/cgqa_split2_tr_obj_ow_aopp" --auc &>>results_cgqa.txt   
