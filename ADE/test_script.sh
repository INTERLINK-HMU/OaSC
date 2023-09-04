cuda_device=1







CUDA_VISIBLE_DEVICES=$cuda_device python test.py --logpath "./logs/zero_shot_split1_tr_ow" --auc  &> "results_osdd.txt"   

 CUDA_VISIBLE_DEVICES=$cuda_device python test.py --logpath "./logs/zero_shot_split1_tr_cw" --auc  &>>"results_osdd.txt"   


 CUDA_VISIBLE_DEVICES=$cuda_device python test.py --logpath "./logs/zero_shot_split1_tr_object" --auc  &>>  "results_osdd.txt"  


 CUDA_VISIBLE_DEVICES=$cuda_device python test.py --logpath "./logs/cgqa_split2_tr_ow" --auc &> "results_cgqa.tx"   

 CUDA_VISIBLE_DEVICES=$cuda_device python test.py --logpath "./logs/cgqa_split2_tr_cw" --auc  &>>"results_cgqa.tx"  


 CUDA_VISIBLE_DEVICES=$cuda_device python test.py --logpath "./logs/cgqa_split2_tr_object" --auc  &>> "results_cgqa.tx"





CUDA_VISIBLE_DEVICES=$cuda_device python test.py --logpath "./logs/mit_split2_tr_ow" --auc &> "results_mit.txt"  

CUDA_VISIBLE_DEVICES=$cuda_device python test.py --logpath "./logs/mit_split2_tr_cw" --auc  &>>"results_mit.txt"  


CUDA_VISIBLE_DEVICES=$cuda_device python test.py --logpath "./logs/mit_split2_tr_object" --auc  &>> "results_mit.txt"  


 



