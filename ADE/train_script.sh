cuda_device=0







 CUDA_VISIBLE_DEVICES=$cuda_device python test.py --logpath "./logs/zero_shot_split1_tr_ow" --auc  &> "results_osdd.txt"   

 CUDA_VISIBLE_DEVICES=$cuda_device python test.py --logpath "./logs/zero_shot_split1_tr_cw" --auc  &>>"results_osdd.txt"   

# CUDA_VISIBLE_DEVICES=$cuda_device python test.py --logpath "./logs/zero_shot_split1_tr_object" --auc  &>> "results_osdd.txt"  

 CUDA_VISIBLE_DEVICES=$cuda_device python test.py --logpath "./logs/zero_shot_split1_tr_object_ow" --auc  &>>  "results_osdd.txt"  


# CUDA_VISIBLE_DEVICES=$cuda_device python test.py --logpath "./logs/cgqa_split2_tr_ow" --attr --open_world  &> "results_cgqa.tx"   

# CUDA_VISIBLE_DEVICES=$cuda_device python test.py --logpath "./logs/cgqa_split2_tr_cw" --auc  &>>"results_cgqa.tx"  

# CUDA_VISIBLE_DEVICES=$cuda_device python test.py --logpath "./logs/cgqa_split2_tr_object" --auc  &>> "results_cgqa.tx"

# CUDA_VISIBLE_DEVICES=$cuda_device python test.py --logpath "./logs/cgqa_split2_tr_object_ow" --auc  &>> "results_cgqa.tx"





#CUDA_VISIBLE_DEVICES=$cuda_device python train.py --config "./configs/mit_states_ow.yml"

#CUDA_VISIBLE_DEVICES=$cuda_device python train.py --config "./configs/mit_states_cw"
#CUDA_VISIBLE_DEVICES=$cuda_device python train.py --config "./configs/mit_states_obj.yml" 

#CUDA_VISIBLE_DEVICES=$cuda_device python train.py --config "./configs/mit_states_obj_ow.yml" 


 



