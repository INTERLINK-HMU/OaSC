cuda_device=0



CUDA_VISIBLE_DEVICES=$cuda_device python test.py --logpath "./logs/vaw_states_ow_tmn"  --auc  &>> results_vaw.txt  


CUDA_VISIBLE_DEVICES=$cuda_device python test.py --logpath "./logs/vaw_states_cw_tmn"  --auc &>> results_vaw.txt  


CUDA_VISIBLE_DEVICES=$cuda_device python test.py --logpath "./logs/vaw_states_tr_object_tmn"   --auc &>> results_vaw.txt  

CUDA_VISIBLE_DEVICES=$cuda_device python test.py --logpath "./logs/vaw_states_object_ow_tmn"   --auc &>> results_vaw.txt  


CUDA_VISIBLE_DEVICES=$cuda_device python test.py --logpath "./logs/vaw_states_ow_le+"   --attr - &>> results_vaw.txt  


CUDA_VISIBLE_DEVICES=$cuda_device python test.py --logpath "./logs/vaw_states_cw_le+"  --auc &>> results_vaw.txt  


CUDA_VISIBLE_DEVICES=$cuda_device python test.py --logpath "./logs/vaw_states_tr_object_le+"   --auc  &>> results_vaw.txt  

CUDA_VISIBLE_DEVICES=$cuda_device python test.py --logpath "./logs/vaw_states_object_ow_le+"   --auc  &>> results_vaw.txt  


CUDA_VISIBLE_DEVICES=$cuda_device python test.py --logpath "./logs/vaw_states_ow_aopp"  --attr --open_world &>> results_vaw.txt  


CUDA_VISIBLE_DEVICES=$cuda_device python test.py --logpath "./logs/vaw_states_cw_aopp"  --attr &>> results_vaw.txt  


CUDA_VISIBLE_DEVICES=$cuda_device python test.py --logpath "./logs/vaw_states_obj_aopp"   --auc &>> results_vaw.txt  

CUDA_VISIBLE_DEVICES=$cuda_device python test.py --logpath "./logs/vaw_states_obj_ow_aopp"   --auc &>> results_vaw.txt  


CUDA_VISIBLE_DEVICES=$cuda_device python test.py --logpath "./logs/vaw_states_ow_symnet"   --auc --open_world &>> results_vaw.txt  


CUDA_VISIBLE_DEVICES=$cuda_device python test.py --logpath "./logs/vaw_states_cw_symnet"  --auc &>> results_vaw.txt  


CUDA_VISIBLE_DEVICES=$cuda_device python test.py --logpath "./logs/vaw_states_obj_symnet"   --auc  &>> results_vaw.txt  

CUDA_VISIBLE_DEVICES=$cuda_device python test.py --logpath "./logs/vaw_states_obj_ow_symnet"   --auc  &>> results_vaw.txt  

CUDA_VISIBLE_DEVICES=$cuda_device python test.py --logpath "./logs/vaw_states_tr_ow"   --auc --open_world &>> results_vaw.txt  


CUDA_VISIBLE_DEVICES=$cuda_device python test.py --logpath "./logs/vaw_states_tr_cw"  --auc &>> results_vaw.txt  


CUDA_VISIBLE_DEVICES=$cuda_device python test.py --logpath "./logs/vaw_states_object"   --auc  &>> results_vaw.txt  

CUDA_VISIBLE_DEVICES=$cuda_device python test.py --logpath "./logs/vaw_states_tr_object_ow"   --auc  &>> results_vaw.txt  

