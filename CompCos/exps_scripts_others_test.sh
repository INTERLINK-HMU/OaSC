cuda_device=1


CUDA_VISIBLE_DEVICES=$cuda_device python test.py --logpath "./logs/zero_shot_split1_tr_tmn"  --auc --open_world &> results.txt  


CUDA_VISIBLE_DEVICES=$cuda_device python test.py --logpath "./logs/zero_shot_split1_tr_cw_tmn"  --auc &>> results.txt  


CUDA_VISIBLE_DEVICES=$cuda_device python test.py --logpath "./logs/zero_shot_split1_tr_object_tmn"   --auc &>> results.txt  

CUDA_VISIBLE_DEVICES=$cuda_device python test.py --logpath "./logs/zero_shot_split1_tr_object_ow_tmn"   --auc &>> results.txt  


CUDA_VISIBLE_DEVICES=$cuda_device python test.py --logpath "./logs/zero_shot_split1_tr_le+"   --attr --open_world &>> results.txt  


CUDA_VISIBLE_DEVICES=$cuda_device python test.py --logpath "./logs/zero_shot_split1_tr_cw_le+"  --auc &>> results.txt  


CUDA_VISIBLE_DEVICES=$cuda_device python test.py --logpath "./logs/zero_shot_split1_tr_object_le+"   --auc  &>> results.txt  

CUDA_VISIBLE_DEVICES=$cuda_device python test.py --logpath "./logs/zero_shot_split1_tr_object_ow_le+"   --auc  &>> results.txt  


CUDA_VISIBLE_DEVICES=$cuda_device python test.py --logpath "./logs/zero_shot_split1_tr_aopp"  --auc --open_world &>> results.txt  


CUDA_VISIBLE_DEVICES=$cuda_device python test.py --logpath "./logs/zero_shot_split1_tr_cw_aopp"  --attr &>> results.txt  


CUDA_VISIBLE_DEVICES=$cuda_device python test.py --logpath "./logs/zero_shot_split1_tr_obj_aopp"   --auc &>> results.txt  

CUDA_VISIBLE_DEVICES=$cuda_device python test.py --logpath "./logs/zero_shot_split1_tr_obj_ow_aopp"   --auc &>> results.txt  


CUDA_VISIBLE_DEVICES=$cuda_device python test.py --logpath "./logs/zero_shot_split1_tr_symnet"   --auc --open_world &>> results.txt  


CUDA_VISIBLE_DEVICES=$cuda_device python test.py --logpath "./logs/zero_shot_split1_tr_cw_symnet"  --auc &>> results.txt  


CUDA_VISIBLE_DEVICES=$cuda_device python test.py --logpath "./logs/zero_shot_split1_tr_obj_symnet"   --auc  &>> results.txt  

CUDA_VISIBLE_DEVICES=$cuda_device python test.py --logpath "./logs/zero_shot_split1_tr_obj_ow_symnet"   --auc  &>> results.txt  

CUDA_VISIBLE_DEVICES=$cuda_device python test.py --logpath "./logs/zero_shot_split1_tr"   --attr --open_world &>> results.txt  


CUDA_VISIBLE_DEVICES=$cuda_device python test.py --logpath "./logs/zero_shot_split1_tr_cw"  --auc &>> results.txt  


CUDA_VISIBLE_DEVICES=$cuda_device python test.py --logpath "./logs/zero_shot_split1_tr_object"   --auc  &>> results.txt  

CUDA_VISIBLE_DEVICES=$cuda_device python test.py --logpath "./logs/zero_shot_split1_tr_object_ow"   --auc  &>> results.txt  

