cuda_device=0


CUDA_VISIBLE_DEVICES=$cuda_device python test.py --logpath "./logs/zero_shot_split1_tr_tmn"  --auc  &> results_osdd.txt  


CUDA_VISIBLE_DEVICES=$cuda_device python test.py --logpath "./logs/zero_shot_split1_tr_cw_tmn"  --auc &>> results_osdd.txt  


CUDA_VISIBLE_DEVICES=$cuda_device python test.py --logpath "./logs/zero_shot_split1_tr_object_tmn"   --auc &>> results_osdd.txt  



CUDA_VISIBLE_DEVICES=$cuda_device python test.py --logpath "./logs/zero_shot_split1_tr_le+"   --auc  &>> results_osdd.txt  


CUDA_VISIBLE_DEVICES=$cuda_device python test.py --logpath "./logs/zero_shot_split1_tr_cw_le+"  --auc &>> results_osdd.txt  


CUDA_VISIBLE_DEVICES=$cuda_device python test.py --logpath "./logs/zero_shot_split1_tr_object_le+"   --auc  &>> results_osdd.txt  



CUDA_VISIBLE_DEVICES=$cuda_device python test.py --logpath "./logs/zero_shot_split1_tr_aopp"  --auc  &>> results_osdd.txt  


CUDA_VISIBLE_DEVICES=$cuda_device python test.py --logpath "./logs/zero_shot_split1_tr_cw_aopp"  --auc &>> results_osdd.txt  


CUDA_VISIBLE_DEVICES=$cuda_device python test.py --logpath "./logs/zero_shot_split1_tr_obj_aopp"   --auc &>> results_osdd.txt  



CUDA_VISIBLE_DEVICES=$cuda_device python test.py --logpath "./logs/zero_shot_split1_tr_symnet"   --auc  &>> results_osdd.txt  


CUDA_VISIBLE_DEVICES=$cuda_device python test.py --logpath "./logs/zero_shot_split1_tr_cw_symnet"  --auc &>> results_osdd.txt  


CUDA_VISIBLE_DEVICES=$cuda_device python test.py --logpath "./logs/zero_shot_split1_tr_obj_symnet"   --auc  &>> results_osdd.txt  


CUDA_VISIBLE_DEVICES=$cuda_device python test.py --logpath "./logs/zero_shot_split1_tr"   --auc &>> results_osdd.txt  


CUDA_VISIBLE_DEVICES=$cuda_device python test.py --logpath "./logs/zero_shot_split1_tr_cw"  --auc &>> results_osdd.txt  


CUDA_VISIBLE_DEVICES=$cuda_device python test.py --logpath "./logs/zero_shot_split1_tr_object"   --auc  &>> results_osdd.txt  


