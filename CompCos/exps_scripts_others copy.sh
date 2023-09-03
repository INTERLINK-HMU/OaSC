cuda_device=0





CUDA_VISIBLE_DEVICES=$cuda_device python train.py --config "./configs/compcos_dif_val/osdd/tmn.yml" 


CUDA_VISIBLE_DEVICES=$cuda_device python train.py --config "./configs/compcos_dif_val/osdd/tmn_cw.yml" 


CUDA_VISIBLE_DEVICES=$cuda_device python train.py --config "./configs/compcos_dif_val/osdd/tmn_obj.yml" 


CUDA_VISIBLE_DEVICES=$cuda_device python train.py --config "./configs/compcos_dif_val/osdd/le+.yml" 


CUDA_VISIBLE_DEVICES=$cuda_device python train.py --config "./configs/compcos_dif_val/osdd/le+_cw.yml" 


CUDA_VISIBLE_DEVICES=$cuda_device python train.py --config "./configs/compcos_dif_val/osdd/le+_obj.yml" 
