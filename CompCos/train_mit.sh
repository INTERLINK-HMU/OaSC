cuda_device=1

dataset='mit_states'



CUDA_VISIBLE_DEVICES=$cuda_device python train.py --config "./configs/compcos_dif_val/$dataset/tmn.yml" 


CUDA_VISIBLE_DEVICES=$cuda_device python train.py --config "./configs/compcos_dif_val/$dataset/tmn_cw.yml" 


CUDA_VISIBLE_DEVICES=$cuda_device python train.py --config "./configs/compcos_dif_val/$dataset/tmn_obj.yml" 


CUDA_VISIBLE_DEVICES=$cuda_device python train.py --config "./configs/compcos_dif_val/$dataset/le+.yml" 


CUDA_VISIBLE_DEVICES=$cuda_device python train.py --config "./configs/compcos_dif_val/$dataset/le+_cw.yml" 


CUDA_VISIBLE_DEVICES=$cuda_device python train.py --config "./configs/compcos_dif_val/$dataset/le+_obj.yml" 



CUDA_VISIBLE_DEVICES=$cuda_device python train.py --config "./configs/compcos_dif_val/$dataset/symnet.yml" 


CUDA_VISIBLE_DEVICES=$cuda_device python train.py --config "./configs/compcos_dif_val/$dataset/symnet_cw.yml" 


CUDA_VISIBLE_DEVICES=$cuda_device python train.py --config "./configs/compcos_dif_val/$dataset/symnet_obj_ow.yml" 


CUDA_VISIBLE_DEVICES=$cuda_device python train.py --config "./configs/compcos_dif_val/$dataset/aopp.yml" 


CUDA_VISIBLE_DEVICES=$cuda_device python train.py --config "./configs/compcos_dif_val/$dataset/aopp_cw.yml" 


CUDA_VISIBLE_DEVICES=$cuda_device python train.py --config "./configs/compcos_dif_val/$dataset/aopp_obj_ow.yml" 



CUDA_VISIBLE_DEVICES=$cuda_device python train.py --config "./configs/compcos_dif_val/$dataset/cgqa_split2_tr.yml" 


CUDA_VISIBLE_DEVICES=$cuda_device python train.py --config "./configs/compcos_dif_val/$dataset/cgqa_split2_tr_cw.yml" 


CUDA_VISIBLE_DEVICES=$cuda_device python train.py --config "./configs/compcos_dif_val/$dataset/cgqa_split2_tr_object.yml"

