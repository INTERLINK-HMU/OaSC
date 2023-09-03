cuda_device=0





type='compcos_dif_val'



extra=''





closed_world=''

# extra='_generic'

for SPLIT in  'zero_shot_split1_tr_object' 'zero_shot_split2_tr_object'  

do
        CUDA_VISIBLE_DEVICES=$cuda_device python train.py --config "./configs/$type/osdd/$SPLIT$extra$closed_world.yml" 


done




# for SPLIT in  'cgqa_split1' 'cgqa_split2' 

# do
#         CUDA_VISIBLE_DEVICES=$cuda_device python train.py --config "./configs/$type/cgqa_states/$SPLIT$extra$closed_world.yml" 


# done



# for SPLIT in  'cgqa_split1' 'cgqa_split2' 

# do
#         CUDA_VISIBLE_DEVICES=$cuda_device python train.py --config "./configs/$type/cgqa_states/$SPLIT$extra$closed_world.yml" 


# done




type='compcos_object'



extra='_object'

# type='compcos_generic'



# extra='_generic'
closed_world=''

# for SPLIT in  'zero_shot_split1' 'zero_shot_split2'  

# do
#         CUDA_VISIBLE_DEVICES=$cuda_device python train.py --config "./configs/$type/osdd/$SPLIT$extra$closed_world.yml" 


# done




# for SPLIT in  'cgqa_split1' 'cgqa_split2' 

# do
#         CUDA_VISIBLE_DEVICES=$cuda_device python train.py --config "./configs/$type/cgqa_states/$SPLIT$extra$closed_world.yml" 


# done