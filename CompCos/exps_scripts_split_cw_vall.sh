cuda_device=0





type='compcos_known'



extra=''





closed_world='_cw'

# extra='_generic'


suf='zero_shot_split1_cw_vall'

CUDA_VISIBLE_DEVICES=$cuda_device python train.py --config "./configs/$type/osdd/$suf.yml" 



suf='zero_shot_split2_cw_vall'

CUDA_VISIBLE_DEVICES=$cuda_device python train.py --config "./configs/$type/osdd/$suf.yml"





suf='cgqa_split1_cw_vall'

CUDA_VISIBLE_DEVICES=$cuda_device python train.py --config "./configs/$type/cgqa_states/$suf.yml" 



suf='cgqa_split1_cwv_all'

CUDA_VISIBLE_DEVICES=$cuda_device python train.py --config "./configs/$type/cgqa_states/$suf.yml"






for SPLIT in  'cgqa_split1' 'cgqa_split2' 
suf='cgqa_split1_cwv_all'



CUDA_VISIBLE_DEVICES=$cuda_device python train.py --config "./configs/$type/cgqa_states/$SPLIT$closed_world$extra.yml"  --attr 


done




type='compcos_object'



extra='_object'

# type='compcos_generic'



# extra='_generic'
closed_world='_cw'

# for SPLIT in  'zero_shot_split1' 'zero_shot_split2'  

# do
#         CUDA_VISIBLE_DEVICES=$cuda_device python train.py --config "./configs/$type/osdd/$SPLIT$extra$closed_world.yml" --attr 


# done


