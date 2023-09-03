cuda_device=0


# type='compcos_known'
# type='compcos_object'


# extra=''
# extra='_object'

# for SPLIT in  'zero_shot_pairs1' 'zero_shot_pairs2' 'zero_shot_pairs3' 'zero_shot_pairs4' 

# do
#         CUDA_VISIBLE_DEVICES=$cuda_device python train.py --config "./configs/$type/osdd/$SPLIT$extra.yml" 


# done




# for SPLIT in  'zero_shot_triplets1' 'zero_shot_triplets2' 'zero_shot_triplets3' 

# do
#         CUDA_VISIBLE_DEVICES=$cuda_device python train.py --config "./configs/$type/osdd/$SPLIT$extra.yml" 


# done

# for SPLIT in  'zero_shot_state1' 'zero_shot_state2' 'zero_shot_state3' 'zero_shot_state4'  'zero_shot_state5' 'zero_shot_state6' 'zero_shot_state7' 'zero_shot_state9'   'zero_shot_state8' 

# do
#         CUDA_VISIBLE_DEVICES=$cuda_device python train.py --config "./configs/$type/osdd/$SPLIT$extra.yml" 


# done


# type='compcos_known'



# extra=''



# for SPLIT in  'glass'  'mug' 'cup' 'bowl' 'book' 'door' 'drawer' 'basket' 'shirt' 'towel' 'newspaper' 'phone' 'socket' 'charger' 
# do
#         CUDA_VISIBLE_DEVICES=$cuda_device python train.py --config "configs/$type/osdd/$SPLIT$extra.yml" 


# done



type='compcos_known'



extra=''

# type='compcos_generic'



# extra='_generic'

for SPLIT in  'zero_shot_pairs1' 'zero_shot_pairs2' 'zero_shot_pairs3' 'zero_shot_pairs4' 

do
        CUDA_VISIBLE_DEVICES=$cuda_device python train.py --config "./configs/$type/osdd/$SPLIT$extra.yml" 


done




for SPLIT in  'zero_shot_triplets1' 'zero_shot_triplets2' 'zero_shot_triplets3' 

do
        CUDA_VISIBLE_DEVICES=$cuda_device python train.py --config "./configs/$type/osdd/$SPLIT$extra.yml" 


done

for SPLIT in  'zero_shot_state1' 'zero_shot_state2' 'zero_shot_state3' 'zero_shot_state4'  'zero_shot_state5' 'zero_shot_state6' 'zero_shot_state7'    'zero_shot_state8'  'zero_shot_state9'

do
        CUDA_VISIBLE_DEVICES=$cuda_device python train.py --config "./configs/$type/osdd/$SPLIT$extra.yml" 


done




type='compcos_object'



extra='_object'

# type='compcos_generic'



# extra='_generic'

for SPLIT in  'zero_shot_pairs1' 'zero_shot_pairs2' 'zero_shot_pairs3' 'zero_shot_pairs4' 

do
        CUDA_VISIBLE_DEVICES=$cuda_device python train.py --config "./configs/$type/osdd/$SPLIT$extra.yml" 


done




for SPLIT in  'zero_shot_triplets1' 'zero_shot_triplets2' 'zero_shot_triplets3' 

do
        CUDA_VISIBLE_DEVICES=$cuda_device python train.py --config "./configs/$type/osdd/$SPLIT$extra.yml" 


done

for SPLIT in  'zero_shot_state1' 'zero_shot_state2' 'zero_shot_state3' 'zero_shot_state4'  'zero_shot_state5' 'zero_shot_state6' 'zero_shot_state7' 'zero_shot_state9'   'zero_shot_state8' 

do
        CUDA_VISIBLE_DEVICES=$cuda_device python train.py --config "./configs/$type/osdd/$SPLIT$extra.yml" 


done

