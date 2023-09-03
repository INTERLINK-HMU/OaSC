cuda_device=1





type='compcos_known'



extra=''





closed_world='_cw'

# extra='_generic'

for SPLIT in  'mit_split1' 'mit_split2'  

do
        CUDA_VISIBLE_DEVICES=$cuda_device python train.py --config "./configs/$type/mit_states/$SPLIT$extra.yml" 


done



for SPLIT in  'mit_split1' 'mit_split2'  

do
        CUDA_VISIBLE_DEVICES=$cuda_device python train.py --config "./configs/$type/mit_states/$SPLIT$extra$closed_world.yml" 


done











# for SPLIT in  'mit_split_object1' 'mit_split_object2'  

# do
#         CUDA_VISIBLE_DEVICES=$cuda_device python train.py --config "./configs/$type/mit_states/$SPLIT$extra.yml" 


# done


