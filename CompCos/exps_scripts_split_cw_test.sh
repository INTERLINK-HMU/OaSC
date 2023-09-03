cuda_device=0





type='compcos_known'



extra=''





closed_world='_cw'

# extra='_generic'

for SPLIT in  'zero_shot_split1' 'zero_shot_split2'  

do
        CUDA_VISIBLE_DEVICES=$cuda_device python test.py --logpath "./logs/$SPLIT$extra$closed_world" --attr


done




for SPLIT in  'cgqa_split1' 'cgqa_split2' 

do
        CUDA_VISIBLE_DEVICES=$cuda_device python test.py --logpath "./logs/$SPLIT$extra$closed_world" --attr


done

for SPLIT in  'mit_split1' 'mit_split2' 

do
        CUDA_VISIBLE_DEVICES=$cuda_device python  test.py --logpath  "./logs/$SPLIT$closed_world$extra" --attr


done



type='compcos_object'



extra='_object'

# type='compcos_generic'



# extra='_generic'
closed_world='_cw'

for SPLIT in  'zero_shot_split1' 'zero_shot_split2'  

do
        CUDA_VISIBLE_DEVICES=$cuda_device python  test.py --logpath  "./logs/$SPLIT$extra$closed_world" --attr


done




for SPLIT in  'cgqa_split1' 'cgqa_split2' 

do
        CUDA_VISIBLE_DEVICES=$cuda_device python  test.py --logpath  "./logs/$SPLIT$closed_world$extra" --attr


done



for SPLIT in  'mit_split_cw_object1' 'mit_split_cw_object2' 

do
        CUDA_VISIBLE_DEVICES=$cuda_device python  test.py --logpath  "./logs/$SPLIT" --attr


done