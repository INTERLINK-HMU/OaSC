cuda_device=0





type='compcos_known'



extra=''





closed_world='_cw'

# extra='_generic'






for SPLIT in  'mit_split1' 'mit_split2' 

do
        CUDA_VISIBLE_DEVICES=$cuda_device python test.py --logpath "./logs/$SPLIT$closed_world" --attr


done

for SPLIT in  'mit_split1' 'mit_split2' 

do
        CUDA_VISIBLE_DEVICES=$cuda_device python test.py --logpath "./logs/$SPLIT$closed_world" --attr --open_world


done


for SPLIT in  'mit_split1' 'mit_split2' 

do
        CUDA_VISIBLE_DEVICES=$cuda_device python test.py --logpath "./logs/$SPLIT" --attr


done




for SPLIT in  'mit_split_object1' 'mit_split_object2' 

do
        CUDA_VISIBLE_DEVICES=$cuda_device python  test.py --logpath  "./logs/$SPLIT" --attr


done