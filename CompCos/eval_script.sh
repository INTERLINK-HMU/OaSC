cuda_device=1




type='compcos_object'



extra='_object'

open_world='--open_world'




printf -v dated '%(%Y-%m-%d)T' -1

results_file_name="results_"$dated".txt"
for SPLIT in  'zero_shot_pairs1' 'zero_shot_pairs2' 'zero_shot_pairs3'   'zero_shot_pairs4' 

do
            

        echo $SPLIT >> $results_file_name
        echo $open_world >> $results_file_name

        res=$(CUDA_VISIBLE_DEVICES=$cuda_device python test.py --logpath "./logs_ow/$SPLIT$extra"  $open_world) 

        echo "$res" | grep "obj_oracle_match" >>$results_file_name

    



done




for SPLIT in  'zero_shot_triplets1' 'zero_shot_triplets2' 'zero_shot_triplets3'

do

            
        echo $SPLIT >> $results_file_name
        echo $open_world >> $results_file_na

        res=$(CUDA_VISIBLE_DEVICES=$cuda_device python test.py --logpath "./logs_ow/$SPLIT$extra"  $open_world) 

        echo "$res" | grep "obj_oracle_match" >>$results_file_name


done

for SPLIT in  'zero_shot_state1' 'zero_shot_state2' 'zero_shot_state3' 'zero_shot_state4'  'zero_shot_state5' 'zero_shot_state6' 'zero_shot_state7' 'zero_shot_state9'   'zero_shot_state8' 
do
        
        
            
        echo $SPLIT >> $results_file_name
        echo $open_world >> $results_file_na
        res=$(CUDA_VISIBLE_DEVICES=$cuda_device python test.py --logpath "./logs_ow/$SPLIT$extra"  $open_world) 

        echo "$res" | grep "obj_oracle_match" >>$results_file_name


done

