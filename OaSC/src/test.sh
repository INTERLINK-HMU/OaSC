


saved_weights_path='../saved_checkpoints/osdd/finetuned_weights.pth'
dataset='osdd'
VARIABLE='empty_open_folded_filled' 
test_dir="../single_label_classification_full_state_14_splits_nedir/zero_shot/$VARIABLE/test"
embeddings='..//embeddings/trgcn_seed_111_osdd.pred'


log_name="../logs/graph_"$dataset"$(date +"%Y_%m_%d_%I_%M_%p").log"
echo  -e "Results for dataset : $dataset " # > $log_name
echo  -e "Using embeddings file : $embeddings " # > $log_name

python test.py  --embs-pred $embeddings --cnn  $saved_weights_path --save_to_file "results_$date"  --test-dir $test_dir  --variable $VARIABLE --classes-ommited $VARIABLE --dataset $dataset

            


saved_weights_path='../saved_checkpoints/cgqa/finetuned_weights.pth'
dataset='cgqa'
VARIABLE='closed_filled_folded'
test_dir="/media/philippos/26eafc3b-724c-49dc-9a04-90056ad2e9f7/Exps_Code_839_WACV24/OaSC/Material_for_save/datasets/cgqa/$VARIABLE/test"
embeddings='../embeddings/trgcn_seed_24_cgqa.pred'
#log_name="../logs/graph_"$dataset"$(date +"%Y_%m_%d_%I_%M_%p").log"
echo  -e "Results for dataset : $dataset " # > $log_name
echo  -e "Using embeddings file : $embeddings " # > $log_name


python test.py --embs-pred $embeddings  --cnn  $saved_weights_path --save_to_file "results_$date"  --test-dir $test_dir --variable $VARIABLE --classes-ommited $VARIABLE --dataset $dataset

            
saved_weights_path='../saved_checkpoints/mit/finetuned_weights.pth'
dataset='mit'
VARIABLE='closed_filled_folded'
test_dir="/media/philippos/26eafc3b-724c-49dc-9a04-90056ad2e9f7/Exps_Code_839_WACV24/OaSC/Material_for_save/datasets/mit_states/$VARIABLE/test"
embeddings='../embeddings/trgcn_seed_24_mit.pred'
log_name="../logs/graph_"$dataset"$(date +"%Y_%m_%d_%I_%M_%p").log"
echo  -e "Results for dataset : $dataset " # > $log_name
echo  -e "Using embeddings file : $embeddings " # > $log_name


python test.py  --embs-pred $embeddings --cnn  $saved_weights_path --save_to_file "results_$date"  --test-dir $test_dir --variable $VARIABLE --classes-ommited $VARIABLE --dataset $dataset

            
