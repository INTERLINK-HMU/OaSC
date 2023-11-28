




saved_weights_path="../saved_checkpoints/vaw/finetuned_weights.pth"
dataset='vaw'
VARIABLE='empty_open_folded_filled' 
test_dir="../datasets/vaw/test"
embeddings='../embeddings/osdd_emb.pred'

cuda_device=0

graph_type='conceptnet_hop1_thresh_10' 

echo  -e "Results for dataset : $dataset " # > $log_name
echo  -e "Using embeddings file : $embeddings " # > $log_name


CUDA_VISIBLE_DEVICES=$cuda_device python test.py --embs-pred $embeddings  --cnn  $saved_weights_path --save_to_file "results_$date"\
 --test-dir $test_dir --variable $VARIABLE --classes-ommited $VARIABLE --dataset $dataset --graph-type $graph_type 

