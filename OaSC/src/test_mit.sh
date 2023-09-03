




saved_weights_path="../saved_checkpoints/mit/finetuned_weights.pth"

dataset='mit'
VARIABLE='closed_filled_folded'
test_dir="../datasets/mit_states/test"
embeddings='../embeddings/mit_emb.pred'


graph_type='conceptnet_wordnet_hop1_thresh_10' 


#log_name="../logs/graph_"$dataset"$(date +"%Y_%m_%d_%I_%M_%p").log"
echo  -e "Results for dataset : $dataset " # > $log_name
echo  -e "Using embeddings file : $embeddings " # > $log_name


python test.py --embs-pred $embeddings  --cnn  $saved_weights_path --test-dir $test_dir --variable $VARIABLE --classes-ommited $VARIABLE --dataset $dataset --graph-type $graph_type 

