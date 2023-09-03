




saved_weights_path="../saved_checkpoints/cgqa/finetuned_weights.pth"


dataset='cgqa'
VARIABLE='closed_filled_folded'
#VARIABLE='open_empty'
test_dir="../datasets/cgqa//test"
embeddings='../embeddings/cgqa_emb.pred'

graph_type='conceptnet_wordnet_hop1_thresh_10' 

echo  -e "Results for dataset : $dataset " # > $log_name
echo  -e "Using embeddings file : $embeddings " # > $log_name


python test.py --embs-pred $embeddings  --cnn  $saved_weights_path --save_to_file "results_$date" --test-dir $test_dir --variable $VARIABLE --classes-ommited $VARIABLE --dataset $dataset --graph-type $graph_type 

