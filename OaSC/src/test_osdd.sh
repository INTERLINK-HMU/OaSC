




saved_weights_path="../saved_checkpoints/osdd/finetuned_weights.pth"


saved_weights_path="../saved_checkpoints/osdd/finetuned_weights.pth"
saved_weights_path='./saved_chechkpoints/finetune_osdd/conceptnet_wordnet_hop1_thresh_10/epoch-49.pth'
dataset='osdd'
VARIABLE='empty_open_folded_filled' 
#VARIABLE='open_empty'
test_dir="../datasets/osdd/test"
embeddings='../embeddings/osdd_emb.pred'
embeddings='..//embeddings/trgcn_seed_111_osdd.pred' 
embeddings='../embeddings/trgcn_seed_111_all_0503.pred' 

#embeddings='..//embeddings/trgcn_seed_24_mit.pred'
#embeddings='..//embeddings/trgcn_seed_111_osdd.pred'

graph_type='conceptnet_wordnet_hop1_thresh_10' 

#embeddings='../embeddings/trgcn_seed_0_all_1707.pred'

#log_name="../logs/graph_"$dataset"$(date +"%Y_%m_%d_%I_%M_%p").log"
echo  -e "Results for dataset : $dataset " # > $log_name
echo  -e "Using embeddings file : $embeddings " # > $log_name


python test.py --embs-pred $embeddings  --cnn  $saved_weights_path --save_to_file "results_$date"\
 --test-dir $test_dir --variable $VARIABLE --classes-ommited $VARIABLE --dataset $dataset --graph-type $graph_type 

