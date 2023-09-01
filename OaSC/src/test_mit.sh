




saved_weights_path="../saved_checkpoints/mit/finetuned_weights.pth"

#saved_weights_path='./saved_chechkpoints/finetune_mit_states/conceptnet_wordnet_hop1_thresh_10/epoch-49.pth'
dataset='mit'
VARIABLE='closed_filled_folded'
#VARIABLE='open_empty'
test_dir="../datasets/mit_states/test"
embeddings='../embeddings/mit_emb.pred'
#embeddings='..//embeddings/trgcn_seed_24_mit.pred'
#embeddings='..//embeddings/trgcn_seed_111_osdd.pred'

graph_type='conceptnet_wordnet_hop1_thresh_10' 
# saved_weights_path='/media/philippos/26eafc3b-724c-49dc-9a04-90056ad2e9f7/Exps_Code_839_WACV24/OaSC/src/saved_chechkpoints/finetune_cgqa/epoch-349.pth'
# saved_weights_path='//media/philippos/26eafc3b-724c-49dc-9a04-90056ad2e9f7/Exps_Code_839_WACV24/OaSC/saved_checkpoints/mit/finetuned_weights.pth'
#embeddings='../embeddings/trgcn_seed_0_all_1707.pred'

#log_name="../logs/graph_"$dataset"$(date +"%Y_%m_%d_%I_%M_%p").log"
echo  -e "Results for dataset : $dataset " # > $log_name
echo  -e "Using embeddings file : $embeddings " # > $log_name


python test.py --embs-pred $embeddings  --cnn  $saved_weights_path --test-dir $test_dir --variable $VARIABLE --classes-ommited $VARIABLE --dataset $dataset --graph-type $graph_type 

