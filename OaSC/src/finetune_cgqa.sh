



dataset='cgqa'
VARIABLE='closed_folded_filled' 

train_dir="../datasets/$dataset/train"
embeddings='../embeddings/cgqa_emb.pred'


save_path="saved_checkpoints/"
num_epochs=350
batch_size=32
cuda_device=0
graph_type='conceptnet_wordnet_hop1_thresh_10' 

CUDA_VISIBLE_DEVICES=$cuda_device python finetune.py  --embs-pred $embeddings --cnn  $save_path --save_to_file "results_$date"\
 --train-dir $train_dir  --variable $VARIABLE --classes-ommited $VARIABLE --dataset $dataset --num-epochs\
 $num_epochs --batch $batch_size --save-path $save_path  --graph-type $graph_type 

            



