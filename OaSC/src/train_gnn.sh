
gnn_type='gcn'
epochs=1
mode='train'
cuda=1

CUDA_VISIBLE_DEVICES=$cuda python train_gnn.py --label_encoder_type $gnn_type --max-epoch $epochs --mode $mode --seed 41182