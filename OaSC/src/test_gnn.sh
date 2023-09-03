
gnn_type='trgcn'

mode='test'
cuda=1

CUDA_VISIBLE_DEVICES=$cuda python train_gnn.py --label_encoder_type $gnn_type --mode $mode