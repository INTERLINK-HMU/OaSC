CUDA_VISIBLE_DEVICES=1 python train.py --cfg config/cgqa_cw.yml
CUDA_VISIBLE_DEVICES=1 python train.py --cfg config/cgqa_obj.yml


CUDA_VISIBLE_DEVICES=1 python train.py --cfg config/osdd_cw.yml
CUDA_VISIBLE_DEVICES=1 python train.py --cfg config/osdd_obj.yml
CUDA_VISIBLE_DEVICES=1 python train.py --cfg config/osdd_ow.yml


CUDA_VISIBLE_DEVICES=1 python train.py --cfg config/mit_cw.yml
CUDA_VISIBLE_DEVICES=1 python train.py --cfg config/mit_obj.yml


