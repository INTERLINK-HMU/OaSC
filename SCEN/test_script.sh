cuda_device=1







CUDA_VISIBLE_DEVICES=$cuda_device python test.py --logpath  "./logs/zero_shot_split1_tr_ow" --test_batch_size   8 --auc  &> "results_osdd.txt"   

CUDA_VISIBLE_DEVICES=$cuda_device python test.py --logpath "./logs/zero_shot_split1_tr_cw" --test_batch_size   8 --auc  &>> "results_osdd.txt"   

CUDA_VISIBLE_DEVICES=$cuda_device python test.py --logpath "./logs/zero_shot_split1_tr_object" --test_batch_size   8 --auc  &>> "results_osdd.txt"  



CUDA_VISIBLE_DEVICES=$cuda_device python test.py --logpath "./logs/cgqa_split2_tr_ow" --test_batch_size   8  --auc &> "results_cgqa.txt"   

CUDA_VISIBLE_DEVICES=$cuda_device python test.py --logpath "./logs/cgqa_split2_tr_cw" --test_batch_size   8  --auc  &>> "results_cgqa.txt"  


CUDA_VISIBLE_DEVICES=$cuda_device python test.py --logpath "./logs/cgqa_split2_tr_object" --test_batch_size   8  --auc  &>> "results_cgqa.txt"





CUDA_VISIBLE_DEVICES=$cuda_device python test.py --logpath "./logs/mit_split2_tr_ow" --test_batch_size   8  --auc &> "results_mit.txt"  

CUDA_VISIBLE_DEVICES=$cuda_device python test.py --logpath "./logs/mit_split2_tr_cw" --test_batch_size   8  --auc  &>> "results_mit.txt"  

CUDA_VISIBLE_DEVICES=$cuda_device python test.py --logpath "./logs/mit_split2_tr_object" --test_batch_size   8  --auc  &>> "results_mit.txt"  



CUDA_VISIBLE_DEVICES=$cuda_device python test.py --logpath "./logs/vaw_ow" --test_batch_size   8  --auc &> "results_vaw.txt"  

CUDA_VISIBLE_DEVICES=$cuda_device python test.py --logpath "./logs/vaw_cw" --test_batch_size   8  --auc  &>> "results_vaw.txt"  

CUDA_VISIBLE_DEVICES=$cuda_device python test.py --logpath "./logs/vaw_object" --test_batch_size   8  --auc  &>> "results_vaw.txt"  


 



