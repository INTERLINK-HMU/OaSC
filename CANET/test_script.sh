

cuda_device=0




CUDA_VISIBLE_DEVICES=$cuda_device python test.py --dataset osdd_obj --auc  &> "results_osdd.txt"   

CUDA_VISIBLE_DEVICES=$cuda_device python test.py --dataset osdd_ow --auc  &>> "results_osdd.txt"   

CUDA_VISIBLE_DEVICES=$cuda_device python test.py --dataset osdd --auc &>> "results_osdd.txt"   



CUDA_VISIBLE_DEVICES=$cuda_device python test.py --dataset cgqa_obj --auc  &> "results_cgqa.txt"   

CUDA_VISIBLE_DEVICES=$cuda_device python test.py --dataset cgqa_states --auc  &>> "results_cgqa.txt"   

CUDA_VISIBLE_DEVICES=$cuda_device python test.py --dataset cgqa_states_ow --auc &>> "results_cgqa.txt"   




CUDA_VISIBLE_DEVICES=$cuda_device python test.py --dataset mit_obj --auc  &> "results_mit.txt"   

CUDA_VISIBLE_DEVICES=$cuda_device python test.py --dataset mit_cw --auc  &>> "results_mit.txt"   

CUDA_VISIBLE_DEVICES=$cuda_device python test.py --dataset mit_ow --auc &>> "results_mit.txt"   



CUDA_VISIBLE_DEVICES=$cuda_device python test.py --dataset vaw_obj --auc  &> "results_vaw.txt"   

CUDA_VISIBLE_DEVICES=$cuda_device python test.py --dataset vaw_cw --auc  &>> "results_vaw.txt"   

CUDA_VISIBLE_DEVICES=$cuda_device python test.py --dataset vaw_ow --auc &>> "results_vaw.txt"   


