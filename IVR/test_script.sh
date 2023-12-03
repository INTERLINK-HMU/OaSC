
cuda_device=0





CUDA_VISIBLE_DEVICES=$cuda_device python test.py --logpath logs/IVR/osdd --auc  &> "results_osdd.txt"   

CUDA_VISIBLE_DEVICES=$cuda_device python test.py --logpath logs/IVR/osdd_cw --auc  &>> "results_osdd.txt"   

CUDA_VISIBLE_DEVICES=$cuda_device python test.py --logpath logs/IVR/osdd_ow --auc &>> "results_osdd.txt"   



CUDA_VISIBLE_DEVICES=$cuda_device python test.py --logpath logs/IVR/cgqa_obj --auc  &> "results_cgqa.txt"   

CUDA_VISIBLE_DEVICES=$cuda_device python test.py --logpath logs/IVR/cgqa_cw --auc  &>> "results_cgqa.txt"   

CUDA_VISIBLE_DEVICES=$cuda_device python test.py --logpath logs/IVR/cgqa_ow --auc &>> "results_cgqa.txt"   




CUDA_VISIBLE_DEVICES=$cuda_device python test.py --logpath logs/IVR/mit_obj --auc  &> "results_mit.txt"   

CUDA_VISIBLE_DEVICES=$cuda_device python test.py --logpath logs/IVR/mit_cw --auc  &>> "results_mit.txt"   

CUDA_VISIBLE_DEVICES=$cuda_device python test.py --logpath logs/IVR/mit_ow --auc &>> "results_mit.txt"   



CUDA_VISIBLE_DEVICES=$cuda_device python test.py --logpath logs/IVR/vaw_obj --auc  &> "results_vaw.txt"   

CUDA_VISIBLE_DEVICES=$cuda_device python test.py --logpath logs/IVR/vaw_cw --auc  &>> "results_vaw.txt"   

CUDA_VISIBLE_DEVICES=$cuda_device python test.py --logpath logs/IVR/vaw_ow --auc &>> "results_vaw.txt" 

