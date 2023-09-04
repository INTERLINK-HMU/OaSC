



cuda_device=0





CUDA_VISIBLE_DEVICES=$cuda_device python test.py --cfg config/osdd_ow.yml --load saved_models/osdd_ow/osdd_ow_124/best.pth &>> "results_osdd.txt"   

CUDA_VISIBLE_DEVICES=$cuda_device python test.py --cfg config/osdd_cw.yml --load saved_models/osdd_cw/osdd_cw_124/best.pth &>> "results_osdd.txt"  
 
CUDA_VISIBLE_DEVICES=$cuda_device python test.py --cfg config/osdd_obj_ow.yml --load saved_models/osdd_obj/osdd_obj_ow_124/best.pth &>> "results_osdd.txt"  



CUDA_VISIBLE_DEVICES=$cuda_device python test.py --cfg config/cgqa_ow.yml --load saved_models/cgqa_ow/cgqa_cw_124/best.pth   &>> "results_cgqa.txt"   

CUDA_VISIBLE_DEVICES=$cuda_device python test.py --cfg config/cgqa_cw.yml --load saved_models/cgqa_cw/cgqa_cw_124/best.pth   &>> "results_cgqa.txt"   

CUDA_VISIBLE_DEVICES=$cuda_device python test.py --cfg config/cgqa_obj_ow.yml --load saved_models/cgqa_obj/cgqa_obj_ow_124/best.pth &>> "results_cgqa.txt"  




CUDA_VISIBLE_DEVICES=$cuda_device python test.py --cfg config/mit_ow.yml --load saved_models/mit_ow/mit_ow_124/best.pth  &>> "results_mit.txt"  

CUDA_VISIBLE_DEVICES=$cuda_device python test.py --cfg config/mit_cw.yml --load saved_models/mit_cw/mit_cw_124/best.pth  &>> "results_mit.txt"  

CUDA_VISIBLE_DEVICES=$cuda_device python test.py --cfg config/mit_obj_ow.yml --load saved_models/mit_obj/mit_obj_ow_124/best.pth  &>> "results_mit.txt"  