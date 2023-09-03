



cuda_device=0





CUDA_VISIBLE_DEVICES=$cuda_device python test.py --cfg config/osdd_ow.yml --load osdd_ow/osdd_ow_124/best.pth &>> "results_osdd.txt"   

CUDA_VISIBLE_DEVICES=$cuda_device python test.py --cfg config/osdd_cw.yml --load osdd_cw/osdd_cw_124/best.pth &>> "results_osdd.txt"  
 
CUDA_VISIBLE_DEVICES=$cuda_device python test.py --cfg config/osdd_obj_ow.yml --load osdd_obj/osdd_obj_ow_124/best.pth &>> "results_osdd.txt"  



CUDA_VISIBLE_DEVICES=$cuda_device python test.py --cfg config/cgqa_ow.yml --load cgqa_ow/cgqa_cw_124/best.pth   &>> "results_cgqa.txt"   

CUDA_VISIBLE_DEVICES=$cuda_device python test.py --cfg config/cgqa_cw.yml --load cgqa_cw/cgqa_cw_124/best.pth   &>> "results_cgqa.txt"   

CUDA_VISIBLE_DEVICES=$cuda_device python test.py --cfg config/cgqa_obj_ow.yml --load cgqa_obj/cgqa_obj_ow_124/best.pth &>> "results_cgqa.txt"  




CUDA_VISIBLE_DEVICES=$cuda_device python test.py --cfg config/mit_ow.yml --load mit_ow/mit_ow_124/best.pth  &>> "results_mit.txt"  

CUDA_VISIBLE_DEVICES=$cuda_device python test.py --cfg config/mit_cw.yml --load mit_cw/mit_cw_124/best.pth  &>> "results_mit.txt"  

CUDA_VISIBLE_DEVICES=$cuda_device python test.py --cfg config/mit_obj_ow.yml --load mit_obj/mit_obj_ow_124/best.pth  &>> "results_mit.txt"  