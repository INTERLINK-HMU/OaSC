

mkdir datasets


cd datasets
CURRENT_DIR=$(pwd)
# ##Donwload splits
#https://drive.google.com/file/d/17-bznLeiNNXc9URRJWqpy2yEabeVwvZP/view?usp=sharing


fileid="17-bznLeiNNXc9URRJWqpy2yEabeVwvZP"
filename="splits.tar.gz"
html=`curl -c ./cookie -s -L "https://drive.google.com/uc?export=download&id=${fileid}"`
curl -Lb ./cookie "https://drive.google.com/uc?export=download&`echo ${html}|grep -Po '(confirm=[a-zA-Z0-9\-_]+)'`&id=${fileid}" -o ${filename}


 tar -zxvf splits.tar.gz
#https://drive.google.com/file/d/1hZeokByPOFWfI-KNUq4GRQ8mvmhir-Ij/view?usp=drive_link
#1OrQHY_tLuEHxL5wo0DSrtYjVbPf0KAEM
#https://drive.google.com/file/d/1Ti0etMidekc7YGzIQ_Waut3d7TVNZIjf/view?usp=sharing
fileid="1OrQHY_tLuEHxL5wo0DSrtYjVbPf0KAEM"
filename="images.tar.gz"
html=`curl -c ./cookie -s -L "https://drive.google.com/uc?export=download&id=${fileid}"`
curl -Lb ./cookie "https://drive.google.com/uc?export=download&`echo ${html}|grep -Po '(confirm=[a-zA-Z0-9\-_]+)'`&id=${fileid}" -o ${filename}


tar -zxvf images.tar.gz



mv dat/* ./
rm -r dat/

tar -zxvf osdd_images.tar.gz -C osdd/

tar -zxvf mit_images.tar.gz -C mit-states/

tar -zxvf cgqa_images.tar.gz -C cgqa_states/

tar -zxvf vaw_images.tar.gz -C vaw_states/

rm  -r vaw_states/images
tar -zxvf vaw_images.tar.gz -C vaw_states/


ln -s  $CURRENT_DIR/osdd/images osdd/compcos_dif_val/images
ln -s  $CURRENT_DIR/cgqa_states/images cgqa_states/compcos_dif_val/images
ln -s  $CURRENT_DIR/mit-states/images mit-states/compcos_dif_val/images



#ln -s  $CURRENT_DIR/vaw/images vaw_states/images

rm  -r  vaw_states/compcos_dif_val/images
ln -s  $CURRENT_DIR/vaw_states/images vaw_states/compcos_dif_val/images

rm  -r  vaw_states/obj_split/images
ln -s  $CURRENT_DIR/vaw_states/images vaw_states/obj_split/images

rm  -r  vaw_states/obj_split/images
ln -s  $CURRENT_DIR/vaw_states/images vaw_states/obj_split/images

rm splits.tar.gz
rm images.tar.gz

rm osdd_images.tar.gz
rm mit_images.tar.gz
rm cgqa_images.tar.gz 
rm vaw_images.tar.gz 
rm vaw_images.tar.gz 








