

mkdir datasets


cd datasets
CURRENT_DIR=$(pwd)
# ##Donwload splits

#https://drive.google.com/file/d/10xe5jAy3stsSi7kf0srEUK3_vfCfREd4/view?usp=drive_link

fileid="10xe5jAy3stsSi7kf0srEUK3_vfCfREd4"
filename="splits.tar.gz"
html=`curl -c ./cookie -s -L "https://drive.google.com/uc?export=download&id=${fileid}"`
curl -Lb ./cookie "https://drive.google.com/uc?export=download&`echo ${html}|grep -Po '(confirm=[a-zA-Z0-9\-_]+)'`&id=${fileid}" -o ${filename}


 tar -zxvf splits.tar.gz

fileid="1AO5f6HSuk6fE0ydK8k370zjcC_WABRqj"
filename="images.tar.gz"
html=`curl -c ./cookie -s -L "https://drive.google.com/uc?export=download&id=${fileid}"`
curl -Lb ./cookie "https://drive.google.com/uc?export=download&`echo ${html}|grep -Po '(confirm=[a-zA-Z0-9\-_]+)'`&id=${fileid}" -o ${filename}


tar -zxvf images.tar.gz



mv dat/* ./
rm -r dat/
tar -zxvf osdd_images.tar.gz -C osdd/
tar -zxvf mit_images.tar.gz -C mit-states/
tar -zxvf cgqa_images.tar.gz -C cgqa_states/

ln -s  $CURRENT_DIR/osdd/images osdd/compcos_dif_val/images
ln -s  $CURRENT_DIR/cgqa_states/images cgqa_states/compcos_dif_val/images
ln -s  $CURRENT_DIR/mit-states/images mit-states/compcos_dif_val/images

rm splits.tar.gz
rm images.tar.gz

rm osdd_images.tar.gz
rm mit_images.tar.gz
rm cgqa_images.tar.gz 








