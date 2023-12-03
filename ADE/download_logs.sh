
wget https://dl.fbaipublicfiles.com/dino/dino_vitbase16_pretrain/dino_vitbase16_pretrain.pth

mkdir pretrain

mv dino_vitbase16_pretrain.pth pretrain/



mkdir logs





#https://drive.google.com/file/d/1Iz5-LmNpDzUyC4am2M4cEr69FvSGXrCu/view?usp=sharing
#https://drive.google.com/file/d/1cVGGIoctEPDDIw8Hode1glmMtsxFz_B5/view?usp=sharing

fileid="1cVGGIoctEPDDIw8Hode1glmMtsxFz_B5"
filename="ADE_logs.tar.gz"
html=`curl -c ./cookie -s -L "https://drive.google.com/uc?export=download&id=${fileid}"`
curl -Lb ./cookie "https://drive.google.com/uc?export=download&`echo ${html}|grep -Po '(confirm=[a-zA-Z0-9\-_]+)'`&id=${fileid}" -o ${filename}


tar -zxvf ADE_logs.tar.gz -C logs/
mv logs/*/* logs/
rmdir logs/ADE_logs


rm cookie
rm ADE_logs.tar.gz 







