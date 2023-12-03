

mkdir logs





#https://drive.google.com/file/d/1JNULVjSe8-QMWJ-p2zTnX1-DLSLXl9Wm/view?usp=sharing
#https://drive.google.com/file/d/1MKrWr-24SZ16-3dptx1Tn2vD0SM0CDYZ/view?usp=sharing

fileid="1MKrWr-24SZ16-3dptx1Tn2vD0SM0CDYZ"
filename="CompCos_logs.tar.gz"
html=`curl -c ./cookie -s -L "https://drive.google.com/uc?export=download&id=${fileid}"`
curl -Lb ./cookie "https://drive.google.com/uc?export=download&`echo ${html}|grep -Po '(confirm=[a-zA-Z0-9\-_]+)'`&id=${fileid}" -o ${filename}


tar -zxvf CompCos_logs.tar.gz -C logs/
mv logs/*/* logs/
rmdir logs/CompCos2_logs

rm cookie
rm CompCos_logs.tar.gz 








