

mkdir saved_models





#https://drive.google.com/file/d/1wlY_gQ9g48mtMbYKHPHbHtsId_xbUaAc/view?usp=sharing
#https://drive.google.com/file/d/1ELa6xjOGdjyoSN5NzsxADVRfUQ-XnQLv/view?usp=sharing

fileid="1ELa6xjOGdjyoSN5NzsxADVRfUQ-XnQLv"
filename="OaDis_logs.tar.gz"
html=`curl -c ./cookie -s -L "https://drive.google.com/uc?export=download&id=${fileid}"`
curl -Lb ./cookie "https://drive.google.com/uc?export=download&`echo ${html}|grep -Po '(confirm=[a-zA-Z0-9\-_]+)'`&id=${fileid}" -o ${filename}


tar -zxvf OaDis_logs.tar.gz -C saved_models/

rm cookie
rm OaDis_logs.tar.gz 








