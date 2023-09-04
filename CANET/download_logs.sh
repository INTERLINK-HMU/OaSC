

mkdir logs





#https://drive.google.com/file/d/1X3X_Uovj7SgXCq153McDcY3gdSV2og7Q/view?usp=sharing


fileid="1X3X_Uovj7SgXCq153McDcY3gdSV2og7Q"
filename="CANET_logs.tar.gz"
html=`curl -c ./cookie -s -L "https://drive.google.com/uc?export=download&id=${fileid}"`
curl -Lb ./cookie "https://drive.google.com/uc?export=download&`echo ${html}|grep -Po '(confirm=[a-zA-Z0-9\-_]+)'`&id=${fileid}" -o ${filename}


tar -zxvf CANET_logs.tar.gz -C logs/
mv logs/*/* logs/
rm -r logs/CANET_logs
rm cookie
rm CANET_logs.tar.gz 








