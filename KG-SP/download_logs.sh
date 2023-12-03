

mkdir logs





#https://drive.google.com/file/d/1_wFP_cVE4_klmxdM4Gea--kc2ASm-MmO/view?usp=sharing


fileid="1_wFP_cVE4_klmxdM4Gea--kc2ASm-MmO"
filename="KG-SP_logs.tar.gz"
html=`curl -c ./cookie -s -L "https://drive.google.com/uc?export=download&id=${fileid}"`
curl -Lb ./cookie "https://drive.google.com/uc?export=download&`echo ${html}|grep -Po '(confirm=[a-zA-Z0-9\-_]+)'`&id=${fileid}" -o ${filename}


tar -zxvf KG-SP_logs.tar.gz -C logs/
mv logs/*/* logs/

rm -r logs/KG-SP_logs
rm cookie
rm KG-SP_logs.tar.gz 








