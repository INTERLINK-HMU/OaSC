

mkdir logs





#https://drive.google.com/file/d/1ntSfCzwXlD510dRm3bJnRIxUCP-HMFgy/view?usp=sharing

#https://drive.google.com/file/d/1hWNkEVLtY5L9Il4A35-xKfrTQQsPJlmo/view?usp=sharing
fileid="1hWNkEVLtY5L9Il4A35-xKfrTQQsPJlmo"
filename="IVR_logs.tar.gz"
html=`curl -c ./cookie -s -L "https://drive.google.com/uc?export=download&id=${fileid}"`
curl -Lb ./cookie "https://drive.google.com/uc?export=download&`echo ${html}|grep -Po '(confirm=[a-zA-Z0-9\-_]+)'`&id=${fileid}" -o ${filename}


tar -zxvf IVR_logs.tar.gz -C logs/
mv logs/*/* logs/
rmdir logs/IVR_logs

rm cookie
rm IVR_logs.tar.gz 








