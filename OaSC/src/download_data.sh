


CURRENT_DIR=$(pwd)


fileid="1i-jaMPKoBOmBncKBQ3qpsGQ9pLWgwRbz"
filename="OaSC_material.tar.gz"
html=`curl -c ./cookie -s -L "https://drive.google.com/uc?export=download&id=${fileid}"`
curl -Lb ./cookie "https://drive.google.com/uc?export=download&`echo ${html}|grep -Po '(confirm=[a-zA-Z0-9\-_]+)'`&id=${fileid}" -o ${filename}


 tar -zxvf OaSC_material.tar.gz





tar -zxvf datasets.tar.gz -C ../
tar -zxvf data.tar.gz -C 
tar -zxvf embeddings.tar.gz -C ../
tar -zxvf saved_checkpoints.tar.gz -C ../

rm OaSC_material.tar.gz
rm data.tar.gz

rm datasets.tar.gz
rm embeddings.tar.gz
rm saved_checkpoints.tar.gz 
rm cookie







