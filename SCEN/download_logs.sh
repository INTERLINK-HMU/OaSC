

mkdir logs





#https://drive.google.com/file/d/1s7Npvw5K6I8WfnVzCeDbaDqRYh4C5NWG/view?usp=sharing
#https://drive.google.com/file/d/1kjopfTwbFtEo-OQoFmFdzvMGvyZi2F5N/view?usp=sharing

fileid="1kjopfTwbFtEo-OQoFmFdzvMGvyZi2F5N"
filename="SCEN_logs.tar.gz"
html=`curl -c ./cookie -s -L "https://drive.google.com/uc?export=download&id=${fileid}"`
curl -Lb ./cookie "https://drive.google.com/uc?export=download&`echo ${html}|grep -Po '(confirm=[a-zA-Z0-9\-_]+)'`&id=${fileid}" -o ${filename}


tar -zxvf SCEN_logs.tar.gz -C logs/
mv logs/*/* logs/

rm -r logs/SCEN_best_logs
rm cookie
rm SCEN_logs.tar.gz 








