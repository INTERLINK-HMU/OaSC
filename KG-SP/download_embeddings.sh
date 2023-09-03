

cd datasets

CURRENT_DIR=$(pwd)




mkdir fast
mkdir glove
mkdir w2v

cp $CURRENT_DIR/../utils/download_embeddings.py $CURRENT_DIR/fast


wget --show-progress -O $CURRENT_DIR/glove.6B.zip https://nlp.stanford.edu/data/glove.6B.zip


 # Glove 

unzip   glove.6B.zip  -d glove/







# Glove 
mv data/glove/* glove/

# FastText

python download_fast.py


Word2Vec
cd ../w2v
wget https://figshare.com/ndownloader/files/10798046 -O GoogleNews-vectors-negative300.bin


rm fast/cc.en.300.bin.gz
rm glove/glove.6B.50d.txt
rm glove/glove.6B.100d.txt
rm glove/glove.6B.200d.txt
rm glove.6B.zip 
