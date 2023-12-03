

#mkdir data
cd datasets
CURRENT_DIR=$(pwd)




mkdir fast
mkdir glove
mkdir w2v




#  # Glove 

wget --show-progress -O $CURRENT_DIR/glove.6B.zip https://nlp.stanford.edu/data/glove.6B.zip




unzip   glove.6B.zip  -d glove/








# # FastText
python ../download_fast.py
mv cc.en.300.bin fast/

Word2Vec
cd ../w2v
wget https://figshare.com/ndownloader/files/10798046 -O GoogleNews-vectors-negative300.bin

mv GoogleNews-vectors-negative300.bin w2v/

cd $CURRENT_DIR

rm glove/glove.6B.50d.txt
rm glove/glove.6B.100d.txt
rm glove/glove.6B.200d.txt
rm glove.6B.zip 
#rm cc.en.300.bin.gz

