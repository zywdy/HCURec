mkdir data
cd data
mkdir train dev test raw result checkpoint
(/root/.local/bin/)kaggle datasets download takuok/glove840b300dtxt
unzip glove840b300dtxt.zip
rm glove840b300dtxt.zip
cd train
wget https://mind201910small.blob.core.windows.net/release/MINDsmall_train.zip
unzip MINDsmall_train.zip
rm MINDsmall_train.zip
cd ../test
wget https://mind201910small.blob.core.windows.net/release/MINDsmall_dev.zip
unzip MINDsmall_dev.zip
rm MINDsmall_dev.zip
