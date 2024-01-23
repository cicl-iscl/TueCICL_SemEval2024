mkdir data
wget https://s3.tebi.io/winkler.stuff/subtaskC_train.jsonl -O data/subtaskC_train.jsonl
wget https://s3.tebi.io/winkler.stuff/subtaskC_dev.jsonl -O data/subtaskC_dev.jsonl
wget https://s3.tebi.io/winkler.stuff/subtaskA_train_monolingual.jsonl -O data/subtaskA_train_monolingual.jsonl
wget http://wikipedia2vec.s3.amazonaws.com/models/en/2018-04-20/enwiki_20180420_500d.txt.bz2 -O data/enwiki_20180420_500d.txt.bz2
bzip2 -d data/enwiki_20180420_500d.txt.bz2