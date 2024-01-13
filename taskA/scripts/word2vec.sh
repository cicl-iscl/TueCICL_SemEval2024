cd ..
source env/bin/activate
cd src

python3 script.py train_word2vec_classifier \
    --word2vec-classifier-checkpoint-prefix word2vec-test \
    --word2vec-classifier-hidden-size 256 \
    --word2vec-classifier-num-layers 1 \
    --word2vec-classifier-dropout 0.5 \
    --word2vec-classifier-lr 0.001 \
    --word2vec-classifier-batch-size 6 \
    --word2vec-classifier-n-epochs 100 \
    --word2vec-classifier-start-epoch 1 \
    --word2vec-classifier-save-every 2000 \
    --word2vec-classifier-do-train 1 \
    --word2vec-classifier-max-len 15000 \
    --word2vec-classifier-tokenizer-txt-path /Volumes/Aron\'s\ Hard\ Drive/codebase/enwiki_20180420_500d.txt \
    --word2vec-classifier-tokenizer-save-path ~/codebase/uniwork/cicl/taskA/data/word2vec_500.pkl \
    --word2vec-classifier-emb-size 500 \
    --word2vec-classifier-tokenizer-extend 1
