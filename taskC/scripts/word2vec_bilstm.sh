cd ..
source env/bin/activate
cd src

python3 script.py train_word2vec_bilstm \
    --word2vec-bilstm-tokenizer-path ~/cicl/taskC/data/enwiki_20180420_500d.txt \
    --word2vec-bilstm-save-vocab ~/cicl/taskC/data/wiki2vec_vocab_500.pkl \
    --word2vec-bilstm-checkpoint-prefix word2vec-bilstm-test \
    --word2vec-bilstm-tokenizer-max-len 15000 \
    --word2vec-bilstm-save-every-extended 1000 \
    --word2vec-bilstm-save-every-pure 200 \
    --word2vec-bilstm-epochs-extended 0 \
    --word2vec-bilstm-epochs-pure 50 \
    --word2vec-bilstm-hidden-size 256 \
    --word2vec-bilstm-emb-size 16 \
    --word2vec-bilstm-num-layers 2 \
    --word2vec-bilstm-dropout 0.2 \
    --word2vec-bilstm-batch-size 4 \
    --word2vec-bilstm-train \
