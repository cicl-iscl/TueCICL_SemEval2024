cd ..
source env/bin/activate
cd src

python3 script.py train_word2vec_classifier \
    --word2vec-classifier-checkpoint-prefix word2vec-256-3-newtok \
    --word2vec-classifier-hidden-size 256 \
    --word2vec-classifier-num-layers 3 \
    --word2vec-classifier-dropout 0.2 \
    --word2vec-classifier-lr 0.005 \
    --word2vec-classifier-batch-size 8 \
    --word2vec-classifier-n-epochs 10 \
    --word2vec-classifier-start-epoch 1 \
    --word2vec-classifier-save-every 2500 \
    --word2vec-classifier-do-train \
    --word2vec-classifier-max-len 2000 \
    --word2vec-classifier-emb-size 500 \
    --word2vec-classifier-tokenizer-pkl-path-vocab ~/cicl/taskA/data/vocab/wiki2vec_vocab_500.pkl \
    --word2vec-classifier-tokenizer-pkl-path-weights ~/cicl/taskA/data/vocab/wiki2vec_weights_500.pkl
