cd ..
source env/bin/activate
cd src

python3 script.py train_word2vec_classifier \
    --word2vec-classifier-checkpoint-prefix word2vec-128-2-unext \
    --word2vec-classifier-hidden-size 128 \
    --word2vec-classifier-num-layers 2 \
    --word2vec-classifier-dropout 0.1 \
    --word2vec-classifier-lr 0.001 \
    --word2vec-classifier-batch-size 6 \
    --word2vec-classifier-n-epochs 10 \
    --word2vec-classifier-start-epoch 1 \
    --word2vec-classifier-save-every 20 \
    --word2vec-classifier-do-train \
    --word2vec-classifier-max-len 1000 \
    --word2vec-classifier-tokenizer-pkl-path-weights ~/cicl/taskA/data/w2v_500_weights.pkl \
    --word2vec-classifier-tokenizer-pkl-path-vocab ~/cicl/taskA/data/w2v_500_vocab.pkl \
    --word2vec-classifier-emb-size 500
