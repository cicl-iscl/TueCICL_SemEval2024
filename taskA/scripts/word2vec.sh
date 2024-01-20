cd ..
source env/bin/activate
cd src

python3 script.py train_word2vec_classifier \
    --word2vec-classifier-checkpoint-prefix word2vec-256-2-unext \
    --word2vec-classifier-hidden-size 256 \
    --word2vec-classifier-num-layers 2 \
    --word2vec-classifier-dropout 0.1 \
    --word2vec-classifier-lr 0.001 \
    --word2vec-classifier-batch-size 8 \
    --word2vec-classifier-n-epochs 10 \
    --word2vec-classifier-start-epoch 1 \
    --word2vec-classifier-save-every 3000 \
    --word2vec-classifier-do-train \
    --word2vec-classifier-max-len 2000 \
    --word2vec-classifier-emb-size 500 \
    --word2vec-classifier-tokenizer-pkl-path-vocab ~/cicl/taskA/data/w2v_500_vocab_unext.pkl \
    --word2vec-classifier-tokenizer-pkl-path-weights ~/cicl/taskA/data/w2v_500_weights_unext.pkl
    # --word2vec-classifier-tokenizer-save-path-weights ~/cicl/taskA/data/w2v_500_weights_unext.pkl \
    # --word2vec-classifier-tokenizer-save-path-vocab ~/cicl/taskA/data/w2v_500_vocab_unext.pkl \
    # --word2vec-classifier-tokenizer-txt-path ~/cicl/taskA/data/enwiki_20180420_500d.txt
    # --word2vec-classifier-tokenizer-extend
