cd ..
source env/bin/activate
cd src

python3 script.py train_word2vec_classifier \
    --word2vec-classifier-checkpoint-prefix none \
    --word2vec-classifier-hidden-size 512 \
    --word2vec-classifier-num-layers 2 \
    --word2vec-classifier-dropout 0.3 \
    --word2vec-classifier-lr 0.005 \
    --word2vec-classifier-batch-size 8 \
    --word2vec-classifier-n-epochs 20 \
    --word2vec-classifier-start-epoch 1 \
    --word2vec-classifier-save-every 2500 \
    --word2vec-classifier-max-len 1500 \
    --word2vec-classifier-emb-size 500 \
    --word2vec-classifier-load-model ~/cicl/taskA/checkpoints/word2vec-512-2-newtok-nopad/best.pt \
    --word2vec-classifier-tokenizer-pkl-path-vocab ~/cicl/taskA/data/vocab/wiki2vec_vocab_500.pkl \
    --word2vec-classifier-tokenizer-pkl-path-weights ~/cicl/taskA/data/vocab/wiki2vec_weights_500.pkl \
    --word2vec-classifier-predict ~/cicl/taskA/data/predictions/word2vec-classifier.jsonl \
