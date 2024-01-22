cd ..
source env/bin/activate
cd src

python3 script.py train_word2vec_bilstm \
    --word2vec-bilstm-tokenizer-vocab ~/cicl/taskC/data/wiki2vec_vocab_500.pkl \
    --word2vec-bilstm-tokenizer-weights ~/cicl/taskC/data/wiki2vec_weights_500.pkl \
    --word2vec-bilstm-checkpoint-prefix word2vec-bilstm-256-2-pure-only \
    --word2vec-bilstm-tokenizer-max-len 15000 \
    --word2vec-bilstm-epochs-extended 0 \
    --word2vec-bilstm-save-every-extended 3000 \
    --word2vec-bilstm-epochs-pure 50 \
    --word2vec-bilstm-save-every-pure 250 \
    --word2vec-bilstm-hidden-size 256 \
    --word2vec-bilstm-emb-size 500 \
    --word2vec-bilstm-num-layers 2 \
    --word2vec-bilstm-dropout 0.2 \
    --word2vec-bilstm-batch-size 6 \
    --word2vec-bilstm-train \
