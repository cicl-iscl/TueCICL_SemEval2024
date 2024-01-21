cd ..
source env/bin/activate
cd src

python3 script.py train_char_bilstm \
    --char-bilstm-tokenizer-path ~/cicl/taskC/data/charlm_vocab_uncondensed.pkl \
    --char-bilstm-checkpoint-prefix char-bilstm-128-2 \
    --char-bilstm-tokenizer-max-len 15000 \
    --char-bilstm-save-every 1000 \
    --char-bilstm-epochs-extended 3 \
    --char-bilstm-epochs-pure 2 \
    --char-bilstm-hidden-size 128 \
    --char-bilstm-emb-size 16 \
    --char-bilstm-num-layers 2 \
    --char-bilstm-dropout 0.2 \
    --char-bilstm-batch-size 8 \
    --char-bilstm-train \
