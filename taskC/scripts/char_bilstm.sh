cd ..
source env/bin/activate
cd src

python3 script.py train_char_bilstm \
    --char-bilstm-tokenizer-path ~/cicl/taskC/data/charlm_vocab_uncondensed.pkl \
    --char-bilstm-checkpoint-prefix char-bilstm-512-2-spacefix \
    --char-bilstm-tokenizer-max-len 10000 \
    --char-bilstm-save-every-extended 2000 \
    --char-bilstm-save-every-pure 200 \
    --char-bilstm-epochs-extended 5 \
    --char-bilstm-epochs-pure 50 \
    --char-bilstm-hidden-size 512 \
    --char-bilstm-emb-size 8 \
    --char-bilstm-num-layers 2 \
    --char-bilstm-dropout 0.2 \
    --char-bilstm-batch-size 8 \
    --char-bilstm-train \
    --char-bilstm-prefer-cuda-device 0 \
