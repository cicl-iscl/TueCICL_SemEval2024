cd ..
source env/bin/activate
cd src

python3 script.py train_char_bilstm \
    --joint-model-char-tokenizer ~/cicl/taskC/data/charlm_vocab_uncondensed.pkl \
    --joint-model-char-model  \
    --joint-model-w2v-tokenizer  \
    --joint-model-w2v-model  \
    --joint-model-checkpoint-prefix joint-model-test \
    --joint-model-tokenizer-max-len 15000 \
    --joint-model-save-every-extended 1000 \
    --joint-model-save-every-pure 200 \
    --joint-model-epochs-extended 0 \
    --joint-model-epochs-pure 50 \
    --joint-model-hidden-size 256 \
    --joint-model-emb-size 16 \
    --joint-model-num-layers 3 \
    --joint-model-dropout 0.2 \
    --joint-model-batch-size 8 \
    --joint-model-train \
