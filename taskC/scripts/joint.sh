cd ..
source env/bin/activate
cd src

python3 script.py train_joint_model \
    --joint-model-char-tokenizer ~/cicl/taskC/data/charlm_vocab_uncondensed.pkl \
    --joint-model-char-model ~/cicl/taskC/checkpoints/char-bilstm-256-3-res2-train2more/best.pt \
    --joint-model-char-max-len 15000 \
    --joint-model-w2v-tokenizer ~/cicl/taskC/data/wiki2vec_vocab_500.pkl \
    --joint-model-w2v-model ~/cicl/taskC/checkpoints/word2vec-bilstm-512-2-res17pure/best.pt \
    --joint-model-w2v-max-len 5000 \
    --joint-model-checkpoint-prefix joint-model-test-2 \
    --joint-model-save-every-extended 1000 \
    --joint-model-save-every-pure 200 \
    --joint-model-epochs-extended 0 \
    --joint-model-epochs-pure 50 \
    --joint-model-hidden-size 256 \
    --joint-model-dropout 0.2 \
    --joint-model-batch-size 8 \
    --joint-model-train \