cd ..
source env/bin/activate
cd src

python3 script.py train_joint_model \
    --joint-model-train \
    --joint-model-cc-model-path ~/cicl/taskA/data/pretrained/cc.pt \
    --joint-model-cc-tokenizer-path ~/cicl/taskA/data/vocab/charlm_vocab_uncondensed.pkl \
    --joint-model-w2v-model-path ~/cicl/taskA/data/pretrained/word2vec.pt \
    --joint-model-w2v-tokenizer-path ~/cicl/taskA/data/vocab/wiki2vec_vocab_500.pkl \
    --joint-model-cc-max-len 15000 \
    --joint-model-w2v-max-len 1000 \
    --joint-model-hidden-size 512 \
    --joint-model-dropout 0.2 \
    --joint-model-batch-size 8 \
    --joint-model-n-epochs 100 \
    --joint-model-save-every 2000 \
    --joint-model-checkpoint-prefix joint-model-test \








