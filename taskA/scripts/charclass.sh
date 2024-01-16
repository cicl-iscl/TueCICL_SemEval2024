cd ..
source env/bin/activate
cd src

python3 script.py train_char_classifier \
    --char-class-do-train 1 \
    --char-class-checkpoint-prefix cc-128-2 \
    --char-class-num-layers 2 \
    --char-class-hidden-size 128 \
    --char-class-save-every 3000 \
    --char-class-emb-size 8 \
    --char-class-lr 0.001 \
    --char-class-start-epoch 1 \
    --char-class-n-epochs 10 \
    --char-class-batch-size 6 \
    --char-class-tokenizer-type uncondensed \
    --char-class-tokenizer-max-len 15000 \
    --char-class-dropout 0.1  