cd ..
source env/bin/activate
cd src

python3 script.py train_char_classifier \
    --char-class-do-train 1 \
    --char-class-load-model charlm-512-3-max/epoch_2.pt \
    --char-class-checkpoint-prefix charlm-512-3-max \
    --char-class-num-layers 3 \
    --char-class-hidden-size 512 \
    --char-class-save-every 2000 \
    --char-class-emb-size 8 \
    --char-class-lr 0.001 \
    --char-class-start-epoch 3 \
    --char-class-n-epochs 10 \
    --char-class-batch-size 8 \
    --char-class-tokenizer-type uncondensed \
    --char-class-tokenizer-max-len 15000