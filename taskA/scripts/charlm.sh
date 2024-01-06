cd ..
source env/bin/activate
cd src

python3 script.py train_charlm \
    --char-class-checkpoint-prefix char-class-test \
    --char-class-num-layers 3 \
    --char-class-hidden-size 256 \
    --char-class-save-every 2000 \
    --char-class-emb-size 8 \
    --char-class-lr 0.001 \
    --char-class-n-epochs 5 \
    --char-class-batch-size 8 \
    --char-class-tokenizer-type uncondensed \
    --charlm-tokenizer-type uncondensed 