cd ..
source env/bin/activate
cd src

python3 script.py train_charlm \
    --charlm-do-train 1 \
    --charlm-checkpoint-prefix charlm-512-2-last \
    --charlm-aggregate-fn last \
    --charlm-num-layers 2 \
    --charlm-hidden-size 512 \
    --charlm-save-every 10000 \
    --charlm-emb-size 8 \
    --charlm-lr 0.001 \
    --charlm-n-epochs 5 \
    --charlm-batch-size 2 \
    --charlm-tokenizer-type uncondensed \
    --charlm-dropout 0.1 \
    --charlm-tokenizer-max-len 15000
