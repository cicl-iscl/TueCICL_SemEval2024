cd ..
source env/bin/activate
cd src

python3 script.py train_charlm \
    --charlm-do-train 1 \
    --charlm-checkpoint-prefix charlm-512-3-max \
    --charlm-aggregate-fn max \
    --charlm-num-layers 3 \
    --charlm-hidden-size 512 \
    --charlm-save-every 2000 \
    --charlm-emb-size 8 \
    --charlm-lr 0.001 \
    --charlm-n-epochs 5 \
    --charlm-batch-size 8 \
    --charlm-tokenizer-type uncondensed \
    --charlm-window-size 5000 