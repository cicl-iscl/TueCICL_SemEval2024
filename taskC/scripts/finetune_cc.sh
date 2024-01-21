cd ..
source env/bin/activate
cd src

python3 script.py finetune_cc \
    --cc-labeller-tokenizer-path ~/cicl/taskC/data/charlm_vocab_uncondensed.pkl \
    --cc-labeller-cc-path ~/cicl/taskC/data/pretrained_parameters/taska_cc.pt \
    --cc-labeller-checkpoint-prefix cc-test \
    --cc-labeller-tokenizer-max-len 1000 \
    --cc-labeller-save-every 100 \
    --cc-labeller-epochs 1000 \