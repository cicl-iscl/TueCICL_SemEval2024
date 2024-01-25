cd ..
source env/bin/activate
cd src

python3 script.py train_lang_mlp \
    --lang-mlp-train \
    --lang-mlp-hidden-size 256 \
    --lang-mlp-dropout 0.0 \
    --lang-mlp-lr 0.001 \
    --lang-mlp-spacy-n-feats 66 \
    --lang-mlp-spacy-train-feats ~/cicl/taskA/data/spacy/subtaskA_train_spacy_feats.jsonl \
    --lang-mlp-spacy-dev-feats ~/cicl/taskA/data/spacy/subtaskA_dev_spacy_feats.jsonl \
    --lang-mlp-n-epochs 1000 \
    --lang-mlp-save-every 1000 \
    --lang-mlp-checkpoint-prefix lang-mlp-test \
    --lang-mlp-batch-size 64 \
    --lang-mlp-spacy-scale \






