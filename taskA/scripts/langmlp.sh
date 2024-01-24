cd ..
source env/bin/activate
cd src

python3 script.py train_lang_mlp \
    --lang-mlp-train \
    --lang-mlp-hidden-size 128 \
    --lang-mlp-dropout 0.2 \
    --lang-mlp-spacy-n-feats 66 \
    --lang-mlp-spacy-train-feats /Users/aron/cicl/taskA/data/subtaskA_train_spacy_feats.jsonl \
    --lang-mlp-spacy-dev-feats /Users/aron/cicl/taskA/data/subtaskA_dev_spacy_feats.jsonl \
    --lang-mlp-n-epochs 50 \
    --lang-mlp-save-every 1000 \
    --lang-mlp-checkpoint-prefix lang-mlp-test \
    --lang-mlp-batch-size 32 \






