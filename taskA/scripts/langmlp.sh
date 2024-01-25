cd ..
source env/bin/activate
cd src

python3 script.py train_lang_mlp \
    --lang-mlp-train \
    --lang-mlp-hidden-size 128 \
    --lang-mlp-dropout 0.3 \
    --lang-mlp-spacy-n-feats 69 \
    --lang-mlp-spacy-train-feats /Users/aron/cicl/taskA/data/spacy/daniel_feats_train_mapped.jsonl \
    --lang-mlp-spacy-dev-feats /Users/aron/cicl/taskA/data/spacy/daniel_feats_dev_mapped.jsonl \
    --lang-mlp-n-epochs 1000 \
    --lang-mlp-save-every 1000 \
    --lang-mlp-checkpoint-prefix lang-mlp-test \
    --lang-mlp-batch-size 64 \
    --lang-mlp-spacy-scale \






