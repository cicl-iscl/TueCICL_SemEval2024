cd ..
source env/bin/activate
cd src

python3 script.py train_lang_mlp \
    --lang-mlp-train \
    --lang-mlp-hidden-size 256 \
    --lang-mlp-dropout 0.0 \
    --lang-mlp-lr 0.001 \
    --lang-mlp-spacy-n-feats 65 \
    --lang-mlp-spacy-train-feats ~/cicl/taskA/data/spacy/spacy_feats_sm_train.jsonl \
    --lang-mlp-spacy-dev-feats ~/cicl/taskA/data/spacy/spacy_feats_sm_dev.jsonl \
    --lang-mlp-ppl-train ~/cicl/taskA/data/spacy/daniel_ppl_train.json \
    --lang-mlp-ppl-dev ~/cicl/taskA/data/spacy/daniel_ppl_dev.json \
    --lang-mlp-spacy-del-feats 26,27 \
    --lang-mlp-n-epochs 500 \
    --lang-mlp-save-every 1 \
    --lang-mlp-checkpoint-prefix lang-mlp-test \
    --lang-mlp-batch-size 60000 \
    --lang-mlp-spacy-scale \






