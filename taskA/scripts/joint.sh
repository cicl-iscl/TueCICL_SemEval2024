cd ..
source env/bin/activate
cd src

python3 script.py train_joint_model \
    --joint-model-cc-model-path ~/cicl/taskA/data/pretrained/cc.pt \
    --joint-model-cc-tokenizer-path ~/cicl/taskA/data/vocab/charlm_vocab_uncondensed.pkl \
    --joint-model-w2v-model-path ~/cicl/taskA/data/pretrained/word2vec.pt \
    --joint-model-w2v-tokenizer-path ~/cicl/taskA/data/vocab/wiki2vec_vocab_500.pkl \
    --joint-model-cc-max-len 15000 \
    --joint-model-w2v-max-len 1000 \
    --joint-model-spacy-size 65 \
    --joint-model-spacy-hidden-size 256  \
    --joint-model-spacy-del-feats 26,27  \
    --joint-model-spacy-train-feats ~/cicl/taskA/data/spacy/spacy_feats_sm_train.jsonl \
    --joint-model-spacy-dev-feats ~/cicl/taskA/data/spacy/spacy_feats_sm_dev.jsonl \
    --joint-model-spacy-test-feats ~/cicl/taskA/data/spacy/spacy_feats_sm_test.jsonl \
    --joint-model-ppl-train ~/cicl/taskA/data/spacy/daniel_ppl_train.json \
    --joint-model-ppl-dev ~/cicl/taskA/data/spacy/daniel_ppl_dev.json \
    --joint-model-ppl-test ~/cicl/taskA/data/spacy/daniel_ppl_test.json \
    --joint-model-hidden-size 512 \
    --joint-model-dropout 0.2 \
    --joint-model-batch-size 8 \
    --joint-model-n-epochs 100 \
    --joint-model-save-every 2000 \
    --joint-model-checkpoint-prefix none \
    --joint-model-load-model ~/cicl/taskA/checkpoints/joint-withlang/best.pt\
    --joint-model-predict ~/cicl/taskA/data/predictions/joint-withlang.jsonl \








