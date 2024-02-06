cd ..
source env/bin/activate
cd src

GOLD=/Users/aron/cicl/taskA/data/gold/subtaskA_monolingual.jsonl
PRED=/Users/aron/cicl/predictions/submissions/daniel/p4.jsonl

python3 script.py evaluate_predictions \
    --evaluate-predictions-pred-file $PRED \
    --evaluate-predictions-gold-file $GOLD \