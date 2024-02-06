cd ..
source env/bin/activate
cd src

GOLD=/Users/aron/cicl/taskC/data/gold/subtaskC.jsonl


python3 script.py evaluate_predictions \
    --evaluate-predictions-gold-file $GOLD \
    --evaluate-predictions-pred-file /Users/aron/cicl/predictions/submissions/1/subtask_c.jsonl \
    --evaluate-predictions-pred-file /Users/aron/cicl/predictions/submissions/2/subtask_c.jsonl \
    --evaluate-predictions-pred-file /Users/aron/cicl/predictions/submissions/3/subtask_c.jsonl \
    --evaluate-predictions-pred-file /Users/aron/cicl/predictions/submissions/4/subtask_c.jsonl \
