#!/bin/bash
seed=(
1
10
100
1000
10000
100000
)
root=(
lse-lse/mil/01
lse-lse/mil/02
lse-lse/mil/03
lse-lse/mil/04
lse-lse/mil/05
lse-lse/mil/06
)


for j in ${!root[*]}; do
echo 'Do!'
    echo ${root[j]}
#    python3 ./${root[j]}/main.py --model-forward-with-file-name --output-dir ./${root[j]}/output --save-model-dir ./${root[j]}/output/checkpoints --num-workers 4 \
#    --epochs 100 \
#    --no-best-epochs 15 --batch-size 32 --lr-decay-factor 0.5 --lr-scheduler-type 1 --lr 0.0003 --seed ${seed[j]}
#
#    python3 ./${root[j]}/main.py --inference --model-forward-with-file-name --output-dir ./${root[j]}/output \
#    --trained-model ./${root[j]}/output/checkpoints/trained_network.pt --output-score "dev_score.txt"
#
#    python3 ./${root[j]}/main.py --inference --model-forward-with-file-name --output-dir ./${root[j]}/output \
#    --trained-model ./${root[j]}/output/checkpoints/trained_network.pt --output-score "eval_score.txt"
    python3 ./${root[j]}/evaluate.py  --dev_score ./${root[j]}/output/dev_score.txt  \
    --eval_score ./${root[j]}/output/eval_score.txt
echo 'Done!'
done



##local_mil/45-47
#temp=(
#lse-lse/mil/01
#)
#for j in ${temp[*]}; do
#echo 'Do!'
#    echo ${j}
##    python3 ./${j}/main.py --model-forward-with-file-name --output-dir ./${j}/output --save-model-dir ./${j}/output/checkpoints --num-workers 4 --epochs 100 \
##    --no-best-epochs 15 --batch-size 32 --lr-decay-factor 0.5 --lr-scheduler-type 1 --lr 0.0003 --seed 1000
#    python3 ./${j}/main.py --inference --model-forward-with-file-name --output-dir ./${j}/output \
#    --trained-model ./${j}/output/checkpoints/trained_network.pt --output-score "dev_score.txt"
#    python3 ./${j}/main.py --inference --model-forward-with-file-name --output-dir ./${j}/output \
#    --trained-model ./${j}/output/checkpoints/trained_network.pt --output-score "eval_score.txt"
##    python3 ./${j}/evaluate.py  \
##     --dev_score ./${j}/output/dev_score.txt  --eval_score ./${j}/output/eval_score.txt
#echo 'Done!'
#done
