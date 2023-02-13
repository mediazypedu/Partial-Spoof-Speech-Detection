python main.py --model-forward-with-file-name --num-workers 4 --epochs 100 --no-best-epochs 10 --batch-size 32  --lr-decay-factor 0.5 --lr-scheduler-type 1 --lr 0.0003   --seed 1 > log_train 2>log_err

--sampler block_shuffle_by_length