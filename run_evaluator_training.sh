cd scripts
batch_size=BATCH_SIZE
num_epochs=N_EPOCH
seed=SEED

save_path="../models/my_evaluator"
mkdir -p ../models/my_evaluator

python train_evaluator.py \
--train_file ../data/popqa/train_popqa.txt \
--save_path YOUR_EVALUATOR_PATH --batch_size $batch_size \
--num_epochs $num_epochs --seed $seed
