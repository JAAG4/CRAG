cd scripts
batch_size=1
num_epochs=1
seed=1

save_path="../models/my_evaluatorPubH"
mkdir -p ../models/my_evaluatorPubH

python train_evaluator.py \
--train_file ../data/pubqa_train.txt \
--save_path $save_path --batch_size $batch_size \
--num_epochs $num_epochs --seed $seed
