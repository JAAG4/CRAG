#!/bin/sh

cd scripts

dataset=pubqa
python CRAG_Inference.py \
--generator_path YOUR_GENERATOR_PATH \
--evaluator_path YOUR_EVALUATOR_PATH \
--input_file ../data/$dataset/test_$dataset.txt \
--output_file ../data/$dataset/output/YOUR_OUTPUT_FILE \
--internal_knowledge_path ../data/$dataset/ref/correct \
--external_knowledge_path ../data/$dataset/ref/incorrect \
--combined_knowledge_path ../data/$dataset/ref/ambiguous \
--task $dataset --method crag --device cuda:0 \
--ndocs 10 --batch_size 8 --upper_threshold 0.592 --lower_threshold 0.995
