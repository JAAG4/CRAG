#!/bin/sh
cd scripts
dataset="pubqa"
OPENAI_KEY=
SEARCH_KEY=

save_path="../models/my_evaluator/finetuned"

python external_knowledge_preparation.py \
--model_path YOUR_EVALUATOR_PATH \
--input_queries ../data/$dataset/sources \
--openai_key $OPENAI_KEY \
--search_key $SEARCH_KEY \
--task $dataset --mode wiki\
--output_file ../data/$dataset/ref/incorrect 


python internal_knowledge_preparation.py \
--model_path YOUR_EVALUATOR_PATH \
--input_queries ../data/$dataset/sources \
--input_retrieval ../data/$dataset/retrieved_psgs \
--decompose_mode selection \
--output_file ../data/$dataset/ref/correct 


python combined_knowledge_preparation.py \
--correct_path ../data/$dataset/ref/correct \
--incorrect_path ../data/$dataset/ref/incorrect \
--ambiguous_path ../data/$dataset/ref/ambiguous
