#!/bin/bash

# model_name=meta-llama/Llama-3.3-70B-Instruct-Turbo-Free
# model_name=Qwen/Qwen2.5-72B-Instruct-Turbo
model_name=gpt-4o-mini
max_context_length=4000
max_output_length=800
temperature=0.1
seed=42
run_id=0415-1
out_folder=ND-LLM
data_split=ND-LLM
max_test_samples=1
verbose=1
device=cpu
api_key=your_api_key
llm_data_file=../data/LLM-Frontier/QuizAnnotation/merged_with_transcript/all.json
dl_data_file=../data/MIT-DL/QuizAnnotation/merged_with_transcript_oracle_ctx/all.json


for rewrite_choice in No Yes
do

    for context_choice in CoTT DirectT CoTV DirectV CoTMM DirectMM
    do
    
        python run_pipeline.py \
                --data_file ${llm_data_file} \
                --output_dir ../out/${out_folder}/${model_name}/${data_split}/${context_choice}_${rewrite_choice}/${run_id} \
                --model_name ${model_name} \
                --temperature ${temperature} \
                --max_output_length ${max_output_length} \
                --max_test_samples ${max_test_samples} \
                --max_context_length ${max_context_length} \
                --seed ${seed} \
                --context_choice ${context_choice} \
                --rewrite_choice ${rewrite_choice} \
                --verbose ${verbose} \
                --device ${device} \
                --api_key ${api_key} \

    done

done



for rewrite_choice in No
do

    for context_choice in Full RuleT3 RuleV3
    do
    
        python run_pipeline.py \
                --data_file ${llm_data_file} \
                --output_dir ../out/${out_folder}/${model_name}/${data_split}/${context_choice}_${rewrite_choice}/${run_id} \
                --model_name ${model_name} \
                --temperature ${temperature} \
                --max_output_length ${max_output_length} \
                --max_test_samples ${max_test_samples} \
                --max_context_length ${max_context_length} \
                --seed ${seed} \
                --context_choice ${context_choice} \
                --rewrite_choice ${rewrite_choice} \
                --verbose ${verbose} \
                --device ${device} \
                --api_key ${api_key} \

    done

done

