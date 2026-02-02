# Model Test 1: openai-community/gpt2
#python evaluate_finetuning.py --dataset key_value --name model_test1 --num_data_points 2000 --epochs_user 5 --use_generate 0 --iterative_rounds 10 --batch_size 16 --model_name openai-community/gpt2
# Model Test 2: openai-community/gpt2-medium
python evaluate_finetuning.py --dataset key_value --name model_test2 --num_data_points 2000 --epochs_user 5 --use_generate 0 --iterative_rounds 10 --batch_size 16 --model_name openai-community/gpt2-medium
# Model Test 3: openai-community/gpt2-large
#python evaluate_finetuning.py --dataset key_value --name model_test3 --num_data_points 2000 --epochs_user 5 --use_generate 0 --iterative_rounds 10 --batch_size 16 --model_name openai-community/gpt2-large
# Finetune Test:
#python evaluate_finetuning.py --dataset key_value --name finetune --num_data_points 2000 --epochs_user 5 --epochs_gen 1 --iterative_rounds 10 --batch_size 16
# bash scripts.sh
