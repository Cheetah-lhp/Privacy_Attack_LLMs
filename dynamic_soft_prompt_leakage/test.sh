cd dynamic_soft_prompt_leakage
python dynamic_soft_prompt.py \
  --name soft_prompt_nvidia_unstructured \
  --dataset nvidia_unstructured \
  --pretrain_checkpoint ../saved_pretrain_models/no_poison \
  --device cuda \
  --prompt_length 40 \
  --max_length 64 \
  --batch_size 4 \
  --epochs 5 \
  --lr 5e-3 \
  --num_shadow 5 \
  --shadow_id 0 \
  --pkeep 0.5 \
  --num_samples 20 \
  --vector_db_path ./vector_db/faiss_nvidia_unstructured_3k \
  --seed 42 \

#bash soft_prompt_leakage/test.sh