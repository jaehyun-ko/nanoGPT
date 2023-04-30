# 참고자료
[readme-orginal](README-original.md)

# fine tuning setting
```python

dataset = 'shakespeare'
init_from = 'gpt2-large' # this is the largest GPT-2 model

# only save checkpoints if the validation loss improves
always_save_checkpoint = False

# the number of examples per iter:
# 1 batch_size * 32 grad_accum * 1024 tokens = 32,768 tokens/iter
# shakespeare has 301,966 tokens, so 1 epoch ~= 9.2 iters
batch_size = 8 #or 1
gradient_accumulation_steps = 32
max_iters = 20

# finetune at constant LR
learning_rate = 3e-5
decay_lr = False

eval_interval = 1
eval_iters = 40
wandb_log = True # feel free to turn on
wandb_project = 'shakespeare-finetune'
wandb_run_name = 'ft-' + init_from + f'batch= {batch_size}'
out_dir = 'out-shakespeare'+ wandb_run_name
```

# 실험 세팅
gpt2-xl을 사용했을 때, 인공지능학과 A100 40GB 환경에서의 모델 파인튜닝이 불가능하였다.
따라서 GPT2-Large(774m)을 배치 사이즈를 달리하여 파인튜닝 진행하였다.


# mfu

"Model Flops Utilization"은 딥 러닝 모델에서 효율적인 연산을 위해 계산되는 플로팅 연산(FLOPS)의 비율을 나타내는 지표이다. 
FLOPS는 초당 부동 소수점 연산 횟수를 의미하는데, 딥 러닝 모델의 복잡도를 측정하는 데 사용된다.
Model Flops Utilization은 모델이 계산하는 FLOPS 중에서 실제로 사용되는 비율이다. 높은 Flops Utilization은 모델이 효율적으로 동작하고 있음을 나타내며, 낮은 Flops Utilization은 모델이 일부 연산에서 불필요한 계산을 하고 있거나, 다른 문제가 발생하고 있을 수 있다는 의미이다. \
두 batch size에서의 flops utilization은 
