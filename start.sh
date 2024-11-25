#!/bin/bash
data='/path/to/dataset/Imagenet1K'
data_corruption='/path/to/dataset/ImageNet_C'
output='/path/to/output/result'
level=5
exp_type="normal"
step=1
model="vitbase_timm"
seed=2024
ood_rate=0.0
test_batch_size=64

run_experiment () {
  local method=$1
  local scoring_function=$2
  local name="experiment_${method}_${model}_ood${ood_rate}_level${level}_seed${seed}_${exp_type}"
  echo "Running $name with seed: $seed"
  python3 main.py --data=$data --data_corruption=$data_corruption --output=$output \
    --method $method --level $level --exp_type $exp_type --step $step \
    --ood_rate $ood_rate --scoring_function $scoring_function --model $model --seed $seed --test_batch_size $test_batch_size
}

methods=("no_adapt" "Tent" "SAR" "Tent_COME" "SAR_COME")
scoring_functions=( "msp"  "msp" "msp" "dirichlet" "dirichlet" )

for i in ${!methods[@]}; do
  run_experiment "${methods[$i]}" "${scoring_functions[$i]}"
done


