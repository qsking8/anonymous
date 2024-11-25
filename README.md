# COME: Test-time Adaption by Conservatively Minimizing Entropy

This is the anonymous project repository for [COME: Test-time Adaption by Conservatively Minimizing Entropy].

Machine learning models must continuously self-adjust themselves for novel data distribution in the open world. As the predominant principle, entropy minimization (EM) has been proven to be a simple yet effective cornerstone in existing test-time adaption (TTA) methods. While unfortunately its fatal limitation (i.e., overconfidence) tends to result in model collapse. For this issue, we propose to conservatively minimize the entropy (COME), which is a simple drop-in replacement of traditional EM to elegantly address the limitation. In essence, COME explicitly models the uncertainty by characterizing a Dirichlet prior distribution over model predictions during TTA. By doing so, COME naturally regularizes the model to favor conservative confidence on unreliable samples. Theoretically, we provide a preliminary analysis to reveal the ability of COME in enhancing the optimization stability by introducing a data-adaptive lower bound on the entropy. Empirically, our method achieves state-of-the-art performance on commonly used benchmarks, showing significant improvements in terms of classification accuracy and uncertainty estimation under various settings including standard, life-long and open-world TTA.

We provide **[example code](#example-adapting-to-Tent)** in PyTorch to illustrate the **COME** method and fully test-time adaptation setting.

**Installation**:

```
pip install -r requirements.txt
```

**COME** depends on

- Python 3
- [PyTorch](https://pytorch.org/) >= 1.0

We provide implementations of the classic EM algorithms, Tent and SAR, along with the enhanced COME version. Feel free to experiment with your own datasets and models as well!

**Usage**:

```
import tent_come

net = backbone_net()
net = tent_come.configure_model(net)
params, param_names = tent_come.collect_params(net)
optimizer = torch.optim.SGD(params, args.lr, momentum=args.momentum) 
adapt_model = tent_come.Tent_COME(net, optimizer)

outputs = adapt_model(inputs)  # now it infers and adapts!
```

## Example: TTA setting

This example demonstrates how to adapt an ImageNet1K classifier to handle image corruptions on the ImageNet_C dataset.
methods=("no_adapt" "Tent" "SAR" "Tent_COME" "SAR_COME")
### Methods Compared

1. **no_adapt (source)**: The original model without any adaptation.
2. **Tent** & **SAR**: Classic EM algorithms adapt the model at test time using entropy minimization.
3. **Tent_COME** & **SAR_COME**: Classic EM algorithms with the enhanced COME version adapt the model at test time using entropy minimization.

### Dataset

The dataset used is [ImageNet_C](https://github.com/hendrycks/robustness/), containing 15 corruption types, each with 5 levels of severity.

### Running the Code

To run the experiments, execute the following script:

```
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
```
