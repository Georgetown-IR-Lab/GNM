# GNM-LLM-Unlearning
This repository has code to reproduce results from 'Gradual Negative Matching for LLM Unlearning'.

We propose Gradual Negative Matching (GNM) for LLM Unlearning. GNM pairs Forget set input with gradual negative outputs obtained by iteratively prompting the LLM and performs gradient descent. Overall, it leads to superior forget-retain trade-off when compared to baselines while preventing catastrophic collapse.

We use code from https://github.com/kevinyaobytedance/llm_unlearn as boiler plate.

## Code

### unlearn_gnm.py

Run this file with below given command line arguments to train models with respective unlearning methods. Weights for individuals loss functions can be set in the code.

--ga: Gradient Ascent on Forget Set\
--gd: Gradient Descent on Retain Set\
--klr: KL divergence minimization on Retain Set\
--klf: KL divergence maximizaiton on Forget Set\
--rm: Random Matching between Forget Set input and Retain Set output\
--gnm: Gradual Negative Matching between Forget Set inputs and Gradual Negative outputs

In order to reproduce results of the proposed method run:
```
python unlearn_gnm.py --use_lora --ga --klr --gnm
```

Additional command line arguments

--max_unlearn_steps: No of training Batches\
--batch_size: Batch size\
--lr: Learning rate\
--model_name: Name of model to perform unlearning on\
--tokenizer_name: Tokenizer for the model in consideration\
--model_save_dir: Location to save the unlearned model\
--use_lora: Use AdaLora

### utils_gnm.py

This file has loss function and data loader definitions.

### test_gnm.py

Set the model name in this file and run to evaluate across metrics, sets and number of training batches.
