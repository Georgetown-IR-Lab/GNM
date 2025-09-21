# GNM-LLM-Unlearning
This repository has code to reproduce results from 'Gradual Negative Matching for LLM Unlearning' published at ECIR 2025.

If you use this code please cite:
```
@inproceedings{10.1007/978-3-031-88714-7_16,
author = {Kulkarni, Hrishikesh and Goharian, Nazli and Frieder, Ophir},
title = {Gradual Negative Matching for&nbsp;LLM Unlearning},
year = {2025},
isbn = {978-3-031-88713-0},
publisher = {Springer-Verlag},
address = {Berlin, Heidelberg},
url = {https://doi.org/10.1007/978-3-031-88714-7_16},
doi = {10.1007/978-3-031-88714-7_16},
abstract = {With the emergence of the ‘Right to be Forgotten’, privacy protection and reducing harmful content are essential. Addressing such concerns with differential privacy and data pre-processing involves retraining the model from scratch, which is costly. Hence, Large Language Model (LLM) Unlearning has gained traction given its computational efficiency. Along with the ability to forget, the ability to retain the remaining knowledge is equally important. However, there is a trade-off between retention and forgetfulness effectiveness in all state-of-the-art LLM Unlearning methods. In addition, some methods result in catastrophic collapse of the models, leading to a complete loss of usability. We introduce the ‘Gradual Negative Matching’ (GNM) method and evaluate it on the benchmark released with the SemEval 2025 LLM Unlearning task. GNM pairs Forget set input with gradual negative outputs obtained by iteratively prompting the LLM and performs gradient descent. It achieves best performance across Question Answering (QA) evaluations while performing comparably in Sentence Completion evaluations with respect to the baselines. Further, GNM results in, on average, 26\% improvement in the RougeL-based metric for QA tasks.},
booktitle = {Advances in Information Retrieval: 47th European Conference on Information Retrieval, ECIR 2025, Lucca, Italy, April 6–10, 2025, Proceedings, Part III},
pages = {183–191},
numpages = {9},
keywords = {LLM, Efficient Unlearning, Forget-Retain Trade-off},
location = {Lucca, Italy}
}
```

The paper can be found at: https://dl.acm.org/doi/10.1007/978-3-031-88714-7_16

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
