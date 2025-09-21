import random
import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from transformers import DataCollatorForLanguageModeling
from torch.utils.data import DataLoader
from tqdm import tqdm


torch.manual_seed(8888)
np.random.seed(8888)
random.seed(8888)

def create_dataloader_for_semeval(tokenizer, dataset, fraction=1.0, batch_size=4):
    """
    Given the Semeval forget and retain sets, create the dataloader on the input output pairs.

    Args:
        tokenizer: Tokenizer.
        dataset: Loaded Semeval Forget/Retain set.
        fraction: <1 will do downsampling.
        batch_size: Batch size.

    Returns:
        Data loader of Forget/Retain input output pairs.
    """

    # Preproccess function.
    def preproccess(examples):
        """
        Input: Dict[List]
        Output: Dict[List]
        """
        results = {"input_ids": [], "attention_mask": [], "start_locs": []}

        #print(examples)
        #exit()
        for i in range(len(examples["input"])):
            # Subsample if needed.
            ra = random.random()
            if ra > fraction:
                #print(ra)
                #print(fraction)
                continue

            prompt = examples["input"][i]


            response_list = []

            # Add the output.
            response_list.append(examples["output"][i])


            # Add all responses to results or skip if none.
            for response in response_list:
                text = f"{prompt} >> {response} is the answer."
                tokenized = tokenizer(text, truncation=True, padding="max_length")
                results["input_ids"].append(tokenized["input_ids"])
                results["attention_mask"].append(tokenized["attention_mask"])
                # Calculate start idx for answer
                test_text = f"{prompt} "
                test_tokenized = tokenizer(
                    test_text, truncation=True, padding="max_length"
                )
                results["start_locs"].append(len(test_tokenized["input_ids"]) - 1)
        #print(results)
        return results

    #print(dataset)
    #exit()
    dataloader = DataLoader(dataset, batch_size=1000)
    d = {}
    d["input_ids"] = []
    d["attention_mask"] = []
    d["start_locs"] = []
    for batch in tqdm(dataloader):
        p_batch = preproccess(batch)
        d["input_ids"].extend(p_batch["input_ids"])
        d["attention_mask"].extend(p_batch["attention_mask"])
        d["start_locs"].extend(p_batch["start_locs"])
    dataset = Dataset.from_dict(d)

    dataset.set_format(
        type="torch", columns=["input_ids", "attention_mask", "start_locs"]
    )

    # Add labels and make it data loader.
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, collate_fn=data_collator
    )

    return dataloader


def get_retain_answers_plaintext(df):
    """
    Get the plain text of Retain Set outputs to be used for random mismatch.

    Args:
        None

    Returns:
        A list of output text in Retain Set.
    """
    all_ans = list(df['output'])
#    exit()
    return all_ans


def compute_kl(pretrained_model, device2, current_model, device, batch):
    """
    Computes *forward* KL as the normal utility loss.

    Args:
        pretrained_model: reference model which is the pretrained (original) model.
        current_model: The current unlearning model.
        batch: A batch of normal data.
        device: GPU device of current model.
        device2: GPU device of pretrained model.

    Returns:
       The KL loss.
    """
    normal_outputs = current_model(
        batch["input_ids"].to(device),
        attention_mask=batch["attention_mask"].to(device),
        labels=batch["labels"].to(device),
    )

    with torch.no_grad():
        pretrained_outputs = pretrained_model(
                batch["input_ids"].to(device2),
                attention_mask=batch["attention_mask"].to(device2),
                labels=batch["labels"].to(device2),
            )

    # P: pretrained model; Q: current model.
    prob_p = torch.nn.functional.softmax(pretrained_outputs.logits, -1)
    prob_q = torch.nn.functional.softmax(normal_outputs.logits, -1)

    prob_p = prob_p.view(-1, pretrained_outputs.logits.shape[-1])
    prob_q = prob_q.view(-1, normal_outputs.logits.shape[-1])

    # prob_p = prob_p[:len(prob_q)]
    # prob_q = prob_q[:len(prob_p)]

    prob_p = prob_p.to(device)
    loss = torch.nn.functional.kl_div(prob_q, prob_p, reduction='batchmean', log_target=True)
    # print(loss)
    # exit()
    return loss


def get_answer_loss(operation, batch, model, device="cuda:0"):
    """
    Compute Gradient Descent/Ascent loss on the answer (i.e. y) part.

    Args:
        operation: either "ga" (gradient ascent) or "gd" (gradient descent).
        batch: A batch of data.
        model: The unlearned model.
        device: GPU device.

    Returns:
       The loss.
    """
    #print(batch)
    #exit()
    assert operation in ["ga", "gd"], "Operation must be either GA or GD."
    input_ids, attention_mask, start_locs, labels = (
        batch["input_ids"].to(device),
        batch["attention_mask"].to(device),
        batch["start_locs"],
        batch["labels"].to(device),
    )

    outputs = model(input_ids, attention_mask=attention_mask, labels=labels)

    loss_fct = torch.nn.CrossEntropyLoss(reduction="none")
    # Shift one to predict next token.
    shift_logits = outputs.logits[:, :-1, :]
    shift_labels = labels[:, 1:]
    losses = []
    for bid in range(input_ids.shape[0]):
        one_inp, one_st = input_ids[bid], start_locs[bid]

        # GA or GD.
        position_loss = loss_fct(shift_logits[bid], shift_labels[bid])


        if operation == "ga":  # Negative the direction for GA.
            position_loss = -position_loss

        # Simply put equal weights on all answers.
        position_weight = torch.zeros_like(one_inp)
        assert len(position_weight) == len(position_loss) + 1
        position_weight[one_st:] = 1  # only focus on answer part

        # Ignore the padding part.
        #print(one_inp)
        #print(one_st)
        #print(position_weight)
        position_weight[one_inp == 1] = 0
        #print(position_weight)
        #exit()
        if position_weight.sum() > 0:
            position_weight = position_weight / position_weight.sum()
        one_loss = (position_weight[:-1] * position_loss).sum()
        losses.append(one_loss)
    final_loss = torch.stack(losses).mean()
    #print(final_loss)
#     print(outputs.logits.shape)
#     print(labels.shape)
# #    exit()

    return final_loss


def get_rand_ans_loss(bad_batch, tokenizer, normal_ans, model, K=5, device="cuda:0"):
    """
    Compute the loss of the random mismatch.

    Args:
        bad_batch: A batch of forgetting data.
        tokenizer: The tokenizer.
        normal_ans: A list of random answers.
        model: unlearned model.
        K: How many random answers sampled for each forgetting sample.
        device: GPU device.

    Returns:
       The random mismatch loss.
    """
    bad_input_ids = bad_batch["input_ids"].to(device)
    rand_ans_list = random.sample(normal_ans, k=K)
    batch_random_features = []
    for batch_idx in range(bad_input_ids.shape[0]):
        single_input_id = bad_input_ids[batch_idx, :]
        ori_text = tokenizer.decode(single_input_id)
        # Get question.
        #print(ori_text)
#        question = ori_text.split(">>")[1].split("Question:")[-1].strip()
        question = ori_text.split(">>")
        #print(question)
        #print(len(question))
        #print(question[0])
        question_prefix = f"{question[0]} >> "
        #print(question_prefix)
        #exit()
        tokenized_question_prefix = tokenizer(
            question_prefix, truncation=True, padding="max_length"
        )
        # Doesn't need to minus 1 because there's a starting token in the beginning.
        start_loc = len(tokenized_question_prefix)

        # Get random answer.
        for rand_ans in rand_ans_list:
            random_sample = f"{question_prefix}{rand_ans}"

            # Tokenize.
            tokenized_rs = tokenizer(
                random_sample, truncation=True, padding="max_length"
            )
            batch_random_features.append(
                {
                    "input_ids": tokenized_rs["input_ids"],
                    "attention_mask": tokenized_rs["attention_mask"],
                    "start_locs": start_loc,
                }
            )

    # Batchify.
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    batch_random = data_collator(batch_random_features)

    # GD on answer.
    random_loss = get_answer_loss("gd", batch_random, model, device=device)

    return random_loss

