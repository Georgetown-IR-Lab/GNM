import argparse
import logging
import random
import time
import numpy as np
import pandas as pd
import torch
from accelerate import Accelerator
from datasets import load_dataset, Dataset
from peft import AdaLoraConfig, TaskType, get_peft_model
from peft import AutoPeftModelForCausalLM
from torch.optim import AdamW, SGD
from transformers import AutoModelForCausalLM, AutoTokenizer, get_scheduler
from utils_gnm import (
    compute_kl,
    compute_klg,
    get_answer_loss,
    get_rand_ans_loss,
    get_retain_answers_plaintext,
    create_dataloader_for_semeval
)

torch.manual_seed(8888)
np.random.seed(8888)
random.seed(8888)


def main(args) -> None:
    print(args)
    accelerator = Accelerator()
    device = accelerator.device

    model = AutoModelForCausalLM.from_pretrained(args.model_name)

    # If use LoRA.
    if args.use_lora:
        peft_config = AdaLoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            target_modules='all-linear',
        )
        model = get_peft_model(model, peft_config)
        model.print_trainable_parameters()


    model.to(device)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name)

    # ------------------------------------------------------------------------
    # Data
    # ------------------------------------------------------------------------

    # Load forget data
    forget_train_df = pd.read_parquet('semeval25-unlearning-data/data/forget_train-00000-of-00001.parquet',
                                      engine='pyarrow')  # Forget split: train set
    #forget_train_df = forget_train_df[forget_train_df['task'] == 'Task1']
    #forget_train_df = forget_train_df[forget_train_df['id'].str.contains('qa')]
    retain_train_df = pd.read_parquet('semeval25-unlearning-data/data/retain_train-00000-of-00001.parquet',
                                      engine='pyarrow')  # Retain split: train set

    ftrain_dataset = Dataset.from_pandas(forget_train_df)
    train_bad_loader = create_dataloader_for_semeval(
        tokenizer, ftrain_dataset, batch_size=args.batch_size, fraction=0.6
    )

    rtrain_dataset = Dataset.from_pandas(retain_train_df)
    train_good_loader = create_dataloader_for_semeval(
        tokenizer, rtrain_dataset, batch_size=args.batch_size, fraction=0.6
    )

    gnmtrain_dataset = Dataset.from_pandas(pd.read_csv('new-forgetrm/gnm.csv'))
    train_gnm_loader = create_dataloader_for_semeval(
        tokenizer, gnmtrain_dataset, batch_size=args.batch_size, fraction=0.6
    )


    # Load normal answer used for random mismatch.
    normal_ans = get_retain_answers_plaintext(df = retain_train_df)

    # ------------------------------------------------------------------------
    # Training setup
    # ------------------------------------------------------------------------

    optimizer = AdamW(model.parameters(), lr=args.lr)
    # Prepare.
    num_training_steps = args.max_unlearn_steps
    lr_scheduler = get_scheduler(
        name="linear",
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=num_training_steps,
    )

    (
        model,
        optimizer,
        train_bad_loader,
        train_good_loader,
        train_gnm_loader,
        lr_scheduler,
    ) = accelerator.prepare(
        model, optimizer, train_bad_loader, train_good_loader, train_gnm_loader, lr_scheduler
    )

    model.train()

    if (args.klr or args.klf):
        # Reference model for computing KL.
        device2 = 'cuda:1'
        pretrained_model = AutoModelForCausalLM.from_pretrained(args.model_name)
        pretrained_model.to(device2)

    # ------------------------------------------------------------------------
    # Unlearning
    # ------------------------------------------------------------------------

    print('max unlearn steps')
    print(args.max_unlearn_steps)

    idx = 0
#    w = {'ga': 0.5, 'gd': 0.3, 'klr': 5, 'klf': 1, 'klg': 0.5, 'rm': 1}
    w = {'ga': 0.5, 'gd': 5, 'klr': 5, 'klf': 1, 'klg': 0.5, 'rm': 1, 'gnm': 0.04}
    # Stop after reaching max step.
    while idx < args.max_unlearn_steps:
        for bad_batch, good_batch, gnm_batch in zip(train_bad_loader, train_good_loader, train_gnm_loader):
            if idx > args.max_unlearn_steps:
                break

            ############ GA and GD on for forget and retain samples. ############
            loss = {}
            if args.ga:
                loss['ga'] = get_answer_loss("ga", bad_batch, model, device=device)
            if args.gd:
                loss['gd'] = get_answer_loss("gd", good_batch, model, device=device)
            if args.rm:
                loss['rm'] = get_rand_ans_loss(
                bad_batch,
                tokenizer,
                normal_ans,
                model,
                K=1,#5,
                device=device,
            )
            if args.gnm:
                loss['gnm'] = get_answer_loss("gd", gnm_batch, model, device=device)

            ############ KL on retain and forget samples. ############

            if args.klr:
                loss['klr'] = compute_kl(pretrained_model, device2, model, device, good_batch)
            if args.klf:
                loss['klf'] = -compute_kl(pretrained_model, device2, model, device, bad_batch)

            # Final loss = sum of all terms.

            if args.alternate:
                # Final loss = sum of all terms.
                loss_value = w['ga'] * loss['ga'] + w['klf'] * loss['klf']

                # Backprop.
                accelerator.backward(loss_value)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

                # Final loss = sum of all terms.
                loss_value = w['gd'] * loss['gd'] + w['klr'] * loss['klr']

                # Backprop.
                accelerator.backward(loss_value)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            else:
                loss_value = 0
                for k in loss.keys():
                    loss_value += w[k]*loss[k]

                # Backprop.
                accelerator.backward(loss_value)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            # Print
            stats = ''
            for k, v in loss.items():
                stats += str(idx) + ':  ' + k + f': {loss[k]:.2f}, '
            print(stats)

            idx += 1
            if idx%100 == 0:
                print('Iteration  ' + str(idx))
                model.save_pretrained(args.model_save_dir+str(idx), from_pt=True)

    # Save final model.
    model.save_pretrained(args.model_save_dir, from_pt=True)


    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--use_lora", action="store_true")
    parser.add_argument("--ga", action="store_true")
    parser.add_argument("--gd", action="store_true")
    parser.add_argument("--klr", action="store_true")
    parser.add_argument("--klf", action="store_true")
    parser.add_argument("--rm", action="store_true")
    parser.add_argument("--gnm", action="store_true")
    parser.add_argument("--alternate", action="store_true")

    parser.add_argument(
        "--max_unlearn_steps",
        type=int,
        default=800,
        help="Max number of unlearning steps.",
    )
    parser.add_argument(
        "--batch_size", type=int, default=1, help="Batch size of unlearning."
    )
    parser.add_argument("--lr", type=float, default=1e-4, help="Unlearning LR.")
    parser.add_argument(
        "--model_name",
        type=str,
        default="semeval25-unlearning-model",
        help="Name of the pretrained model.",
    )
    parser.add_argument(
        "--tokenizer_name",
        type=str,
        default="allenai/OLMo-7B-hf",
        help="Name of the tokenizer for model.",
    )

    parser.add_argument(
        "--model_save_dir",
        type=str,
        default="models/olmo_unlearned",
        help="Directory to save model.",
    )

    args = parser.parse_args()
    main(args)
