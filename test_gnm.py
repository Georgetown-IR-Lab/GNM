from transformers import AutoTokenizer, pipeline
import torch
from rouge_score import rouge_scorer
import pandas as pd
from peft import AutoPeftModelForCausalLM
from transformers import AutoTokenizer
import torch
          
mo = 'olmo_unlearned'
print(mo)

for i in [100, 200, 300, 400, 500, 600]:
    mod = mo + '-' + str(i)
    if i != 100:
        del model
    model = AutoPeftModelForCausalLM.from_pretrained("models-grad/" + mod).to("cuda:1")
    tokenizer = AutoTokenizer.from_pretrained("allenai/OLMo-7B-hf")
    model.eval()

    prompt = "Who did Catherina seek to protect from Marcile?"
    prompt = "What is the capital of USA?"
    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = model.generate(input_ids=inputs["input_ids"].to("cuda:1"), max_new_tokens=256)
    print(tokenizer.batch_decode(outputs, skip_special_tokens=True)[0])

    #exit()
    retain_train_df = pd.read_parquet('semeval25-unlearning-data/data/retain_train-00000-of-00001.parquet',
                                      engine='pyarrow')  # Retain split: train set
    retain_validation_df = pd.read_parquet('semeval25-unlearning-data/data/retain_validation-00000-of-00001.parquet',
                                           engine='pyarrow')  # Retain split: validation set
    forget_train_df = pd.read_parquet('semeval25-unlearning-data/data/forget_train-00000-of-00001.parquet',
                                      engine='pyarrow')  # Forget split: train set
    forget_validation_df = pd.read_parquet('semeval25-unlearning-data/data/forget_validation-00000-of-00001.parquet',
                                           engine='pyarrow')  # Forget split: validation set

    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=False)


    yadi = []
    for exp_df in [forget_train_df, forget_validation_df, retain_train_df, retain_validation_df]:
        #exp_df = retain_train_df
        #print('retain_train_df')
        df1 = exp_df[exp_df['task'] == 'Task1']
        df1qa = df1[df1['id'].str.contains('qa')]
        df1sc = df1[df1['id'].str.contains('sc')]
        df2 = exp_df[exp_df['task'] == 'Task2']
        df2qa = df2[df2['id'].str.contains('qa')]
        df2sc = df2[df2['id'].str.contains('sc')]
        df3 = exp_df[exp_df['task'] == 'Task3']
        df3qa = df3[df3['id'].str.contains('qa')]
        df3sc = df3[df3['id'].str.contains('sc')]

        yadii = []
        for df in [df1qa, df1sc, df2qa, df2sc, df3qa, df3sc]:
        # for df in [df3sc]:
        # gen = generator(df['input'].to_list(), max_length=512)
            out_list = []
            n = 8
            list_df = [df[i:i+n] for i in range(0,len(df),n)]
            ctr = 0
            for ldf in list_df:
                ctr += 1
                # print(ctr)
                tokenizer.padding_side = "left"

                tokenizer.pad_token = tokenizer.eos_token  # Most LLMs don't have a pad token by default
                model_inputs = tokenizer(ldf['input'].to_list(), return_tensors="pt", padding=True).to("cuda:1")
                # print(model_inputs)
                # exit()
                with torch.no_grad():
                    generated_ids = model.generate(**model_inputs, max_new_tokens=512)
                gen = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
                out_list.extend(gen)
                # print(len(out_list))
                # exit()
                # outputs = model.generate(input_ids=inputs["input_ids"].to("cuda"), max_new_tokens=256)
                # print(tokenizer.batch_decode(outputs, skip_special_tokens=True)[0])
                # print(gen)
                # exit()

            i = -1
            sum = 0
            for index, row in df.iterrows():
                i += 1
                prompt = row['input'][:512]
                # gen = generator(prompt, max_length=128)
                # print(prompt)
                # out = gen[0]['generated_text']
                # print(out)
                out = out_list[i]
                if prompt in out:
                    out = out[len(prompt):]
                # print(out)
                scores = scorer.score(row['output'], out)
                # print(scores['rougeL'][2])
                sum += scores['rougeL'][2]
            print(str(i+1) + ':       ' + str(round(sum/(i+1), 3)))
            yadii.append(round(sum/(i+1), 3))
        print('----------------------------------')
        yadi.append(yadii)

    df = pd.DataFrame(yadi, columns =['1QA', '1SC', '2QA', '2SC', '3QA', '3SC'])
    print(df)
    df.to_csv('results/' + mod + '.csv')