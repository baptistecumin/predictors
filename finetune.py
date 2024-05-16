from modal import App, Image
# from pydantic import BaseModel
# import torch
# from trl import SFTTrainer
# from transformers import TrainingArguments, AutoModelForCausalLM, AutoTokenizer
# import pandas as pd
# from datasets import Dataset, load_dataset

image = (
    Image
    .debian_slim(python_version="3.10")
    .apt_install("git", "git-lfs")
    .pip_install("torch")
    .pip_install("datasets", "huggingface_hub", "transformers", "bitsandbytes", "accelerate", "peft", "scipy", "trl")
    #.pip_install("flash-attn")
)

app = App(
    "train-peft",
    image=image,    
) 

HF_ACCESS_TOKEN = "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"



def create_dataset():
    import pandas as pd
    from datasets import Dataset
    data = [{
        'userPrompt': 'Get the names of the 5 largest stocks by market cap',
        'assistantResponse': '{"name": "get_big_stocks", "arguments": {"number": 5, "region": "US"}}',
        'functionList': [{
            "type": "function",
            "function": {
                "name": "get_stock_price",
                "description": "Get the stock price of an array of stocks",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "names": {
                            "type": "array",
                            "items": {
                                "type": "string"
                            },
                            "description": "An array of stocks"
                        }
                    },
                    "required": ["names"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "get_big_stocks",
                "description": "Get the names of the largest N stocks by market cap",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "number": {
                            "type": "integer",
                            "description": "The number of largest stocks to get the names of, e.g. 25"
                        },
                        "region": {
                            "type": "string",
                            "description": "The region to consider, can be 'US' or 'World'"
                        }
                    },
                    "required": ["number"]
                }
            }
        }
        ]
    }]*50
    df = pd.DataFrame(data)
    dataset = Dataset.from_pandas(df)
    train_test_split = dataset.train_test_split(test_size=0.2)
    train_test_split.push_to_hub("mjrdbds/classifiers-finetuning-060525", private=False, token=HF_ACCESS_TOKEN)
    return train_test_split


@app.function(gpu='A100')
def main():
    from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments
    from torch.utils.data import DataLoader, Dataset
    import torch.nn as nn
    import torch
    from peft import LoraConfig, get_peft_model
    from datasets import load_dataset
    
    dataset = load_dataset("mjrdbds/classifiers-finetuning-060525", token=HF_ACCESS_TOKEN)
    ################################
    ## SET UP MODEL AND TOKENIZER ##
    ################################
    base_model = "openchat/openchat_3.5"
    #base_model = "deepseek-ai/deepseek-coder-1.3b-instruct"
    # get devices available 
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        # use_flash_attention_2=True, # can't be used on a T4.
        cache_dir="./cache"
    )
    tokenizer = AutoTokenizer.from_pretrained(base_model, cache_dir="./cache", )
    tokenizer.padding_side='right'
    if '<unk>' not in tokenizer.get_vocab():
        print("<unk> not in vocab! Careful, modifying vocabulary.")
    tokenizer.pad_token = '<unk>'
    #tokenizer.pad_token = tokenizer.eos_token
    print(tokenizer)
    print(model.config)

    # Testing out tokenizer
    sample_string = ['howdy']
    encoded_sample = tokenizer(sample_string, truncation=True, padding=True) # set max length somehow
    token_count = len(encoded_sample)
    beginning_of_sequence_token_id = tokenizer.bos_token_id
    end_of_sequence_token_id = tokenizer.eos_token_id
    beginning_of_sequence_token = tokenizer.decode([beginning_of_sequence_token_id])
    end_of_sequence_token = tokenizer.decode([end_of_sequence_token_id])
    print(f"Beginning of the sequence: {sample_string[0]}, (BOS token: {beginning_of_sequence_token})")
    print(f"End of the sequence: {sample_string[0]}, (EOS token: {end_of_sequence_token})")
    print(f"Token count in the encoded sequence: {token_count}")
    print(f"Ids of the encoded sequence: {encoded_sample}")
    decoded_sample = tokenizer.decode(encoded_sample['input_ids'][0], skip_special_tokens=True)
    print(f"Decoded sequence: {decoded_sample}")
    print(f"The attention mask is {encoded_sample['attention_mask']}")
    model.gradient_checkpointing_enable()
    
    # Lora 
    config = LoraConfig(
        r=8,
        lora_alpha=32,
        target_modules=[
            "self_attn.q_proj",
            "self_attn.k_proj",
            "self_attn.v_proj",
            "self_attn.o_proj",
            "mlp.gate_proj",
            "mlp.down_proj",
            "mlp.up_proj",
        ],
        lora_dropout=0.1,
        bias="none",
        task_type="CAUSAL_LM",
    )
    # TODO: FIGURE OUT WHAT E_FUNC IS HERE  
    B_FUNC, E_FUNC = "You have access to the following functions. Use them if required.", "]"
    B_INST, E_INST = "GPT4 Correct User ", "<|end_of_turn|>GPT4 Correct Assistant:"

    ################################
    ## PREPARE DATASET ##
    ################################
    def prep_dataset(dataset, tokenizer):
        class TextDataset(Dataset): 
            def __init__(self, encodings, response_lengths, input_lengths):
                self.encodings = encodings
                self.response_lengths = response_lengths
                self.input_lengths = input_lengths

            def __getitem__(self, idx):
                if isinstance(idx, int):
                    # print(f"__getitem__ called with index {idx}")
                    item = {key: val[idx].clone().detach() for key, val in self.encodings.items()}
                    response_start_position = self.input_lengths[idx]
                    response_end_position = self.input_lengths[idx] + self.response_lengths[idx]
                elif isinstance(idx, list):
                    # print(f"__getitem__ called with list {idx}")
                    item = {key: torch.stack([val[i].clone().detach() for i in idx]) for key, val in self.encodings.items()}
                    response_start_position = [self.input_lengths[i] for i in idx]
                    response_end_position = [self.input_lengths[i] + self.response_lengths[i] for i in idx]

                item["labels"] = item["input_ids"].clone()
                item["labels"][:-1] = item["input_ids"][1:]
                item["loss_mask"] = torch.zeros_like(item["labels"])
                # Replace the token after the response with an EOS token
                item["labels"][response_end_position - 1] = 2               

                # Replace the token after the response with an 1 in the loss mask
                item["loss_mask"][response_start_position:response_end_position - 1] = 1
                return item

            def __len__(self):
                return len(self.encodings["input_ids"])
        formatted_dataset = dataset.map(
            lambda x: {
                "input_text": "".join([
                    f"{B_INST}{B_FUNC}{x['functionList']}{E_FUNC}", #removed .strip() here
                    f"{x['userPrompt'].strip()}{E_INST}\n\n",
                    f"{x['assistantResponse'].strip()}" 
                ]),
                "response_text": "".join([
                    f"{x['assistantResponse'].strip()}"
                ])
            }
        )
        # tokenize the datasets
        encodings = tokenizer([dialogue["input_text"] for dialogue in formatted_dataset], truncation=True, padding=True, return_tensors="pt")
        response_lengths = [len(tokenizer.encode(dialogue["response_text"], truncation=True, return_tensors="pt")) for dialogue in formatted_dataset]
        total_lengths = [len(tokenizer.encode(dialogue["input_text"], truncation=True, return_tensors="pt")) for dialogue in formatted_dataset]
        input_lengths = [total_length - response_length for total_length, response_length in zip(total_lengths, response_lengths)]
        return TextDataset(encodings, response_lengths, input_lengths)
    
    train_dataset = prep_dataset(dataset["train"], tokenizer)
    test_dataset = prep_dataset(dataset["test"], tokenizer)

    def generate(index, data_split="test"):
        # generates response for dataset index
        import gc
        functionList = dataset[data_split][index]["functionList"]
        userPrompt = dataset[data_split][index]["userPrompt"]
        correct_answer = dataset[data_split][index]["assistantResponse"]
        model.config.use_cache = True
        prompt = f"{B_INST}{B_FUNC}{functionList}{E_FUNC}{userPrompt.strip()}{E_INST}"
        print(f"Using the {data_split} dataset, the prompt is:")
        print(prompt)
        
        inputs = tokenizer([prompt], return_tensors="pt").to("cuda")

        if "token_type_ids" in inputs:
            del inputs["token_type_ids"]

        output = model.generate(**inputs, max_new_tokens=200, pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id)
        # remove the input ids to only get the model's response.
        output_text = tokenizer.decode(output[0, len(inputs.input_ids[0]):], skip_special_tokens=True)
        print(f"The output tokens were {output}")
        print(f"The decoded output is: {output_text}")
        print(f"The correct answer is: {correct_answer}")
        del inputs
        gc.collect()
        torch.cuda.empty_cache()
    
    generate(0)

    ################################
    ## TRAINING ##
    ################################
    class CustomTrainer(Trainer):
        def compute_loss(self, model, inputs, return_outputs=False):
            num_tokens = 25
            labels = inputs.pop('labels')
            loss_mask = inputs.pop('loss_mask')

            # forward pass
            outputs = model(**inputs)
            logits = outputs.logits

            # Check for NaN in logits and labels
            if torch.isnan(logits).any():
                print("NaN in logits!")
                print(logits)
            
            # convert logits to probabilities with a softmax 
            probs = nn.functional.softmax(logits, dim=-1)
            # most probably tokens
            predicted_token_ids = torch.argmax(probs, dim=-1)
            # calculate the loss for each token
            loss_fct = nn.CrossEntropyLoss(reduction='none')
            print(f"Labels shape: {labels.shape}")
            print(f"Logits shape: {logits.shape}")
            # TODO VERIFY: Computes loss between [batch_size, seq_length, vocab_size,] and [batch_size, seq_length] labels? 
            losses = loss_fct(logits.view(-1, self.model.config.vocab_size), labels.view(-1)) 
            # TODO VERIFY: reshape losses from ? WHAT ? to dimensions [batch_size, seq_length] 
            losses = losses.view(-1, inputs['input_ids'].shape[-1])
            # apply loss mask, to avoid considering the prompt in the loss 
            masked_loss = losses * loss_mask
            # check for NaNs in losses and zero in the sum
            if torch.isnan(losses).any():
                print("NaN in masked_loss!")
                print(losses)
            if loss_mask.sum() == 0:
                print("Sum of loss_mask is 0!")
            # TODO VERIFY: aggregate the masked losses, normalized by sum of loss mask to upweigh longer sequences?
            loss = masked_loss.sum() / (loss_mask.sum() + 1e-10) 
            
            # debugging!
            # print formatted tokens
            # batch_size, seq_length = inputs['input_ids'].size()
            # num_tokens = len(inputs['input_ids'][0])
            # print("-"*100)
            # print(f"Token analysis for last {num_tokens} tokens:")
            # header_format = "{:<10}{:<20}{:<20}{:<20}{:<20}{:<30}{:<30}".format("Token", "Input ID", "Predicted ID", "Label", "Loss", "Loss Mask", "Loss * Mask")
            # print(header_format)
            # for i in range(num_tokens):
            #     token_input_id = inputs['input_ids'][0][i]
            #     token_predicted_id = predicted_token_ids[0][i]
            #     token_label = labels[0][i]
            #     token_loss = losses[0][i]
            #     token_loss_mask = loss_mask[0][i]
            #     token_loss_times_mask = losses[0][i] * loss_mask[0][i]
            #     print(f"{i:<10}{token_input_id:<20}{token_predicted_id:<20}{token_label:<20}{token_loss:<20}{token_loss_mask:<30}{token_loss_times_mask:<30}")
            return (loss, outputs) if return_outputs else loss

        def get_train_dataloader(self):
            train_dataset = self.train_dataset
            data_collator = self.data_collator
            dataloader_params = {
                "batch_size": self.args.train_batch_size,
                "collate_fn": data_collator,
                "num_workers": self.args.dataloader_num_workers,
                "pin_memory": self.args.dataloader_pin_memory
            }
            if not isinstance(train_dataset, torch.utils.data.IterableDataset):
                dataloader_params["sampler"] = self._get_train_sampler()
                dataloader_params["drop_last"] = self.args.dataloader_drop_last

            return DataLoader(train_dataset, **dataloader_params)

        def get_eval_dataloader(self):
            eval_dataset = self.eval_dataset
            data_collator = self.data_collator
            dataloader_params = {
                "batch_size": self.args.eval_batch_size,
                "collate_fn": data_collator,
                "num_workers": self.args.dataloader_num_workers,
                "pin_memory": self.args.dataloader_pin_memory
            }
            if not isinstance(eval_dataset, torch.utils.data.IterableDataset):
                dataloader_params["sampler"] = self._get_eval_sampler()
                dataloader_params["drop_last"] = False # Typically, you don't drop the last batch in evaluation.

            return DataLoader(eval_dataset, **dataloader_params)

    class CustomDataCollator: # needed if the EOS token is to be included in training
        def __init__(self, tokenizer):
            self.tokenizer = tokenizer
        
        def __call__(self, batch):
            input_ids = torch.stack([item["inputs_ids"] for item in batch])
            attention_mask = torch.stack([item["attention_mask"] for item in batch])
            labels = torch.stack([item["labels"] for item in batch])
            loss_mask = torch.stack([item["loss_mask"] for item in batch])
            # Debugging: print details of the first sequence in the batch
            num_elements_to_view = 20
            # decoding input ids
            decoded_input_tokens = self.tokenizer.convert_ids_to_tokens(input_ids[0][:num_elements_to_view])
            print(f"Decoded input tokens: {decoded_input_tokens}")
            return {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "labels": labels,
                "loss_mask": loss_mask
            }
    data_collator = CustomDataCollator(tokenizer)
    trainer = CustomTrainer(
        model=model,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        args=TrainingArguments(
            num_train_epochs=1, # only using 1 here for function calling
            per_device_train_batch_size=1,
            per_device_eval_batch_size=1,
            evaluation_strategy="steps",
            warmup_ratio=0.1,
            eval_steps=0.2,
            learning_rate=1e-4,
            bf16=True,
            logging_steps=1,
            output_dir="outputs",
            optim="adamw_torch",
            lr_scheduler_type="constant",
        ),
        data_collator=data_collator
    )
    model.config.use_cache= False # silences warning, reenable for inference.
    trainer.train() # TODO ???+
    torch.cuda.empty_cache()

@app.local_entrypoint()
def starter():
    main.remote()

if __name__ == "__main__":
    create_dataset()