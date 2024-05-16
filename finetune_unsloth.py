import os
from modal import App, Image, method
from datetime import datetime

image = (
    Image
    .debian_slim(python_version="3.10")
    .apt_install("git", "git-lfs")
    .pip_install("unsloth[cu121] @ git+https://github.com/unslothai/unsloth.git")
    .pip_install("torch")
)

app = App(
    "train-peft",
    image=image,
)

HF_ACCESS_TOKEN = os.getenv("HF_ACCESS_TOKEN")


@app.cls(gpu='T4', container_idle_timeout=240, image=image)
class FinetunedClassifier:
    classification_prompt = """Below is a set of classification labels, paired with an input. Classify the input into the closest of the provided classes. Only return the class, nothing else.

    ### Classification labels:
    {}

    ### Input:
    {}

    ### Closest label:
    {}"""

    def __init__(self, base_model_name, finetuned_model_name):
        self.model_name = base_model_name
        self.finetuned_model_name = finetuned_model_name

    @method()
    def train(self):
        """Train the model."""
        from datasets import load_dataset
        from trl import SFTTrainer
        from transformers import TrainingArguments
        from unsloth import FastLanguageModel
        import torch

        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=self.model_name,
            max_seq_length=2048,
            dtype=torch.float16,
            load_in_4bit=True,
            token=HF_ACCESS_TOKEN    
        )
        model = FastLanguageModel.get_peft_model(
            model,
            r = 16, # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128
            target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                            "gate_proj", "up_proj", "down_proj",],
            lora_alpha = 16,
            lora_dropout = 0, # Supports any, but = 0 is optimized
            bias = "none",    # Supports any, but = "none" is optimized
            # [NEW] "unsloth" uses 30% less VRAM, fits 2x larger batch sizes!
            use_gradient_checkpointing = "unsloth", # True or "unsloth" for very long context
            random_state = 3407,
            use_rslora = False,  # We support rank stabilized LoRA
            loftq_config = None, # And LoftQ
        )        

        dataset = load_dataset("mjrdbds/classifiers-finetuning-060525", split="train[:10%]")
        dataset = dataset.map(self.formatting_prompts_func(self.classification_prompt, tokenizer.eos_token), batched=True)

        # Configure training
        trainer = SFTTrainer(
            model=model,
            tokenizer=tokenizer,
            train_dataset=dataset,
            dataset_text_field="text",
            max_seq_length=2048,
            dataset_num_proc=2,
            args=TrainingArguments(
                per_device_train_batch_size=4,
                gradient_accumulation_steps=4,
                warmup_steps=5,
                max_steps=10,
                learning_rate=2e-4,
                fp16=True,
                logging_steps=1,
                optim="adamw_8bit",
                weight_decay=0.01,
                lr_scheduler_type="linear",
                seed=3407,
                output_dir="outputs",
            )
        )
        trainer.train()
        self.save_to_hub(model, tokenizer)
        
    def formatting_prompts_func(self, classification_prompt, eos_token):
        """Helper to format prompts for training."""
        def inner_formatting_prompts_func(examples):
            instructions = examples["vocabulary"]
            inputs = examples["input"]
            outputs = examples["label"] if "label" in examples else [""] * len(examples["input"])
            EOS_TOKEN = eos_token
            texts = [
                classification_prompt.format(instruction, input, output) + EOS_TOKEN
                for instruction, input, output in zip(instructions, inputs, outputs)
            ]
            return {"text": texts}

        return inner_formatting_prompts_func

    @method()
    def inference(self):
        """Run inference to classify input texts."""
        from datasets import load_dataset
        from unsloth import FastLanguageModel

        def predict_category(prompted_inputs):
            inputs = tokenizer(prompted_inputs, padding=True, truncation=True, return_tensors="pt").to("cuda")
            outputs = model.generate(**inputs, max_new_tokens=10, use_cache=True)
            generated_texts = tokenizer.batch_decode(outputs[:,inputs['input_ids'].shape[1]:], skip_special_tokens=True)
            return {'predicted_label': generated_texts}
        
        model, tokenizer = self.load_from_hub()
        FastLanguageModel.for_inference(model)  # Enable native 2x faster inference
        inference_dataset = load_dataset("mjrdbds/classifiers-finetuning-060525", split="train[:10%]")
        prompted_input = inference_dataset.map(self.formatting_prompts_func(self.classification_prompt, tokenizer.eos_token), batched=True)
        output = prompted_input.map(lambda batch: predict_category(batch['text']), batched=True, batch_size=4)
        return output.to_dict()

    def save_to_hub(self, model, tokenizer):
        """Save the model and tokenizer to Hugging Face's Model Hub."""
        model.push_to_hub(self.finetuned_model_name, token=HF_ACCESS_TOKEN)
        tokenizer.push_to_hub(self.finetuned_model_name, token=HF_ACCESS_TOKEN)

    def load_from_hub(self):
        """Load the model and tokenizer from Hugging Face's Model Hub."""
        from unsloth import FastLanguageModel
        import torch
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=self.finetuned_model_name,
            max_seq_length=2048,
            dtype=torch.float16,
            load_in_4bit=True,
            token=HF_ACCESS_TOKEN    
        )
        return model, tokenizer

@app.local_entrypoint()
def starter():
    current_date = datetime.now().strftime("%Y%m%d")
    finetuned_model_name = f"mjrdbds/llama3-4b-classifierunsloth-{current_date}-lora"
    t = FinetunedClassifier(
        base_model_name="unsloth/llama-3-8b-bnb-4bit",
        finetuned_model_name=finetuned_model_name)
    #t.train.remote()
    inference = t.inference.remote()
    print(inference)