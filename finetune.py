import os
from typing import Optional
from modal import App, Image, method, Secret, Mount
from tasks import Classify, ClassifierClass
from datetime import datetime
import dotenv
from jinja2 import Environment, FileSystemLoader

dotenv.load_dotenv()

image = (
    Image
    .debian_slim(python_version="3.10")
    .apt_install("git", "git-lfs")
    .pip_install("unsloth[cu121] @ git+https://github.com/unslothai/unsloth.git")
    # .pip_install("torch", "python-dotenv", "pydantic", "jinja2")
    .pip_install_from_requirements('requirements.txt')
)

app = App("train-peft", image=image)


@app.cls(gpu='T4', 
         container_idle_timeout=30, 
         image=image, 
         secrets=[Secret.from_dotenv()],
         mounts=[Mount.from_local_dir("./prompts", remote_path="/prompts")])
class UnslothFinetunedClassifier:

    prompt_template_file = "classification_finetune.jinja"

    DEFAULT_TRAINING_ARGUMENTS = {
        'per_device_eval_batch_size': 4,
        'per_device_train_batch_size': 4,
        'gradient_accumulation_steps': 4,
        'warmup_steps': 5,
        'max_steps': 10,
        'learning_rate': 2e-4,
        'fp16': True,
        'logging_steps': 1,
        'optim': "adamw_8bit",
        'weight_decay': 0.01,
        'lr_scheduler_type': "linear",
        'seed': 3407,
        'output_dir': "outputs",
    }

    DEFAULT_TRAINING_PEFT_ARGUMENTS = {
        'r': 16, # Choose any number > 0. Suggested 8, 16, 32, 64, 128
        'target_modules': ["q_proj", "k_proj", "v_proj", "o_proj",
                           "gate_proj", "up_proj", "down_proj",],
        'lora_alpha': 16,
        'lora_dropout': 0, # Supports any, but = 0 is optimized
        'bias': "none",    # Supports any, but = "none" is optimized
        'use_gradient_checkpointing': "unsloth", # True or "unsloth" for very long context
        'random_state': 3407,
        'use_rslora': False,  # We support rank stabilized LoRA
        'loftq_config': None, # And LoftQ
    }

    def __init__(self, 
                 model_name, 
                 task,
                 prompt_template_file=None,
                 base_model_name=None, # not needed for inference 
                 training_arguments=None,
                 training_peft_arguments=None):
        self.model_name = model_name
        self.base_model_name = base_model_name
        self.task = task
        self.training_arguments = self.DEFAULT_TRAINING_ARGUMENTS if training_arguments is None else training_arguments
        self.training_peft_arguments = self.DEFAULT_TRAINING_PEFT_ARGUMENTS if training_peft_arguments is None else training_peft_arguments
        self.hf_access_token = os.getenv("HUGGING_FACE_ACCESS_TOKEN")
        env = Environment(loader=FileSystemLoader('/prompts'))
        self.prompt_template_file = self.prompt_template_file if prompt_template_file is None else prompt_template_file
        self.prompt_template = env.get_template(self.prompt_template_file)
        

    @method()
    def train(self, dataset):
        """Train the base model. Dataset can either be a string, interpreted as a huggingface dataset, or a dict."""
        from datasets import load_dataset, Dataset
        from trl import SFTTrainer
        from transformers import TrainingArguments
        from unsloth import FastLanguageModel
        import torch

        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=self.base_model_name,
            max_seq_length=2048,
            dtype=torch.float16,
            load_in_4bit=True,
            token=self.hf_access_token    
        )
        model = FastLanguageModel.get_peft_model(model, **self.training_peft_arguments)
        
        if type(dataset) == str:
            dataset = load_dataset(dataset, split="train")
        else:
            dataset = Dataset.from_dict(dataset)

        dataset = dataset.map(self.formatting_prompts_func(tokenizer.eos_token), batched=True)

        # Configure training
        trainer = SFTTrainer(
            model=model,
            tokenizer=tokenizer,
            train_dataset=dataset,
            dataset_text_field="text",
            max_seq_length=2048,
            dataset_num_proc=2,
            args=TrainingArguments(**self.training_arguments)
        )
        trainer.train()
        self.save_to_hub(model, tokenizer, self.model_name)
        
    def formatting_prompts_func(self, eos_token):
        """Helper to format prompts for training."""
        def inner_formatting_prompts_func(examples):
            inputs = examples["input"]
            vocabulary = examples["vocabulary"]
            labels = examples["label"] if "label" in examples else [""] * len(examples["input"])
            tasks = [self.task] * len(examples["input"]) if "vocabulary" not in examples \
                else [self.task.set_classes(v) for v in vocabulary]
            texts = [
                self.prompt_template.render(task=task, input=input, label=label) + eos_token
                for input, label, task in zip(inputs, labels, tasks)
            ]
            print(f"Example prompt: {texts[0]}")
            return {"text": texts}

        return inner_formatting_prompts_func

    @method()
    def inference(self, dataset):
        """Run inference to classify input texts"""
        from unsloth import FastLanguageModel
        from datasets import Dataset, load_dataset
        def predict_category(prompted_inputs):
            inputs = tokenizer(prompted_inputs, padding=True, truncation=True, return_tensors="pt").to("cuda")
            outputs = model.generate(**inputs, max_new_tokens=10, use_cache=True)
            generated_texts = tokenizer.batch_decode(outputs[:,inputs['input_ids'].shape[1]:], skip_special_tokens=True)
            return {'predicted_label': generated_texts}
        
        model, tokenizer = self.load_from_hub(self.model_name)
        FastLanguageModel.for_inference(model)  # Enable native 2x faster inference
        if type(dataset) == str:
            dataset = load_dataset(dataset, split="train")
        else:
            dataset = Dataset.from_dict(dataset)

        prompted_input = dataset.map(self.formatting_prompts_func(tokenizer.eos_token), batched=True)
        output = prompted_input.map(lambda batch: predict_category(batch['text']), batched=True, batch_size=4)
        return output.to_dict()['predicted_label']

    def save_to_hub(self, model, tokenizer, hf_model_name):
        """Save the model and tokenizer to Hugging Face's Model Hub."""
        print(f"Saving model and tokenizer to {hf_model_name} with token {self.hf_access_token}.")
        model.push_to_hub(hf_model_name, token=self.hf_access_token)
        tokenizer.push_to_hub(hf_model_name, token=self.hf_access_token)

    def load_from_hub(self, hf_model_name):
        """Load the model and tokenizer from Hugging Face's Model Hub."""
        from unsloth import FastLanguageModel
        import torch
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=hf_model_name,
            max_seq_length=2048,
            dtype=torch.float16,
            load_in_4bit=True,
            token=self.hf_access_token    
        )
        return model, tokenizer

@app.local_entrypoint()
def starter():
    current_date = datetime.now().strftime("%Y%m%d")
    model_name = f"mjrdbds/llama3-4b-classifierunsloth-{current_date}-lora"
    t = UnslothFinetunedClassifier(
        base_model_name="unsloth/llama-3-8b-bnb-4bit",
        model_name=model_name,
        hf_access_token=os.getenv("HF_ACCESS_TOKEN")
    )
    #t.train.remote(dataset="mjrdbds/llama-3-8b-bnb-4bit")
    inference_dataset = {
        'input': ['this is an antique table very nice yay', 'A painting for use'],
        'vocabulary': [['table', 'painting'], ['table', 'painting']]
    }
    inference = t.inference.remote(inference_dataset)
    print(inference)

if __name__ == "__main__":
    model_name = "mjrdbds/llama3-4b-classifierunsloth-20240516-lora"
    base_model_name = "unsloth/llama-3-8b-bnb-4bit"
    dataset = "mjrdbds/classifiers-finetuning-060525"
    from modal.runner import deploy_app
    from modal import Cls
    d = deploy_app(app)
    print(f"Deployed modal app: {d}")
    task = Classify(
            name="category",
            description="The category of the input text.",
            classes=[
                ClassifierClass(name="furniture", description="Is the item a piece of furniture"),
                ClassifierClass(name="not furniture", description="Is the item not a piece of furniture"),
            ]
    )
    _UnslothFinetunedClassifier = Cls.lookup("train-peft", "UnslothFinetunedClassifier")
    predictor = _UnslothFinetunedClassifier(
        model_name=model_name,
        base_model_name="unsloth/llama-3-8b-bnb-4bit",
        task=task,
    )
    print(f"Beginning training on {base_model_name}, dataset {dataset}. View logs within Modal.")
    predictor.train.remote(dataset=dataset)
    print(f"Finished training on {base_model_name}, dataset {dataset}, output model in {model_name}. View model in huggingface.")
    print(f"Beginning inference on {base_model_name}, dataset {dataset}. View logs within Modal.")
    inference = predictor.inference.remote(dataset=dataset)
    print(inference)
    