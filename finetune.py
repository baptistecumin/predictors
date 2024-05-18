"""
TODO - Keep the model in memory for many subsequent inference calls? 
"""
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Union
import json
import time
import os
from modal import App, Image, method, Secret, Mount, web_endpoint, Volume, Cls
from tasks import Classify, ClassifierClass
import dotenv
from jinja2 import Environment, FileSystemLoader
from openai_models import ChatCompletionRequest, ChatCompletionMessageToolCall, ChatMessage, Function

dotenv.load_dotenv()

image = (
    Image
    .debian_slim(python_version="3.10")
    .apt_install("git", "git-lfs")
    .pip_install("unsloth[cu121] @ git+https://github.com/unslothai/unsloth.git")
    .pip_install_from_requirements('requirements.txt')
)
volume = Volume.from_name("model-weights-vol", create_if_missing=True)
MODEL_WEIGHTS_DIR = "/vol/"
TASK_CONFIG_DIR = "/vol/task_config"
app = App("train-peft", image=image)

@dataclass
class TrainingArguments:
    per_device_eval_batch_size: int = 4
    per_device_train_batch_size: int = 4
    gradient_accumulation_steps: int = 4
    warmup_steps: int = 5
    max_steps: int = 10
    learning_rate: float = 2e-4
    fp16: bool = True
    logging_steps: int = 1
    optim: str = "adamw_8bit"
    weight_decay: float = 0.01
    lr_scheduler_type: str = "linear"
    seed: int = 3407
    output_dir: str = "outputs"

@dataclass
class TrainingPeftArguments:
    r: int = 16
    target_modules: List[str] = field(default_factory=lambda: ["q_proj", "k_proj", "v_proj", "o_proj",
                                                               "gate_proj", "up_proj", "down_proj"])
    lora_alpha: int = 16
    lora_dropout: int = 0
    bias: str = "none"
    use_gradient_checkpointing: Union[bool, str] = "unsloth"
    random_state: int = 3407
    use_rslora: bool = False
    loftq_config: Optional[Dict] = None

@app.cls(gpu='T4', 
         container_idle_timeout=30, 
         image=image, 
         secrets=[Secret.from_dotenv()],
         mounts=[Mount.from_local_dir("./prompts", remote_path="/prompts")],
         volumes={'/vol': volume})
class UnslothFinetunedClassifier:

    def __init__(self, finetuned_model_name, base_model_name):
        self.finetuned_model_name = finetuned_model_name
        self.base_model_name = base_model_name
        self.task = None
        self.prompt_template = None
        self.hf_access_token = os.getenv("HUGGING_FACE_ACCESS_TOKEN")

        if os.path.exists(f"{MODEL_WEIGHTS_DIR}/{finetuned_model_name}"):
            print('Weights exist! Initializing')
            self.model, self.tokenizer = self.load_from_volume()
        else:
            print('Weights do not exist yet')
        
        self.get_task_prompt()

    @method()
    def set_task_prompt(self, task, prompt_template_file='classification_finetune.jinja'):
        """Save task and prompt_template_file to volume for persistence.
        Note we do this in configure instead of init as containers are uniquely
        defined within by the init call."""
        self.task = task
        self.prompt_template_file = prompt_template_file

        os.makedirs(TASK_CONFIG_DIR, exist_ok=True)
        with open(f"{TASK_CONFIG_DIR}/task.json", 'w') as f:
            json.dump(task.dict(), f)
        with open(f"{TASK_CONFIG_DIR}/prompt_template_file.txt", 'w') as f:
            f.write(prompt_template_file)
        volume.commit()

    def get_task_prompt(self):
        """Load task and prompt_template_file from volume if they exist."""
        try:
            with open(f"{TASK_CONFIG_DIR}/task.json", 'r') as f:
                task_dict = json.load(f)
                self.task = Classify(**task_dict)
            with open(f"{TASK_CONFIG_DIR}/prompt_template_file.txt", 'r') as f:
                self.prompt_template_file = f.read()

            env = Environment(loader=FileSystemLoader('/prompts'))
            self.prompt_template = env.get_template(self.prompt_template_file)
        except FileNotFoundError:
            print("No task configuration found.")

    @method()
    def train(self, 
              dataset, 
              training_arguments: TrainingArguments = TrainingArguments(), 
              training_peft_arguments: TrainingPeftArguments = TrainingPeftArguments()):
        """Train the base model. Dataset can either be a string, interpreted as a huggingface dataset, or a dict."""
        from datasets import load_dataset, Dataset
        from trl import SFTTrainer
        from transformers import TrainingArguments as HFTrainingArguments
        from unsloth import FastLanguageModel
        import torch

        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=self.base_model_name,
            max_seq_length=2048,
            dtype=torch.float16,
            load_in_4bit=True,
            token=self.hf_access_token    
        )
        model = FastLanguageModel.get_peft_model(model, **training_peft_arguments.__dict__)
        
        if isinstance(dataset, str):
            dataset = load_dataset(dataset, split="train")
        else:
            dataset = Dataset.from_dict(dataset)

        dataset = dataset.map(self.formatting_prompts_func(tokenizer.eos_token), batched=True)

        trainer = SFTTrainer(
            model=model,
            tokenizer=tokenizer,
            train_dataset=dataset,
            dataset_text_field="text",
            max_seq_length=2048,
            dataset_num_proc=2,
            args=HFTrainingArguments(**training_arguments.__dict__),
        )
        trainer.train()
        self.save_to_hub(model, tokenizer)
        self.save_to_volume(model, tokenizer)

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
        """Run inference to classify input texts."""
        from unsloth import FastLanguageModel
        from datasets import Dataset, load_dataset

        def predict_category(prompted_inputs):
            inputs = self.tokenizer(prompted_inputs, padding=True, truncation=True, return_tensors="pt").to("cuda")
            outputs = self.model.generate(**inputs, max_new_tokens=10, use_cache=True)
            generated_texts = self.tokenizer.batch_decode(outputs[:,inputs['input_ids'].shape[1]:], skip_special_tokens=True)
            return {'predicted_label': generated_texts}
        
        FastLanguageModel.for_inference(self.model)  # Enable native 2x faster inference
        if isinstance(dataset, str):
            dataset = load_dataset(dataset, split="train")
        else:
            dataset = Dataset.from_dict(dataset)

        prompted_input = dataset.map(self.formatting_prompts_func(self.tokenizer.eos_token), batched=True)
        output = prompted_input.map(lambda batch: predict_category(batch['text']), batched=True, batch_size=4)
        return output.to_dict()['predicted_label']
    
    @web_endpoint(method='POST')
    def inference_openai(self, request: ChatCompletionRequest):
        results_dict = {"name": "test", "description": "test"}
        json_results_dict = json.dumps(results_dict)
        tool_calls = ChatCompletionMessageToolCall(id='1',
            function=Function(name="Response", arguments=json_results_dict), 
                                                args=[])
        message = ChatMessage(role="user", content="Say this is a test", tool_calls=[tool_calls])
        return {
            "id": "1337",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": request.model,
            "choices": [{
                "message": message,
            }],
        }

    def save_to_hub(self, model, tokenizer):
        """Save the model and tokenizer to Hugging Face's Model Hub."""
        print(f"Saving model and tokenizer to {self.finetuned_model_name} with token {self.hf_access_token}.")
        model.push_to_hub(self.finetuned_model_name, token=self.hf_access_token)
        tokenizer.push_to_hub(self.finetuned_model_name, token=self.hf_access_token)

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

    def load_from_volume(self):
        from unsloth import FastLanguageModel
        import torch
        path = f"{MODEL_WEIGHTS_DIR}/{self.finetuned_model_name}"
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=path,
            max_seq_length=2048,
            dtype=torch.float16,
            load_in_4bit=True
        )
        return model, tokenizer

    def save_to_volume(self, model, tokenizer):
        path = f"{MODEL_WEIGHTS_DIR}/{self.finetuned_model_name}"
        model.save_pretrained(path)
        tokenizer.save_pretrained(path)
        volume.commit()

if __name__ == "__main__":
    from modal.runner import deploy_app
    from tasks import Classify, ClassifierClass

    deploy_app(app)

    task = Classify(
        name="category",
        description="The category of the input text.",
        classes=[
            ClassifierClass(name="furniture", description="Is the item a piece of furniture"),
            ClassifierClass(name="not furniture", description="Is the item not a piece of furniture"),
        ]
    )
    
    finetuned_model_name = "mjrdbds/llama3-4b-classifierunsloth-20240516-lora"
    base_model_name = "unsloth/llama-3-8b-bnb-4bit"
    dataset = "mjrdbds/classifiers-finetuning-060525"

    training_arguments = TrainingArguments()
    training_peft_arguments = TrainingPeftArguments()

    classifier = UnslothFinetunedClassifier(
        finetuned_model_name=finetuned_model_name,
        base_model_name=base_model_name
    )

    classifier.set_task_prompt.remote(task=task, prompt_template_file="classification_finetune.jinja")

    classifier.train.remote(dataset=dataset)
    result = classifier.inference.remote(dataset=dataset)
    print(result)

    ## The next day
    cls = Cls.lookup("train-peft", "UnslothFinetunedClassifier")
    m1 = cls(finetuned_model_name=finetuned_model_name, base_model_name=base_model_name)
    print(m1.inference.remote(dataset=dataset))
