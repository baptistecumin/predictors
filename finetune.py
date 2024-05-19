from dataclasses import dataclass, field
from typing import Dict, List, Optional, Union
import json
import time
import os
from modal import App, Image, method, Secret, Mount, web_endpoint, Volume, Cls
from tasks import Classify, ClassifierClass
import dotenv
from jinja2 import Environment, FileSystemLoader, meta
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
    per_device_train_batch_size: int = 2
    gradient_accumulation_steps: int = 4
    warmup_steps: int = 5
    max_steps: int = 60
    learning_rate: float = 2e-4
    bf16: bool = False 
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
         timeout=3000,
         image=image, 
         secrets=[Secret.from_dotenv()],
         mounts=[Mount.from_local_dir("./prompts", remote_path="/prompts")],
         volumes={'/vol': volume})
class UnslothFinetunedClassifier:

    def __init__(self, finetuned_model_name, base_model_name):
        self.finetuned_model_name = finetuned_model_name
        self.base_model_name = base_model_name
        self.hf_access_token = os.getenv("HUGGING_FACE_ACCESS_TOKEN")
        self.task_config_dir = f"{TASK_CONFIG_DIR}/{self.finetuned_model_name}"
        
        if os.path.exists(self.task_config_dir):
            print('Task and prompt config exists. Loading from volume.')
            self.get_task_prompt(self.task_config_dir)

            if os.path.exists(f"{MODEL_WEIGHTS_DIR}/{finetuned_model_name}"):
                print('Weights exist. Initializing pre-configured model.')
                self.model, self.tokenizer = self.load_from_volume()
        else:
            print('Weights do not exist yet. Please configure task, prompt and train.')
        
    @method()
    def set_task_prompt(self, task, prompt_template_file="classification_labels.jinja"):
        """
        Configure the model with a task and a prompt template. Saves them to volume for persistence.
        Note we do this in configure instead of init as Modal functions are uniquely
        defined by the init call args. This ensures a subsequent call to uses a warm instance.
        """
        env = Environment(loader=FileSystemLoader('/prompts'))
        template_source = env.loader.get_source(env, prompt_template_file)[0]
        parsed_content = env.parse(template_source)
        fields_required_in_prompt = list(meta.find_undeclared_variables(parsed_content))
        fields_required_in_dataset = [i for i in fields_required_in_prompt if i not in ['task', 'label']]
        
        os.makedirs(self.task_config_dir, exist_ok=True)
        with open(f"{self.task_config_dir}/task.json", 'w') as f:
            json.dump(task.dict(), f)
        with open(f"{self.task_config_dir}/prompt_template_file.txt", 'w') as f:
            f.write(prompt_template_file)
        with open(f"{self.task_config_dir}/fields_required_in_prompt.txt", 'w') as f:
            f.write('\n'.join(fields_required_in_dataset))
        volume.commit()
        return list(fields_required_in_dataset)

    def get_task_prompt(self, task_config_dir):
        """Load task and prompt_template_file from volume if they exist."""
        try:
            with open(f"{task_config_dir}/task.json", 'r') as f:
                task_dict = json.load(f)
                task_dict = {k: v for k, v in task_dict.items() if v is not None}
                self.task = Classify(**task_dict)
            with open(f"{task_config_dir}/prompt_template_file.txt", 'r') as f:
                self.prompt_template_file = f.read()
            with open(f"{task_config_dir}/fields_required_in_prompt.txt", 'r') as f:
                self.fields_required_in_dataset = f.read().split('\n')
            env = Environment(loader=FileSystemLoader('/prompts'))
            self.prompt_template = env.get_template(self.prompt_template_file)
        except FileNotFoundError:
            print("No task configuration found.")

    def formatting_prompts_func(self, eos_token, train=True):
        """Helper to format prompts."""
        def inner_formatting_prompts_func(examples):
            print(f'Examples: {examples}')
            data = {field: examples[field] for field in self.fields_required_in_dataset}
            if train:
                data['label'] = examples['label']
            texts = []
            for i in range(len(examples[next(iter(examples))])): 
                print(f"Example {i}: {data}")
                example_data = {field: data[field][i] for field in data}
                example_data['task'] = self.task
                print(f"Example data: {example_data}")
                text = self.prompt_template.render(**example_data)
                if train:
                    text = text + eos_token
                texts.append(text)            
            print(f"Example prompt: {texts[0]}")
            return {"text": texts}

        return inner_formatting_prompts_func
    
    @method()
    def train(self, 
              dataset: Union[str, Dict],
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
            print(dataset)
            dataset = Dataset.from_dict(dataset)
        missing_fields = [field for field in self.fields_required_in_dataset if field not in dataset.column_names]
        if missing_fields:
            raise ValueError(f"Dataset is missing required fields: {missing_fields}")
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

    @method()
    def predict(self, dataset: Union[str, Dict]):
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
            dataset = load_dataset(dataset, split="train[:1%]")
        else:
            dataset = Dataset.from_dict(dataset)

        prompted_input = dataset.map(self.formatting_prompts_func(self.tokenizer.eos_token, train=False), batched=True, batch_size=4)
        output = prompted_input.map(lambda batch: predict_category(batch['text']), batched=True, batch_size=4)
        return output.to_dict()['predicted_label']
    
    @web_endpoint(method='POST')
    def inference_openai(self, request: ChatCompletionRequest):
        """TODO: not currently working. Parameterized functions have no web endpoints."""
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
        print(f"Saving model and tokenizer to {self.finetuned_model_name}.")
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
