"""
 Copied finetune_local.py with following modifications to run remotely.
1. Save and load from volume instead of locally
2. Imports via image.import, not local
3. Modal function decorators 
"""
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Union 
from modal import App, Image, method, Secret, Mount, web_endpoint, Volume, Cls
import json
import time
import os
import dotenv
from jinja2 import Environment, FileSystemLoader, meta
from .tasks import Classify, ClassifierClass

current_file_dir = os.path.dirname(os.path.realpath(__file__))
image = (
    Image
    .debian_slim(python_version="3.10")
    .apt_install("git", "git-lfs")
    .pip_install("unsloth[cu121] @ git+https://github.com/unslothai/unsloth.git")
    .pip_install_from_requirements(os.path.join(current_file_dir, 'requirements.txt'))
)
volume = Volume.from_name("model-weights-vol", create_if_missing=True)
app = App("train-peft", image=image)
ROOT_PATH = "/predictors_output/"
MODEL_WEIGHTS_DIR = "model"
TASKS_CONFIG_DIR = "tasks"
dir_path = os.path.dirname(os.path.realpath(__file__))
prompts_path = os.path.join(dir_path, 'prompts')

with image.imports():
    from unsloth import FastLanguageModel
    import torch
    from datasets import load_dataset, Dataset
    from trl import SFTTrainer
    from transformers import TrainingArguments as HFTrainingArguments

dotenv.load_dotenv()

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
         mounts=[Mount.from_local_dir(prompts_path, remote_path="/prompts")],
         volumes={ROOT_PATH: volume})
class UnslothFinetunedClassifier:

    def __init__(self, finetuned_model_name: str, base_model_name: str) -> None:
        self.finetuned_model_name = finetuned_model_name
        self.base_model_name = base_model_name
        self.hf_access_token = os.getenv("HUGGING_FACE_ACCESS_TOKEN")
        self.tasks_config_dir = os.path.join(ROOT_PATH, self.finetuned_model_name, TASKS_CONFIG_DIR)
        self.model_weights_dir = os.path.join(ROOT_PATH, self.finetuned_model_name, MODEL_WEIGHTS_DIR)

        if os.path.exists(self.tasks_config_dir):
            print('Tasks and prompt config exists. Loading from volume.')
            self.get_config()

            if os.path.exists(self.model_weights_dir):
                print('Weights exist. Initializing pre-configured model.')
                self.model, self.tokenizer = self.load_from_volume()
            else:
                print('Weights do not exist yet. Please configure tasks, prompt and train.')
        else:
            print("Could not find task and prompt config, please call set_config.")

    @method()
    def set_config(self, tasks: List[Classify], prompt_template_file: str) -> tuple:
        """
        Configure the model with a task and a prompt template. Checks for the template in the local 'prompts' directory,
        falls back to the current directory, and saves the template content and tasks to the config directory for persistence.
        Done separately from init to make sure Modal container routing works.
        """
        env = Environment(loader=FileSystemLoader('/prompts'))
        try:
            self.prompt_template = env.get_template(prompt_template_file)
        except TemplateNotFound:
            # Now, try the current python process directory if not found in 'prompts'
            env = Environment(loader=FileSystemLoader(os.getcwd()))
            self.prompt_template = env.get_template(prompt_template_file)
        template_source = env.loader.get_source(env, prompt_template_file)[0]
        parsed_content = env.parse(template_source)
        fields_required_in_prompt = list(meta.find_undeclared_variables(parsed_content))
        self.fields_required_in_inference_dataset = [i for i in fields_required_in_prompt if i not in ['tasks', 'labels']] 
        self.fields_required_in_train_dataset = self.fields_required_in_inference_dataset + [task.name for task in tasks] # labels
        self.tasks = tasks
        config_data = {
            "tasks": [t.dict() for t in self.tasks],
            "fields_required_in_train_dataset": self.fields_required_in_train_dataset,
            "fields_required_in_inference_dataset": self.fields_required_in_inference_dataset
        }

        os.makedirs(self.tasks_config_dir, exist_ok=True)
        with open(f"{self.tasks_config_dir}/config.json", 'w') as f:
            json.dump(config_data, f, indent=4)
        with open(f"{self.tasks_config_dir}/prompt_template_file.jinja", 'w') as f:
            f.write(template_source)
        volume.commit()
        return list(self.fields_required_in_train_dataset), list(self.fields_required_in_inference_dataset)

    def get_config(self) -> None:
        """Load tasks and the Jinja template content from the configuration directory if they exist."""
        try:
            # Load configuration data from JSON file
            with open(f"{self.tasks_config_dir}/config.json", 'r') as f:
                config_data = json.load(f)
            
            self.tasks = [Classify(**task_dict) for task_dict in config_data['tasks'] if task_dict is not None]
            self.fields_required_in_train_dataset = config_data['fields_required_in_train_dataset']
            self.fields_required_in_inference_dataset = config_data['fields_required_in_inference_dataset']

            env = Environment(loader=FileSystemLoader(self.tasks_config_dir))
            self.prompt_template = env.get_template("prompt_template_file.jinja")
        
        except FileNotFoundError as e:
            print(f"No task configuration found: {e}")

    def formatting_prompts_func(self, eos_token: str, train: bool = True) -> callable:
        """Helper to format prompts. Note labels are reformatted from example[class_name]: value to example['labels'][class_name] = value"""
        def inner_formatting_prompts_func(examples):
            texts = []
            for i in range(len(examples[next(iter(examples))])): 
                example = {field: examples[field][i] for field in examples}
                if train:
                    example['labels'] = {}
                    print(example)
                    try: 
                        for task in self.tasks: # allows us to unpack it in jinja.
                            example['labels'][task.name] = example.pop(task.name)
                    except KeyError:
                        raise ValueError(f'Dataset does not contain the required fields: {self.fields_required_in_train_dataset}. Found {example}.')
                example['tasks'] = self.tasks
                text = self.prompt_template.render(**example)
                if train:
                    text = text + eos_token
                texts.append(text)            
                if i == 0:
                    print(f"Example prompt: {texts[0]}, from {example}")
            return {"text": texts}
        return inner_formatting_prompts_func
    
    @method()
    def fit(self, 
            X: List[str], 
            y: List[Dict[str, Union[str, int]]], 
            **kwargs: Dict) -> None:
        dataset = self.dataset_loader(dataset=X, y=y)
        return self._train(dataset, **kwargs)

    @method()
    def train(self, 
              dataset: Union[str, Dict[str, Union[List[str], List[Dict[str, Union[str, int]]]]]],
              training_arguments: TrainingArguments = TrainingArguments(), 
              training_peft_arguments: TrainingPeftArguments = TrainingPeftArguments()) -> None:
        dataset = self.dataset_loader(dataset, split="train")
        missing_fields = [field for field in self.fields_required_in_train_dataset if field not in dataset.column_names]
        if missing_fields:
            raise ValueError(f"Dataset is missing required fields: {missing_fields}. Found: {dataset.column_names}")
        return self._train(dataset, training_arguments, training_peft_arguments)

    def _train(self,
               dataset: Dataset,
               training_arguments: TrainingArguments = TrainingArguments(),
               training_peft_arguments: TrainingPeftArguments = TrainingPeftArguments()) -> None:
        """Train the base model. Dataset can either be a string, interpreted as a huggingface dataset, or a dict."""

        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=self.base_model_name,
            max_seq_length=2048,
            dtype=torch.float16,
            load_in_4bit=True,
            token=self.hf_access_token    
        )
        model = FastLanguageModel.get_peft_model(model, **training_peft_arguments.__dict__)
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

    def dataset_loader(self, dataset, split=None, y=None):
        if y is not None and isinstance(dataset, list): # passed .fit(X, y)
            dataset = {'input': dataset}
            try:
                for task in self.tasks:
                    dataset[task.name] = [y[i][task.name] for i in range(len(y))]
            except KeyError as e:
                raise ValueError(f"Dataset is missing required fields: {e}. Found: {y[0]}")
            print(dataset)
            dataset = Dataset.from_dict(dataset)
        elif isinstance(dataset, str):
            dataset = load_dataset(dataset, split=split)
        elif isinstance(dataset, list):
            if isinstance(dataset[0], str):
                dataset = Dataset.from_dict({'input': dataset})
            elif isinstance(dataset[0], dict):
                dataset = Dataset.from_list(dataset)
        elif isinstance(dataset, dict):
            dataset = Dataset.from_dict(dataset)
        else:
            raise ValueError(f"Unrecognized dataset type: {type(dataset)}")
        return dataset

    @staticmethod
    def _predict(tokenizer, model: FastLanguageModel, prompts: List[str]) -> Dict[str, List[str]]:
        print(type(tokenizer))
        inputs = tokenizer(prompts, padding=True, truncation=True, return_tensors="pt").to("cuda")
        outputs = model.generate(**inputs, max_new_tokens=10, use_cache=True)
        generated_texts = tokenizer.batch_decode(outputs[:,inputs['input_ids'].shape[1]:], skip_special_tokens=True)
        return {'predicted_label': generated_texts}

    @method()
    def predict(self, dataset) -> List[str]:
        """Run inference to classify input texts."""
        dataset = self.dataset_loader(dataset, 'test')
        missing_fields = [field for field in self.fields_required_in_inference_dataset if field not in dataset.column_names]
        if missing_fields:
            raise ValueError(f"Dataset is missing required fields: {missing_fields}. Found: {dataset.column_names}")
        
        FastLanguageModel.for_inference(self.model)
        prompted_input = dataset.map(self.formatting_prompts_func(self.tokenizer.eos_token, train=False), batched=True, batch_size=4)
        output = prompted_input.map(lambda batch: self._predict(self.tokenizer, self.model, batch['text']), batched=True, batch_size=4)
        output = output.to_dict()['predicted_label']
        for i in range(len(output)):
            try:
                output[i] = json.loads(output[i])
            except json.JSONDecodeError:
                output[i] = None
        return output
        
    def save_to_hub(self, model: FastLanguageModel, tokenizer) -> None:
        """Save the model and tokenizer to Hugging Face's Model Hub."""
        print(f"Saving model and tokenizer to {self.finetuned_model_name}.")
        model.push_to_hub(self.finetuned_model_name, token=self.hf_access_token)
        tokenizer.push_to_hub(self.finetuned_model_name, token=self.hf_access_token)

    def load_from_hub(self, hf_model_name: str) -> tuple:
        """Load the model and tokenizer from Hugging Face's Model Hub."""
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=hf_model_name,
            max_seq_length=2048,
            dtype=torch.float16,
            load_in_4bit=True,
            token=self.hf_access_token    
        )
        return model, tokenizer

    def load_from_volume(self) -> tuple:
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=self.model_weights_dir,
            max_seq_length=2048,
            dtype=torch.float16,
            load_in_4bit=True
        )
        return model, tokenizer

    def save_to_volume(self, model: FastLanguageModel, tokenizer) -> None:
        model.save_pretrained(self.model_weights_dir)
        tokenizer.save_pretrained(self.model_weights_dir)
        volume.commit()

