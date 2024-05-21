## Predictors

Finetuned text classifiers for the GPU lower classes.

Get state of the art text classification for 10% of LLM API prices, on demand, with full control.
- Finetune your own LLMs on 1 GPU for text classification using unsloth.
- Laptops allowed: switch to remote GPU finetuning using Modal with one line of code. Persistently deploy in 1 line with Modal.
- Full control: keep your models, export weights for VLLM / Ollama, set custom training params, bring your own finetuning data, bring your own prompts

## Quickstart

```bash
 git clone https://github.com/baptistecumin/predictors.git
 cd ./predictors
 pip3 install -r predictors/requirements.txt
 touch .env 
```
Create an OPENAI_API_KEY, and a HUGGINGFACE_ACCESS_TOKEN. 

1. Run a simple classifier via API.
```python
from predictors.predictors import ZeroShotPredictor
from predictors.tasks import Predict, Classify, ClassifierClass
tasks = [
    Predict(
        name="square",
        description="What is the square of this number?",
        dtype='int'),
    Classify(
        name="is_prime",
        description="Is this prime number?",
        classes=[
            ClassifierClass(name="yes", description="Yes"),
            ClassifierClass(name="no", description="No"),
        ]
    )        
]
X = ["1", "2", "3", "3"]
cls = ZeroShotPredictor(
    tasks=tasks,
    model="gpt-4o",
)
print(cls.predict(X))
```

2. Finetune your own LLM classifier remotely via Modal + Unsloth. 

Set up Modal for remote training. 
> modal setup

```python
from predictors.predictors import FineTunedPredictor
from predictors.tasks import Classify, Predict
finetuned_model_name = "mjrdbds/llama3-4b-classifierunsloth-20240516-lora"
base_model_name = "unsloth/llama-3-8b-bnb-4bit"
n = FineTunedPredictor(
    model=finetuned_model_name,
    base_model_name=base_model_name,
    remote=True,
)
n.set_config(
    tasks=[Classify(name="classify", description="Classify the category of the input"),
            Predict(
                name="price",
                description="The price of the input product.")
    ], 
    prompt_template_file='classification_labels.jinja'
)
X = ["the product is not a piece of furniture", "the product is a piece of furniture"]
y = [{'classify': 'not furniture', 'price': 5}, {'classify': 'furniture', 'price': 10}]
n.fit(X, y) # can also pass a huggingface dataset here, or config as in train_config.py
print(n.predict(X))
```

Predictors are saved locally or to a Modal volume. They are keyed by finetuned_model_name - base_model_name, and persist from one session to the next, reloading from disc.
```python
from predictors.predictors import FineTunedPredictor
finetuned_model_name = "mjrdbds/llama3-4b-classifierunsloth-20240516-lora"
base_model_name = "unsloth/llama-3-8b-bnb-4bit"
n = FineTunedPredictor(
    model=finetuned_model_name,
    base_model_name=base_model_name,
    remote=True,
)
X = ["the product is not a piece of furniture", "the product is a piece of furniture"]
y = [{'classify': 'not furniture', 'price': 5}, {'classify': 'furniture', 'price': 10}]
print(n.predict(X)) # no need to retrain.
```

## Contribute

This is a POC and an experiment. If you like the idea, DM [me](https://x.com/baptiste_cumin). 

There's plenty of room to optimize this.
