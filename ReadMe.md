## Quickstart

Use LLMs as text classifiers. 

Finetune your own LLMs for text classification.

Run them locally, or remotely. Get SOTA performance for 1/10th the cost of APIs.

1. Run a simple classifier 
```python
from predictors.predictors import FewShotTeacherPredictor, Predict, Classify
tasks = [
    Predict(
        name="square",
        description="What is the square of this number?",
        dtype='int'),
    Classify(
        name="is_prime",
        description="Is this prime?",
        classes=[
            ClassifierClass(name="yes", description="Yes"),
            ClassifierClass(name="no", description="No"),
        ]
    )        
]
X_train = ["1", "2", "3", "3"]
ZeroShotPredictor(
    tasks=tasks,
    model="gpt-3.5-turbo",
    teacher_model="gpt-4-turbo"
)
```

2. Finetune your own LLM classifier via modal + unsloth. Finetune 

Set up Modal for remote training. 
> modal setup

```python
from predictors.predictors import FineTunedPredictor
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
n.fit(X, y)
print(n.predict(X))
```

Predictors are saved locally or to a volume, they persist from one session to the next. 
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
print(n.predict(X))
```
