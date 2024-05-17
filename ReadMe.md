## Quickstart

Use LLMs as text classifiers. Finetune LLMs for text classification. 

1. Run a simple classifier 
```python
from predictors import ZeroShotPredictor, Predict
X_train = ["1", "2", "3", "3"]
clf = ZeroShotPredictor(
    tasks=[
        Predict(
            name="is_even",
            description="Is this even?",
            dtype='bool'
        )
    ],
    model="gpt-4-turbo",
)
clf.predict(X_train)
```
> [False, True, False, False]

2. Run one classifier for multiple predictions 

```python
from predictors import FewShotTeacherPredictor, Predict, Classify
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

3. Finetune your own LLM classifier 
