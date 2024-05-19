from tasks import Predict, Classify, ClassifierClass
from predictors import ZeroShotPredictor,FewShotTeacherPredictor, Predict, Classify, FineTunedPredictor

if __name__ == "__main__":
    # Option 1: multi-class labels, out of box models with zero shot or few shot teacher-student classifiers.
    tasks = [
        Predict(
            name="number_squared",
            description="The square of the input number.",
            chain_of_thought=True,
            dtype='int'),
        Classify(
            name="is_prime",
            description="If the input number is a prime number.",
            classes=[
                ClassifierClass(name="yes", description="Yes, it is a prime"),
                ClassifierClass(name="no", description="No, it is not a prime"),
            ]
        )        
    ]
    X_train = ["1", "2", "3", "3"]
    y_train = [[1, "no"], [4, "yes"], [9, "yes"], [9, "yes"]]
    X_test = ["4", "5", "6", "7"]
    print(ZeroShotPredictor(tasks=tasks, model="claude-3-haiku-20240307").predict(X_test))
    cls = FewShotTeacherPredictor(
        tasks=tasks,
        model="claude-3-haiku-20240307",
        teacher_model="claude-3-haiku-20240307"
    )
    print(cls.fit(X_train))
    print(cls.predict(X_test))

    # Option 2: finetuned models. In memory dataset, or huggingface dataset.
    finetuned_model_name = "mjrdbds/example-model"
    base_model_name = "unsloth/llama-3-8b-bnb-4bit" # supports any unsloth variant
    prompt_template_file = 'classification_labels.jinja'
    tasks = [Classify(name="classify", description="Classify the category of the input")]
    X = ["the product is not a piece of furniture", "the product is a piece of furniture"]
    y = ["not furniture", "furniture"]
    cls = FineTunedPredictor(
        tasks=tasks,
        model="mjrdbds/llama3-4b-classifierunsloth-20240516-lora",
        base_model_name=base_model_name,
        prompt_template_file=prompt_template_file,
    )
    cls.fit(X, y) # can also fit a huggingface dataset with cls.fit_hf(mjrdbds/classifiers-finetuning-060525)
    print(cls.predict(X))

    # Supports model persistence. The next day, boot up your model, it's still there.
    cls = FineTunedPredictor(
        tasks=tasks,
        model="mjrdbds/llama3-4b-classifierunsloth-20240516-lora",
        base_model_name=base_model_name,
        prompt_template_file=prompt_template_file,
    )
    X = ["the product is not a piece of furniture", "the product is a piece of furniture"]
    print(cls.predict(X))