from predictors import ZeroShotPredictor,FewShotTeacherPredictor, Predict, Classify, ClassifierClass, FineTunedPredictor

if __name__ == "__main__":
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

    # ideal interface. note classifier only for now.
    tasks = [
        Classify(
            name="category",
            description="The category of the input text.",
            classes=[
                ClassifierClass(name="furniture", description="Is the item a piece of furniture"),
                ClassifierClass(name="decorative", description="Is the item a decorative object"),
            ]
        )
    ]
    X_train = ["table", "chair", "mirror"]
    y_train = ["furniture", "furniture", "decorative"]
    cls = FineTunedPredictor(
        tasks=tasks,
        model="unsloth/llama-3-8b-bnb-4bit",
        hf_model_name="mjrdbds/llama3-4b-classifierunsloth-20240517-lora"
    )
    cls.train(X_train, y_train)
    print(cls.predict(X_test))
    