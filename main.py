from predictors import ZeroShotPredictor,FewShotTeacherPredictor, Predict, Classify, ClassifierClass

if __name__ == "__main__":
    from predictors import FewShotTeacherPredictor, Predict, Classify
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
    X_test = ["4", "5", "6", "7"]
    print(ZeroShotPredictor(tasks=tasks, model="claude-3-haiku-20240307").predict(X_test))
    cls = FewShotTeacherPredictor(
        tasks=tasks,
        model="claude-3-haiku-20240307",
        teacher_model="claude-3-haiku-20240307"
    )
    print(cls.fit(X_train))
    print(cls.predict(X_test))