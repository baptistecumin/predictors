from classifiers import FewShotTeacherClassifier, ZeroShotClassifier, ClassifierClass, ClassifierTask

if __name__ == "__main__":
    tasks = [
        ClassifierTask(
            name="classif",
            description="Classify integers based on range",
            classes=[
                ClassifierClass(name='Low', description='Zero to one'),
                ClassifierClass(name='High', description='Two or more'),
            ]
        ),
        ClassifierTask(
            name="is_even",
            description="Is this even?",
            classes=[
                ClassifierClass(name='Yes', description='Even'),
                ClassifierClass(name='No', description='Odd'),
            ]
    )]
    X_train = ["1", "2", "3", "3"]
    clf = ZeroShotClassifier(
        tasks=tasks,
        model="command-r-plus",
    )
    print(clf.predict(X_train))
    # clf = FewShotTeacherClassifier(
    #     tasks=tasks,
    #     model='gpt-3.5-turbo',
    #     teacher_model="gpt-4-turbo"
    # )
    # clf.fit(X_train)
    # predictions = clf.predict(X_train)
    # print(predictions)