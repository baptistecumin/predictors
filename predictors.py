"""
TODO: allow for finetuning on multiple tasks at once. Separate answers by <task>key:value</task>
"""
from modal.runner import deploy_app
from modal import Cls
from tasks import Classify
from abc import ABC, abstractmethod
from pydantic import BaseModel, Field, create_model
from typing import Optional, Type, Literal, Union, List, Any
from tasks import InputData, TrainExample, Classify, Predict, PromptTemplate
from finetune import app
import instructor
from litellm import completion
import logger
import dotenv
dotenv.load_dotenv()

from jinja2 import Environment, FileSystemLoader
template_dir = './prompts'
env = Environment(loader=FileSystemLoader(template_dir))

logger = logger.get_logger()

class BasePredictor(ABC, BaseModel):
    model: str = Field(..., description="The model identifier used for predictions.")
    tasks: List[Union[Classify, Predict]] = Field(..., description="A list of tasks that the predictor handles.")
    prompt_template_file: str = Field(..., description="File path to the user prompt template.")
    
    task_prompt_template_file: str = Field(None, description="File path for a task prompt. This adds more specificity to the function. ")
    prompt_template: PromptTemplate = Field(None, description="Compiled Jinja2 template for user prompts.")
    task_prompt_template: PromptTemplate = Field(None, description="Compiled Jinja2 template for task prompts.")
    response_model: Optional[Type[BaseModel]] = Field(None, description="Dynamically created response model based on tasks.")

    def model_post_init(self, __context) -> None:
        """
        Dynamically creates a response model based on the predictor tasks and allowed classes.
        """
        prediction_object_fields = {}
        for task in self.tasks:
            if type(task) == Predict:
                prediction_object_fields[task.name] = (task.dtype, Field(..., description=task.description))
            else:
                class_labels = [n.name for n in task.classes]
                class_labels_type = Literal[tuple(class_labels)] # type: ignore
                prediction_object_fields[task.name] = (class_labels_type, Field(..., description=task.description))
            if task.chain_of_thought:
                prediction_object_fields["chain_of_thought"] = ("str", Field(..., description=f"Think step by step to determine the correct {task.name}"))
        task_prompt_template = env.get_template(self.task_prompt_template_file) if self.task_prompt_template_file else None
        self.response_model = create_model(
            __model_name='Labels', 
            __doc__=task_prompt_template.render(tasks=self.tasks) if task_prompt_template else None,
            **prediction_object_fields,
            )
        self.prompt_template = env.get_template(self.prompt_template_file)
        return self

    @abstractmethod
    def fit(self, X, y):
        pass

    @abstractmethod
    def predict(self, X):
        pass
    
    def apply_prompt(self, x: str):
        return self.prompt_template.render(tasks=self.tasks, input=x)


class ZeroShotPredictor(BasePredictor):

    prompt_template_file: str = 'tasks_one_shot.jinja'
    task_prompt_template_file: str = 'tasks_description.jinja'

    def fit(self, X) -> List[int]:
        return self.predict(X)

    def predict(self, X) -> List[int]:
        X = InputData(data=X)
        client = instructor.from_litellm(completion)
        outputs = []
        for x in X.data:
            output = client.chat.completions.create(
                model=self.model,
                max_tokens=1024,
                messages=[
                    {
                        "role": "user",
                        "content": self.apply_prompt(x=x),
                    }
                ],
                response_model=self.response_model,
            ).dict()
            outputs.append([output[task.name] for task in self.tasks])
        # Simplify output: if each item only has one class, flatten the list
        if len(self.tasks) == 1:
            return [output[0] for output in outputs]
        return outputs


class FewShotTeacherPredictor(BasePredictor):
    teacher_model: str
    prompt_template_file: str = 'tasks_few_shot.jinja'
    few_shot_examples: List[TrainExample] = []

    def apply_prompt(self, x: str):
        few_shot_examples = [{'input': ex.input, 'label': str(ex.labels)} for ex in self.few_shot_examples]
        return self.prompt_template.render(
            tasks=self.tasks,
            few_shot_examples=few_shot_examples,
            input=x
        )

    def fit(self, X) -> List[TrainExample]:
        logger.debug(f"Fitting model using model {self.teacher_model}.")
        classifier = ZeroShotPredictor(
            model=self.teacher_model,
            tasks=self.tasks
        )
        predictions = classifier.predict(X)
        # note: we currently assume that the predictions are in the same order as the tasks, and are all present.
        self.few_shot_examples = [TrainExample(input=x, labels={self.tasks[i].name: y[i] for i in range(len(self.tasks))}) 
                                  for x, y in zip(X, predictions)]
        return self.few_shot_examples

    def predict(self, X):
        logger.debug("Predicting...")
        client = instructor.from_litellm(completion)
        outputs = []
        for x in X:
            output = client.chat.completions.create(
                model=self.model,
                max_tokens=1024,
                messages=[
                    {
                        "role": "user",
                        "content": self.apply_prompt(x=x),
                    }
                ],
                response_model=self.response_model,
            ).dict()
            outputs.append([output[task.name] for task in self.tasks])
        # Simplify output: if each item only has one class, flatten the list
        if len(self.tasks) == 1:
            return [output[0] for output in outputs]
        return outputs

class FineTunedPredictor(BaseModel): 
    """ TODO how do we merge this with the BasePredictor?"""
    base_model_name: str
    model: str = Field(..., description="The model identifier used for predictions.")
    tasks: List[Union[Classify, Predict]] = Field(..., description="A list of tasks that the predictor handles.")

    prompt_template_file: str = Field(..., description="File path to the user prompt template, deployed remotely to modal.")
    predictor: Any = Field(None, description="The model object deployed remotely to modal.")

    def model_post_init(self, __context) -> None:
        # how do we know if we should deploy the app or not? If it's initializing for the first time?
        deploy_app(app)
        cls = Cls.lookup("train-peft", "UnslothFinetunedClassifier")
        self.predictor = cls(finetuned_model_name=self.model, base_model_name=self.base_model_name)
        required_dataset_fields = self.predictor.set_task_prompt.remote(task=self.tasks[0], prompt_template_file=self.prompt_template_file)
        print("Required dataset fields: ", required_dataset_fields)

    def fit(self, X, y):
        if len(self.tasks) > 1:
            raise NotImplementedError("Only one task supported for now.")
        self.predictor.set_task_prompt.remote(task=self.tasks[0], prompt_template_file=self.prompt_template_file)
        dataset_dict = {'input': X, 'label': y}
        self.predictor.train.remote(dataset=dataset_dict)
        return self       
    
    def fit_hf(self, dataset):
        self.predictor.train.remote(dataset=dataset)
        return self

    def predict(self, X):
        if type(X) == List:
            dataset = {'input': X}
        elif type(X) == str:
            print('Interpreting X as a huggingface dataset str')
            dataset = X
        return self.predictor.predict.remote(dataset=dataset)

class FineTunedFewShotPredictor(BasePredictor):
    pass

class EmbeddingPredictor():
    """
    Prompted embeddings. Not currently used.
    """

if __name__ == "__main__":
    finetuned_model_name = "mjrdbds/llama3-4b-classifierunsloth-20240516-lora"
    base_model_name = "unsloth/llama-3-8b-bnb-4bit"
    prompt_template_file = 'classification_labels.jinja'
    tasks = [Classify(name="classify", description="Classify the category of the input")]
    
    # Option 1: fit with an in-memory dataset
    X = ["the product is not a piece of furniture", "the product is a piece of furniture"]
    y = ["not furniture", "furniture"]
    cls = FineTunedPredictor(
        tasks=tasks,
        model="mjrdbds/llama3-4b-classifierunsloth-20240516-lora",
        base_model_name=base_model_name,
        prompt_template_file=prompt_template_file,
    )
    cls.fit(X, y)
    print(cls.predict(X))

    # Option 2: fit with a huggingface dataset
    dataset = "mjrdbds/classifiers-finetuning-060525"
    cls.fit_hf(dataset) # automatically trains on train set
    print(cls.predict(dataset)) # automatically predicts on test set

    # Model persistence. The next day, boot up your model, it's still there.
    cls = FineTunedPredictor(
        tasks=tasks,
        model="mjrdbds/llama3-4b-classifierunsloth-20240516-lora",
        base_model_name=base_model_name,
        prompt_template_file=prompt_template_file,
    )
    X = ["the product is not a piece of furniture", "the product is a piece of furniture"]
    print(cls.predict(X))

