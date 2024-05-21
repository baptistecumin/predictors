import os
from modal.runner import deploy_app
from modal import Cls
from abc import ABC, abstractmethod
from pydantic import BaseModel, Field, create_model
from typing import Optional, Type, Literal, Union, List, Any
import instructor
from litellm import completion
import dotenv
dotenv.load_dotenv()

from .logger import get_logger
from .tasks import InputData, TrainExample, Classify, Predict, PromptTemplate
from .finetune_modal import app
from .finetune_local import UnslothFinetunedClassifier

from jinja2 import Environment, FileSystemLoader
def load_prompts_jinja_env():
    dir_path = os.path.dirname(os.path.realpath(__file__))
    prompts_path = os.path.join(dir_path, 'prompts')  # assumes prompts dir is in the same dir as this file
    env = Environment(loader=FileSystemLoader(prompts_path))
    return env
logger = get_logger()

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
        
        env = load_prompts_jinja_env()
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
    base_model_name: str
    model: str = Field(..., description="The model identifier used for predictions.")
    remote: bool = Field(..., description="Whether the model is deployed remotely to modal or trained locally.")

    tasks: List[Union[Classify, Predict]] = Field([], description="A list of tasks that the predictor handles.")
    prompt_template_file: str = Field("", description="File path to the user prompt template, deployed remotely to modal.")
    predictor: Any = Field(None, description="Set internally. The model object deployed remotely to modal.")

    def model_post_init(self, __context) -> None:
        if self.remote:
            print('Deploying container for model...', self.model)
            deploy_app(app)
            cls = Cls.lookup("train-peft", "UnslothFinetunedClassifier")
            self.predictor = cls(finetuned_model_name=self.model, base_model_name=self.base_model_name)
        else:
            self.predictor = UnslothFinetunedClassifier(finetuned_model_name=self.model, base_model_name=self.base_model_name)

    def set_config(self, tasks: List[Union[Classify, Predict]], prompt_template_file: str):
        if self.remote:
            return self.predictor.set_config.remote(tasks, prompt_template_file)
        return self.predictor.set_config(tasks, prompt_template_file)

    def fit(self, X, y=None):
        """ 
        X can be an hf dataset, a list of input strings, a list of dicts.
        """
        if y is not None:
            if self.remote:
                return self.predictor.fit.remote(X, y)
            return self.predictor.fit(X, y)
        else: 
            if self.remote:
                return self.predictor.fit.remote(X)
            return self.predictor.fit(X)

    def predict(self, X):
        if self.remote:
            return self.predictor.predict.remote(X)
        return self.predictor.predict(X)

