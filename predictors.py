from abc import ABC, abstractmethod
from pydantic import BaseModel, Field, create_model
from typing import Optional, Type, Literal, Union, List
from tasks import InputData, TrainExample, Classify, Predict, PromptTemplate

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
    user_prompt_template_file: str = Field(..., description="File path to the user prompt template.")
    task_prompt_template_file: str = Field(..., description="File path to the task prompt template.")
    user_prompt_template: PromptTemplate = Field(None, description="Compiled Jinja2 template for user prompts.")
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
        task_prompt_template = env.get_template(self.task_prompt_template_file)
        self.response_model = create_model(
            __model_name='Labels', 
            __doc__=task_prompt_template.render(tasks=self.tasks),
            **prediction_object_fields,
            )
        self.user_prompt_template = env.get_template(self.user_prompt_template_file)
        return self

    @abstractmethod
    def fit(self, X, y):
        pass

    @abstractmethod
    def predict(self, X):
        pass
    
    def apply_prompt(self, x: str):
        return self.user_prompt_template.render(tasks=self.tasks, input=x)


class ZeroShotPredictor(BasePredictor):

    user_prompt_template_file: str = 'tasks_one_shot.jinja'
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
    user_prompt_template_file: str = 'tasks_few_shot.jinja'
    few_shot_examples: List[TrainExample] = []

    def apply_prompt(self, x: str):
        few_shot_examples = [{'input': ex.input, 'label': str(ex.labels)} for ex in self.few_shot_examples]
        return self.user_prompt_template.render(
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

class FineTunedPredictor(BasePredictor):

    def train(self, X, y):
        pass



    def predict(self, X):
        pass

class FineTunedFewShotPredictor(BasePredictor):
    pass

class EmbeddingPredictor():
    """
    Prompted embeddings. Not currently used.
    """

    # from InstructorEmbedding import INSTRUCTOR
    # model = INSTRUCTOR('hkunlp/instructor-large')
    # sentence = "3D ActionSLAM: wearable person tracking in multi-floor environments"
    # instruction = "Represent the Science title:"
    # embeddings = model.encode([[instruction,sentence]])
    # print(embeddings)
