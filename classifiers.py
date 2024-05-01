from abc import ABC, abstractmethod
from pydantic import BaseModel, Field, field_validator, create_model
from typing import Any, List, Optional, Type, Literal
from data import InputData, TrainExample, ClassifierClass

import instructor
from litellm import completion
import logger
import dotenv
dotenv.load_dotenv()

from jinja2 import Environment, FileSystemLoader
template_dir = './prompts'
env = Environment(loader=FileSystemLoader(template_dir))

logger = logger.get_logger()

class ClassifierTask(BaseModel):
    name: str
    description: str
    classes: Optional[List[ClassifierClass]] = None
    dtype: Optional[str] = None
    
    @field_validator('classes')
    def check_class_names_unique(cls, value):
        names = [classifier_class.name for classifier_class in value]
        assert len(names) == len(set(names)), f"Class names must be unique, {names} names contains duplicates"
        return value
    
class ClassifierConfig(BaseModel):
    model: str
    tasks: List[ClassifierTask]
    
    prompt: Optional[str] = None
    response_model: Optional[Type[BaseModel]] = None
    
    def model_post_init(self, __context):
        self.response_model = self.create_response_model()
        self.prompt = self.response_model.__doc__
    
    def create_response_model(self) -> Type[BaseModel]:
        """
        Dynamically creates a response model based on the classifier tasks and allowed classes.
        """
        prediction_object_fields = {}
        for task in self.tasks:
            if task.classes is None:
                prediction_object_fields[task.name] = (task.dtype, Field(..., description=task.description))
                continue
            else:
                class_labels = [n.name for n in task.classes]
                class_labels_type = Literal[tuple(class_labels)] # type: ignore
                prediction_object_fields[task.name] = (class_labels_type, Field(..., description=task.description))
        template = env.get_template('classification_task.jinja')
        model = create_model(
            __model_name='Labels', 
            __doc__=template.render(classifier_tasks=self.tasks),
            **prediction_object_fields
            )
        return model

class FewShotTeacherConfig(ClassifierConfig):
    teacher_model: str

class BaseClassifier(ABC, BaseModel):
    config: ClassifierConfig

    def __init__(self, **data: Any):
        if 'config' not in data: # can be passed up by subclasses.
            data['config'] = ClassifierConfig(**data)
        super().__init__(**data)  

    @abstractmethod
    def fit(self, X, y):
        """Fit the classifier model."""
        pass

    @abstractmethod
    def predict(self, X) -> List[int]:
        """Predict using the classifier model."""
        pass

class ZeroShotClassifier(BaseClassifier):
    def fit(self, X) -> List[int]:
        return self.predict(X)

    def predict(self, X) -> List[int]:
        X = InputData(data=X)
        client = instructor.from_litellm(completion)
        # import cohere
        # client = instructor.from_cohere(cohere.Client())
        outputs = []
        for x in X.data:
            output = client.chat.completions.create(
                model=self.config.model,
                max_tokens=1024,
                messages=[
                    {
                        "role": "user",
                        "content": self.config.prompt + "\n\n object: " + x,
                    }
                ],
                response_model=self.config.response_model,
            ).dict()
            outputs.append([output[task.name] for task in self.config.tasks])
        # Simplify output: if each item only has one class, flatten the list
        if len(self.config.tasks) == 1:
            return [output[0] for output in outputs]
        return outputs


class FewShotTeacherClassifier(BaseClassifier):
    config: FewShotTeacherConfig

    def __init__(self, **kwargs: Any):
        kwargs['config'] = FewShotTeacherConfig(**kwargs)
        super().__init__(**kwargs)

    def _update_prompt(self, examples: List[TrainExample]):
        few_shot_template = env.get_template('classification_task_few_shot.jinja')
        few_shot_examples = [{'input': ex.input, 'label': str(ex.labels)} for ex in examples]
        self.config.prompt = few_shot_template.render(
            classifier_tasks=self.config.tasks,
            few_shot_examples=few_shot_examples
        )
        return self.config.prompt

    def fit(self, X) -> List[TrainExample]:
        logger.debug(f"Fitting model using model {self.config.teacher_model}. \nUsing prompt: \n{self.config.prompt}")
        client = instructor.from_litellm(completion)
        outputs = []
        
        X = InputData(data=X)
        for x in X.data:
            output = client.chat.completions.create(
                model=self.config.teacher_model,
                max_tokens=1024,
                messages=[
                    {
                        "role": "user",
                        "content": f"{self.config.prompt} \n\n object: {x}",
                    }
                ],
                response_model=self.config.response_model,
            )
            outputs.append(TrainExample(input=x,
                                        labels=output.dict()))
        new_prompt = self._update_prompt(outputs)
        logger.debug(f"Completed few shot fitting. New prompt: \n{new_prompt}")
        return new_prompt

    def predict(self, X):
        logger.debug("Predicting...")
        client = instructor.from_litellm(completion)
        outputs = []
        for x in X:
            output = client.chat.completions.create(
                model=self.config.model,
                max_tokens=1024,
                messages=[
                    {
                        "role": "user",
                        "content": self.config.prompt + "\n\n object: " + x,
                    }
                ],
                response_model=self.config.response_model,
            ).dict()
            outputs.append([output[task.name] for task in self.config.tasks])
        # Simplify output: if each item only has one class, flatten the list
        if len(self.config.tasks) == 1:
            return [output[0] for output in outputs]
        return outputs

class FineTunedClassifier(BaseClassifier):
    pass

class FineTunedFewShotClassifier(BaseClassifier):
    pass

