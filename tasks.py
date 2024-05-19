from pydantic import BaseModel, Field
from pydantic.functional_validators import field_validator
from typing import Union, Sequence, Dict, Optional, List
from jinja2 import Template

class ClassifierClass(BaseModel):
    """A possible label to predict for a task.
    """
    name: str 
    description: str = ""

    def dict(self):
        return {
            "name": self.name,
            "description": self.description
        }

class TrainExample(BaseModel):
    input: str
    labels: Dict[str, Union[str, float, int]] 

class InputData(BaseModel):
    data: Sequence[Union[str, float, int]]

    @field_validator('data')
    def check_length(cls, value):
        assert len(value) > 0, "Input data must not be empty."
        return value    

class Classify(BaseModel):
    name: str = Field(..., description="The name of the classification task.")
    description: str = Field(..., description="A brief description of the classification task.")
    classes: Optional[List[ClassifierClass]] = Field([], description="A list of possible classification classes.")
    chain_of_thought: Optional[bool] = Field(False, description="Whether to use chain of thought reasoning.")

    @field_validator('classes')
    def check_class_names_unique(cls, value):
        names = [classifier_class.name for classifier_class in value]
        assert len(names) == len(set(names)), f"Class names must be unique, {names} names contains duplicates"
        return value

    def set_classes(self, classes: Union[List[str], List[ClassifierClass]]) -> None:
        if type(classes[0]) == str:
            self.classes = [ClassifierClass(name=v, description="") for v in classes]
        else:
            self.classes = classes

class Predict(BaseModel):
    name: str = Field(..., description="The name of the prediction task.")
    description: str = Field(..., description="A brief description of the prediction task.")
    dtype: Optional[str] = Field(None, description="The data type of the prediction output.")
    chain_of_thought: Optional[bool] = Field(False, description="Whether to use chain of thought reasoning.")

class PromptTemplate(BaseModel):
    template: Template = Field(..., description="The Jinja2 template used for creating prompts.")
    class Config:
        arbitrary_types_allowed = True
