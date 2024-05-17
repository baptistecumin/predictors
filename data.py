from pydantic import BaseModel, field_validator
from typing import Iterable, Union, Sequence, Dict

class ClassifierClass(BaseModel):
    """A possible label to predict for a task.
    """
    name: str 
    description: str = ""

class TrainExample(BaseModel):
    input: str
    labels: Dict[str, Union[str, float, int]] 

class InputData(BaseModel):
    data: Sequence[Union[str, float, int]]

    @field_validator('data')
    def check_length(cls, value):
        assert len(value) > 0, "Input data must not be empty."
        return value    
