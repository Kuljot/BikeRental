from typing import Any, List, Optional
from pydantic import BaseModel
#from bikesharing import bikeshare_model
from bikeshare_model.processing.validation import DataInputSchema


class PredictionResults(BaseModel):
    errors: Optional[Any]
    version: str
    #predictions: Optional[List[int]]
    predictions: Optional[int]


class MultipleDataInputs(BaseModel):
    inputs: List[DataInputSchema]

    class Config:
        schema_extra = {
            "example": {
                "inputs": [
                    {
                        'dteday':'2012-04-10',
                        'season':'summer',
                        'hr':'3am',
                        'holiday':'No',
                        'weekday':'Tue',
                        'workingday':'Yes',
                        'weathersit':'Clear',
                        'temp':8.92,
                        'atemp':7.0010,
                        'hum':71.0,
                        'windspeed':8.9981,
                        'yr':2012,
                        'mnth':'April'
                    }
                ]
            }
        }
