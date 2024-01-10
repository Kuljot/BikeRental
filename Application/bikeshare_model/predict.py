import sys
import pandas as pd
import numpy as np
from pathlib import Path
file = Path(__file__).resolve()
parent, root = file.parent, file.parents[1]
sys.path.append(str(root))
from typing import Union
from bikeshare_model import __version__ as _version
from bikeshare_model.config.core import config
from bikeshare_model.pipeline import bikeshare_pipe
from bikeshare_model.processing.data_manager import load_pipeline
from bikeshare_model.processing.data_manager import pre_pipeline_preparation
from bikeshare_model.processing.validation import validate_inputs

pipeline_file_name = f"{config.app_config.pipeline_save_file}{_version}.pkl"
bikeshare_pipe= load_pipeline(file_name=pipeline_file_name)

def make_prediction(*,input_data:Union[pd.DataFrame, dict]) -> dict:
    """Make a prediction using a saved model """
    validated_data, errors = validate_inputs(input_df=pd.DataFrame(input_data))
    validated_data = validated_data.reindex(columns=config.model_config.features)
    
    #results = {"predictions": None, "version": _version, "errors": errors}
    
    predictions = bikeshare_pipe.predict(validated_data)
    results = {"predictions": predictions,"version": _version, "errors": errors}

    return results

if __name__ == "__main__":

    data_in={'dteday':['2012-04-10'],'season':['summer'],'hr':['3am'],'holiday':['No'],'weekday':['Tue'],
                'workingday':['Yes'],'weathersit':['Clear'],'temp':[8.92],'atemp':[7.0010],'hum':[71.0],
                'windspeed':[8.9981],'yr':[2012], 'mnth':['April']}
    
    res = make_prediction(input_data=data_in)
    print(res)



# 	    dteday	    season	hr	holiday	weekday	workingday	weathersit	temp	atemp	hum	    windspeed	casual	registered	yr	    mnth
# 3439	2012-04-10	summer	3am	No	    Tue	    Yes	        Clear	    8.92	7.0010	71.0	8.9981	    0	    2	        2012	April