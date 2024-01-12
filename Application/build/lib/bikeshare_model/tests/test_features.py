import sys
from pathlib import Path
file = Path(__file__).resolve()
parent, root = file.parent, file.parents[1]
sys.path.append(str(root))

import pandas as pd
from bikeshare_model.config.core import config
from bikeshare_model.tests.configtest import sample_input_data
from bikeshare_model.processing.features import WeathersitImputer

def test_weathersit_imputation(sample_input_data):
    # Given
    imputer = WeathersitImputer(config.model_config.weathersit_var)
    #print(sample_input_data['weathersit'].head(10))

    # When
    imputed = imputer.fit(sample_input_data).transform(sample_input_data)

    # Then
    #print(imputed['weathersit'].head(10))
    assert(imputed.loc[12230,'weathersit'] is not None)