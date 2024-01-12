import sys
from pathlib import Path
file = Path(__file__).resolve()
parent, root = file.parent, file.parents[1]
sys.path.append(str(root))

import pytest
from sklearn.model_selection import train_test_split
from bikeshare_model.config.core import config
from bikeshare_model.processing.data_manager import _load_raw_dataset
from bikeshare_model.processing.features import WeekdayImputer

@pytest.fixture
def sample_input_data():
    data = _load_raw_dataset(file_name=config.app_config.training_data_file)

    X_train, X_test, y_train, y_test = train_test_split(
        data,
        data[config.model_config.target],
        test_size=config.model_config.test_size,
        random_state=config.model_config.random_state,
    )

    return X_test