import os
from typing import List

import lightgbm as lgb
import numpy as np
import pandas as pd


class FlatRegressor:
    def __init__(self, model: lgb.Booster) -> None:
        self._model = model

    @classmethod
    def load(cls, model_folder: str) -> "FlatRegressor":
        model = lgb.Booster(model_file=os.path.join(model_folder, "model.pkl"))
        return cls(model=model)

    def feature_name(self):
        return self._model.feature_name()

    def predict(self, samples: List[str]) -> List[int]:
        categorical_features = [
            'wall_type_uk',
            'floor',
            'city_name_uk',
            'floors_count',
            'total_square_meters',
            'state_name_uk',
            'has_metro',
            'is_newbuild',
            'З_меблями',
            'З_парковкою',
            'З_ремонтом',
            'Кухня-студія',
            'Можна_з_тваринами',
            'Поруч_з_метро',
            'Поруч_з_парком',
            'Поруч_зі_школою',
            'Поруч_із_дитячим_садком',
            'rooms_count_size',
            'last_floor',
            'first_floor'
        ]

        df = pd.DataFrame(samples)
        df = df[self._model.feature_name()]
        df[categorical_features] = df[categorical_features].astype('category')
        predictions = self._model.predict(df)
        return list(np.expm1(predictions))
