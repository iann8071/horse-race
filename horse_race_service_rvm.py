import numpy as np
from dao.input_dao import InputDao
from dao.output_dao import OutputDao
from transform.scale.boolean_scaler import BooleanScaler
from transform.transform import Transform
from validation.cross_validation import CrossValidation
from executor.parallel import Parallel
from learn.classification.rvm import RVM
from analytics import Analytics
import pandas as pd


class HorseRaceService:

    output_file = 'horse_race_result.csv'
    spark_app_name = 'Horse Race Analysis'

    def __init__(self):
        self.input_dao = InputDao()
        self.output_dao = OutputDao()
        self.features = [
            'DISTANCE',
            'WEATHERCD',
            'LAWNBABACD',
            'DARTBABACD',
            'BURDWEIGHT',
            'INCDECWEIGHT',
            'TANODDS',
            'FUKUMINODDS',
            'FUKUMAXODDS',
            'PREDTIME',
            'MINGDERPLUS',
            'MINGDERMINUS',
            'FOURHALONTIME',
            'FOURRAPTIME',
            'THREEHALONTIME',
            'THREERAPTIME',
            'TWOHALONTIME',
            'TWORAPTIME',
            'RAPTIME',
            'pwin',
            'prentai',
            'pfukusyo',
            'twin',
            'trentai',
            'tfukusyo',
            'uwin',
            'urentai',
            'ufukusyo'
        ]
        self.answer = 'CONFTYAKU'
        self.valid_ranges = {

        }
        self.scale_types = {
            'CONFTYAKU': BooleanScaler('01')
        }
        self.outputs = {
            "sale":"double",
            "expense": "double",
            "profit": "double"
        }
        self.k_fold = 2

    def execute(self):
        pass

    def test(self):
        data = self.input_dao.read_data_as_pdf()
        self.output_dao.init_table(self.outputs)
        transformed_data = Transform(self.valid_ranges, self.scale_types).execute(data)
        Analytics.start(
            Parallel(self.spark_app_name),
            CrossValidation(self.k_fold),
            RVM(),
            transformed_data,
            self.features,
            self.answer,
            self.write_score
        )

    @classmethod
    def write_score(cls, answer, expect, data, count):
        np.set_printoptions(threshold=np.inf)
        print(data)
        print(answer)
        print(expect)
        data = pd.DataFrame(data)
        data['answer'] = answer
        data['expect'] = expect
        data.to_csv(str(count) + "_outputs.csv", index=False)


if __name__ == '__main__':
    HorseRaceService().test()

