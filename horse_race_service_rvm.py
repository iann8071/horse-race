import numpy as np
from dao.input_dao import InputDao
from dao.output_dao import OutputDao
from transform.scale.boolean_scaler import BooleanScaler
from transform.transform import Transform
from validation.cross_validation import CrossValidation
from search.grid_search import GridSearch
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
            'PWIN',
            'PRENTAI',
            'PFUKUSYO',
            'TWIN',
            'TRENTAI',
            'TFUKUSYO',
            'UWIN',
            'URENTAI',
            'UFUKUSYO'
        ]
        self.answer = 'CONFTYAKU'
        self.valid_ranges = {

        }
        self.scale_types = {
            'CONFTYAKU': BooleanScaler('01')
        }
        self.hyper_parameter_values = {
            "dummy": {
                "from": -1,
                "to": -1,
                "unit": 1
            }
        }
        self.outputs = {
            "sale":"numeric(10,10)",
            "expense": "numeric(10,10)",
            "profit": "numeric(10,10)"
        }
        self.k_fold = 2

    def execute(self):
        pass

    def test(self):
        data = self.input_dao.read_data_as_pdf()
        self.output_dao.init_table(self.outputs)
        transformed_data = Transform(self.valid_ranges, self.scale_types).execute(data)
        Analytics.start(
            GridSearch(),
            Parallel(self.spark_app_name),
            RVM(),
            CrossValidation(self.k_fold),
            self.hyper_parameter_values,
            self.features,
            self.answer,
            transformed_data,
            self.write_score
        )

    @classmethod
    def write_score(cls, answer, expect, data, hyper_parameters, count):
        sale = np.sum((answer == 1) * (answer == expect) * data['TANODDS'])
        expense = len(np.where(answer == 1))
        field_names = list(hyper_parameters.keys())
        field_names.append('sale')
        field_names.append('expense')
        field_names.append('profit')
        row = hyper_parameters
        row['sale'] = sale
        row['expense'] = expense
        row['profit'] = sale - expense
        print('result=' + str(row))
        OutputDao().write_data(row)
        data = pd.DataFrame(data)
        data['answer'] = answer
        data['expect'] = expect
        data.to_csv(str(count) + "_" + "_".join(list(hyper_parameters.values())) + "_outputs.csv", index=False)

    def save_predictor(self):
        data = self.input_dao.read_data_as_pdf()
        self.output_dao.init_table(self.outputs)
        transformed_data = Transform(self.valid_ranges, self.scale_types).execute(data)
        Analytics.save_predictor(
            Parallel(self.spark_app_name),
            RVM(),
            self.features,
            self.answer,
            transformed_data
        )

if __name__ == '__main__':
    HorseRaceService().save_predictor()
