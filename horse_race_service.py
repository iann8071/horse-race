import numpy as np
from dao.input_dao import InputDao
from dao.output_dao import OutputDao
from transform.scale.boolean_scaler import BooleanScaler
from transform.transform import Transform
from utils.output_writer import OutputWriter
from validation.cross_validation import CrossValidation
from search.grid_search import GridSearch
from executor.parallel import Parallel
from learn.classification.svm import SVM
from analytics import Analytics

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
        self.hyper_parameter_values = {
            "c": {
                "from": 1 / 64,
                "to": 64,
                "unit": 2
            },
            "gamma": {
                "from": 1 / 64,
                "to": 64,
                "unit": 2
            }
        }
        self.outputs = {
            "c":"double",
            "gamma":"double",
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
            GridSearch(),
            Parallel(self.spark_app_name),
            CrossValidation(self.k_fold),
            SVM(),
            self.hyper_parameter_values,
            transformed_data,
            self.features,
            self.answer,
            self.write_score
        )

    @classmethod
    def write_score(cls, answer, expect, data, hyper_parameters):
        sale = np.sum((answer == 1) * (answer == expect) * data['TANODDS'])
        expense = len(np.where(answer == 1))
        field_names = list(hyper_parameters.keys())
        field_names.append('sale')
        field_names.append('expense')
        field_names.append('profit')
        row = hyper_parameters
        row['sale'] = str(sale)
        row['expense'] = str(expense)
        row['profit'] = str(sale - expense)
        print('result=' + str(row))
        OutputDao().write_data(row)

if __name__ == '__main__':
    HorseRaceService().test()

