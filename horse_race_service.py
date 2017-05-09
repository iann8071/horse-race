import numpy as np
from dao.input_dao import InputDao
from transform.scale.boolean_scaler import BooleanScaler
from transform.transform import Transform
from utils.output_writer import OutputWriter
from validation.cross_validation import CrossValidation
from search.grid_search import GridSearch
from executor.parallel import Parallel


class HorseRaceService:

    output_file = 'horse_race_result.csv'
    spark_app_name = 'Horse Race Analysis'

    def __init__(self, cross_validation_config):
        self.dao = InputDao()
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
        self.cross_validation_config = cross_validation_config

    def execute(self):
        pass

    def test(self, k_fold):
        data = self.dao.read_data_as_pdf()
        transformed_data = Transform(self.valid_ranges, self.scale_types).execute(data)
        GridSearch.svm(CrossValidation(k_fold), self.features, self.answer, transformed_data, self.write_score, Parallel(self.spark_app_name))

    @classmethod
    def write_score(cls, answer, expect, data, hyper_parameters):
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
        OutputWriter.dict_to_csv(cls.output_file, field_names, row)

if __name__ == '__main__':
    HorseRaceService(None).test(2)