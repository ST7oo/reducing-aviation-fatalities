import pandas as pd
import json

from trainer import util

_, _, eval_x, eval_y = util.load_data()

prediction_input = eval_x.sample(2000)
prediction_targets = eval_y[prediction_input.index]

_, eval_file_path = util.download(util.DATA_DIR)
raw_eval_data = pd.read_csv(eval_file_path, names=util._CSV_COLUMNS, na_values='?')

with open('prediction_input.json', 'w') as json_file:
    for row in prediction_input.values.tolist():
        json.dump(row, json_file)
        json_file.write('\n')

