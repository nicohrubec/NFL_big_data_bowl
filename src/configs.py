from pathlib import Path

project_folder = Path.cwd().parent
model_folder = project_folder / 'models'
data_folder = project_folder / 'data'

train_raw = data_folder / 'train.csv'
