from src import features
from src import preprocessing
from src import training

df_defense, df_rusher = preprocessing.do_preprocessing()
meta_df, feature_df = features.do_feature_engineering(df_defense, df_rusher)
training.train_KFolds(meta_df, feature_df, debug=True)
