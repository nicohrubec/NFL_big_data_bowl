import numpy as np
import pandas as pd


def do_feature_engineering(df_defense, df_rusher):
    df_defense = get_speed_along_axis(df_defense)
    df_rusher = get_speed_along_axis(df_rusher)

    # (play1_rusher, play2_rusher) --> (play1_rusher, play1_rusher, play2_rusher, play2_rusher)
    # that way relative features in next step can be calculated in a vectorized way
    # requiring no for loop over plays
    cols = df_rusher.columns
    data = np.repeat(df_rusher.values, repeats=11, axis=0)
    df_rusher = pd.DataFrame(data=data, columns=cols)

    # build df with relative features between defense and rusher
    df = obtain_relative_features(df_defense, df_rusher)
    df = reshape_data(df)

    return df


# reshapes dataframe with features for n defense players so that
# we get a numpy array with the per player features with shape (n_samples, n_defense_players, n_features)
# also a meta data dataframe is returned containing GameId, PlayId, Season and the Yards per Play

def reshape_data(df):
    n_plays = len(df.PlayId.unique())
    n_players = 11

    play_features = ['GameId', 'PlayId', 'Season', 'Yards']
    player_features = [feature for feature in df.columns if feature not in play_features]
    meta_df = df[play_features].values
    df = df[player_features].values

    meta_df_reshaped = np.zeros((n_plays, len(play_features)))
    df_reshaped = np.zeros((n_plays, len(player_features) * n_players))

    for i in range(n_plays):
        meta_df_temp = meta_df[(i * n_players):((i + 1) * n_players)]
        df_temp = df[(i * n_players):((i + 1) * n_players)]

        meta_df_reshaped[i] = meta_df_temp[0]
        df_reshaped[i] = df_temp.flatten()

    meta_df = pd.DataFrame(data=meta_df_reshaped, columns=play_features)
    df = np.reshape(df_reshaped, newshape=(n_plays, n_players, len(player_features)))

    return meta_df, df


# replace Speed and Direction column with the speed in x and y direction for each player
def get_speed_along_axis(df):
    df['Sx'], df['Sy'] = df.S * np.cos(df.Dir), df.S * np.sin(df.Dir)
    df.drop(['Dir', 'S'], axis=1, inplace=True)

    return df


def obtain_relative_features(df_defense, df_rusher):
    df_defense.reset_index(inplace=True, drop=True)
    df_rusher.reset_index(inplace=True, drop=True)

    df = df_defense.copy()
    df['X'] = df_defense['X'] - df_rusher['X']
    df['Y'] = df_defense['Y'] - df_rusher['Y']
    df['Sxr'] = df_defense['Sx'] - df_rusher['Sx']
    df['Syr'] = df_defense['Sy'] - df_rusher['Sy']

    return df
