import math

import numpy as np
import pandas as pd

from src import configs


def do_preprocessing():
    used_feats = ['GameId', 'PlayId', 'X', 'Y', 'S', 'Dir', 'Season', 'Yards']
    df = pd.read_csv(configs.train_raw)

    df = clean_data(df)  # data cleanage and preprocessing
    df = df[df.Season != 2019]  # exclude 2019 data for testing purposes

    # build df containing only defense players and one containing only the rushers for a given play
    # filter out columns that will be needed for features and CV setup + target
    df_defense = df[(df.IsOnOffense == 0)][used_feats]
    df_rusher = df[(df.IsBallCarrier == 1)][used_feats]

    return df_defense, df_rusher


def clean_data(df):
    # clear abbreviations so that one team corresponds to the same abbreviation in different columns
    df = clean_abbreviations(df)
    # normalize play direction --> rusher always from left to right
    df = standardize_direction(df)
    # small trick so that S is linearly related to dis
    df['S'] = 10 * df['Dis']

    df.fillna(0, inplace=True)

    return df


# just cleaning some abbreviation so that they are
# in sync with the abbreviations used in the PossessionTeam column
def clean_abbreviations(df):
    df.loc[df.VisitorTeamAbbr == "ARI", 'VisitorTeamAbbr'] = "ARZ"
    df.loc[df.HomeTeamAbbr == "ARI", 'HomeTeamAbbr'] = "ARZ"

    df.loc[df.VisitorTeamAbbr == "BAL", 'VisitorTeamAbbr'] = "BLT"
    df.loc[df.HomeTeamAbbr == "BAL", 'HomeTeamAbbr'] = "BLT"

    df.loc[df.VisitorTeamAbbr == "CLE", 'VisitorTeamAbbr'] = "CLV"
    df.loc[df.HomeTeamAbbr == "CLE", 'HomeTeamAbbr'] = "CLV"

    df.loc[df.VisitorTeamAbbr == "HOU", 'VisitorTeamAbbr'] = "HST"
    df.loc[df.HomeTeamAbbr == "HOU", 'HomeTeamAbbr'] = "HST"

    return df


def standardize_direction(df):
    df['Dir_rad'] = np.mod(90 - df.Dir, 360) * math.pi / 180.0
    df['ToLeft'] = df.PlayDirection == "left"
    df['IsBallCarrier'] = df.NflId == df.NflIdRusher

    df['TeamOnOffense'] = "home"
    df.loc[df.PossessionTeam != df.HomeTeamAbbr, 'TeamOnOffense'] = "away"
    df['IsOnOffense'] = df.Team == df.TeamOnOffense  # Is player on offense?
    df['YardLine_std'] = 100 - df.YardLine
    df.loc[df.FieldPosition.fillna('') == df.PossessionTeam, 'YardLine_std'] = \
        df.loc[df.FieldPosition.fillna('') == df.PossessionTeam, 'YardLine']
    df['X_std'] = df.X
    df.loc[df.ToLeft, 'X'] = 120 - df.loc[df.ToLeft, 'X']
    df['Y_std'] = df.Y
    df.loc[df.ToLeft, 'Y'] = 160 / 3 - df.loc[df.ToLeft, 'Y']
    df['Dir'] = df.Dir_rad
    df.loc[df.ToLeft, 'Dir'] = np.mod(np.pi + df.loc[df.ToLeft, 'Dir_rad'], 2 * np.pi)

    return df


def prepare_targets(targets):
    y = np.zeros((len(targets), 199))
    for idx, target in enumerate(list(targets)):
        y[idx][99 + int(target)] = 1

    return y
