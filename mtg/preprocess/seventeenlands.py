import pandas as pd
import numpy as np
import requests
from mtg.obj.cards import CardSet

def clean_bo1_games(df, cards, rename_cols=dict(), drop_cols=[]):
    df = df.dropna()
    df.columns = [c.lower() for c in df.columns]
    df.loc[:,'on_play'] = df['on_play'].astype(float)
    df.loc[:,'won'] = df['won'].astype(float)
    card_names = [x.split("_",1)[1].lower() for x in df.columns if x.startswith("deck_")]
    if isinstance(cards, pd.DataFrame):
        flip_cards = set([name.lower() for name in cards['name'].tolist()]).difference(set(card_names))
    else:
        flip_cards = set([x.name.lower() for x in cards.cards]).difference(set(card_names))
    #below drops empty columns, however they may be useful, so commenting that out. Instead will
    # drop columns that are for cards that don't satisfy the "is:booster" scryfall restriction that
    # is imposed on the `cards` object
    #drop_cols += df.columns[(df == 0).all()].tolist()
    # align flip cards and scryfall
    delimiter = " // "
    for fp in flip_cards:
        front,back = fp.split(delimiter)
        for column in df.columns:
            if column.lower().endswith(front):
                init_name = column[:-len(front)]
                rename_cols[column] = init_name + front + delimiter + back
            elif column.lower().endswith(back):
                drop_cols.append(column)
    df = df.rename(columns=rename_cols)
    # remove columns for cards that are not in the `cards` object
    prefixes = ['deck','opening_hand','drawn','sideboard']
    drop_cols += [c for c in df.columns if any([
            c.startswith(prefix) for prefix in prefixes
        ])  and (c.split('_')[-1] not in [name for name in cards['name'].tolist()])
    ]
    drop_cols = list(set(drop_cols))
    df = df.drop(drop_cols, axis=1)
    df.columns = [x.lower() for x in df.columns]
    return df

def get_card_rating_data(expansion, endpoint=None, start=None, end=None, colors=None):
    if endpoint is None:
        endpoint = f'https://www.17lands.com/card_ratings/data?expansion={expansion.upper()}&format=PremierDraft'
        if start is not None:
            endpoint += f'&start_date={start}'
        if end is not None:
            endpoint += f'&end_date={end}'
        if colors is not None:
            endpoint += f'&colors={colors}'
    card_json = requests.get(endpoint).json()
    card_df = pd.DataFrame(card_json).fillna(0.0)
    numerical_cols = card_df.columns[card_df.dtypes != object]
    card_df['name'] = card_df['name'].str.lower()
    card_df = card_df.set_index('name')
    return card_df[numerical_cols]

def add_archetypes(df, min_2c_basics=5, min_1c_basic=11):
    color_pairs = [
        'WU',
        'WB',
        'WR',
        'WG',
        'UB',
        'UR',
        'UG',
        'BR',
        'BG',
        'RG'
    ]
    def map_cp_to_lands(cp):
        result = []
        for color in cp:
            if color == "W":
                result.append("deck_plains")
            elif color == "U":
                result.append("deck_island")
            elif color == "B":
                result.append("deck_swamp")
            elif color == "R":
                result.append("deck_mountain")
            elif color == "G":
                result.append("deck_forest")
        return result

    for cp in color_pairs:
        col1, col2 = map_cp_to_lands(cp)
        where_cp = df[(df[col1] >= min_2c_basics) & (df[col2] >= min_2c_basics)].index
        df.loc[where_cp,'color_pair'] = cp
    mono_c = list('WUBRG')
    for c in mono_c:
        col = map_cp_to_lands(c)[0]
        where_cp = df[(df[col] >= min_1c_basic) & (df['color_pair'].isna())].index
        df.loc[where_cp,'color_pair'] = c
    df.loc[:,'color_pair'] = df['color_pair'].fillna('5c')
    return df

#depreciated
def isolate_decks(df):
    df.loc[:,'lost'] = 1 - df['won']
    losses = df.groupby("date")["lost"].sum()
    wins = df.groupby("date")["won"].sum()
    index = wins.index
    too_many_wins = index[np.where(wins > 7)]
    too_many_losses = index[np.where(losses > 3)]
    not_possible = index[np.where((wins == 7) & (losses == 3))]
    incomplete = index[np.where((wins < 7) & (losses < 3))]
    bad_dates = set(
        too_many_wins.tolist() +
        too_many_losses.tolist() + 
        not_possible.tolist() + 
        incomplete.tolist()
    )
    df = df[~df['date'].isin(bad_dates)]
    d = {
        column: 'last' for column in df.columns if column not in ["opp_colors","date"]
    }
    d.update({
            "won":"sum",
            "lost":"sum",
            "on_play":"mean",
            "num_mulligans":"mean",
            "opp_num_mulligans": "mean",
            "num_turns": "mean",
    })
    df = df.groupby('date').agg(d)
    return df
