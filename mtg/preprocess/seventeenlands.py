import pandas as pd
import numpy as np
import requests

def clean_bo1_games(df, cards, rename_cols=dict(), drop_cols=set()):
    df = df.dropna()
    df.loc[:,'on_play'] = df['on_play'].astype(float)
    df.loc[:,'won'] = df['won'].astype(float)
    card_names = [x.split("_",1)[1].lower() for x in df.columns if x.startswith("deck_")]
    flip_cards = set([x.name.lower() for x in cards.cards]).difference(set(card_names))
    drop_cols += df.columns[(df == 0).all()].tolist()
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
    df = df.drop(drop_cols, axis=1).rename(columns=rename_cols)
    df.columns = [x.lower() for x in df.columns]
    return df

def get_card_rating_data(full_set, endpoint=None, join=False):
    if endpoint is None:
        endpoint = "https://www.17lands.com/card_ratings/data?expansion=KHM&format=PremierDraft&start_date=2021-01-01&end_date=2021-02-19"
    card_json = requests.get(endpoint).json()
    card_df = pd.DataFrame(card_json)
    flip_cards = {x.name.split("//")[0].strip():x.name.split("//")[1].strip() for x in full_set.cards if "//" in x.name}

    def change_flip_name(name):
        if name in flip_cards:
            return name + " // " + flip_cards[name]
        else:
            return name

    card_df.loc[:,'name'] = card_df['name'].str.lower().apply(change_flip_name)
    scry = full_set.to_dataframe()
    if join:
        card_df = card_df.join(scry, how="left", rsuffix="__extra")
        extras = [col for col in card_df.columns if col.endsith("__extra")]
        card_df = card_df.drop(extras)
    return card_df

def add_archetypes(df, cardset, min_2c_basics=5, min_1c_basic=11):
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
