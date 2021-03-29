import pandas as pd
import numpy as np

def clean_bo1_games(df, cards, rename_cols=dict(), drop_cols=set()):
    df = df.dropna()
    df['on_play'] = df['on_play'].astype(float)
    df['won'] = df['won'].astype(float)
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

def isolate_decks(df):
    df['lost'] = 1 - df['won']
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
