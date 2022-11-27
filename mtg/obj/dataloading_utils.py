import pandas as pd
import requests
import re


def load_data(filename, cards, name=None):
    if name == "draft":
        return load_draft_data(filename, cards)
    elif name == "bo1":
        return load_bo1_data(filename, cards)
    else:
        return pd.read_csv(filename)


def sort_cols_by_card_idxs(df, card_col_prefixes, cards):
    # initialize columns to start with the non-card columns
    column_order = [
        c
        for c in df.columns
        if not any([c.startswith(prefix) for prefix in card_col_prefixes])
    ]
    card_names = cards.sort_values(by="idx", ascending=True)["name"].tolist()
    for prefix in card_col_prefixes:
        prefix_columns = [prefix + "_" + name for name in card_names]
        column_order += prefix_columns
    # reorder dataframe to abide by new column ordering
    #   this is just so df[self.deck_cols].to_numpy()
    #   yields a comparable matrix to df[self.sideboard_cols].to_numpy()
    df = df[column_order]
    return df


def load_bo1_data(filename, cards):
    COLUMN_REGEXES = {
        re.compile(r"user_game_win_rate_bucket"): "float16",
        re.compile(r"rank"): "str",
        re.compile(r"draft_id"): "str",
        re.compile(r"draft_time"): "str",
        re.compile(r"expansion"): "str",
        re.compile(r"event_type"): "str",
        re.compile(r"deck_.*"): "int8",
        re.compile(r"sideboard_.*"): "int8",
        re.compile(r"drawn_.*"): "int8",
        re.compile(r"sideboard_.*"): "int8",
        re.compile(r"opening_hand_.*"): "int8",
        re.compile(r"on_play"): "int8",
        re.compile(r"won"): "int8",
        re.compile(r"num_turns"): "int8",
        re.compile(r"num_mulligans"): "int8",
        re.compile(r"opp_num_mulligans"): "int8",
    }
    col_names = pd.read_csv(filename, nrows=0).columns
    data_types = {}
    draft_cols = []
    for c in col_names:
        if any(
            [
                c.startswith(prefix)
                for prefix in ["sideboard_", "deck_", "drawn_", "opening_hand_"]
            ]
        ):
            draft_cols.append(c)
        for (r, t) in COLUMN_REGEXES.items():
            if r.match(c):
                data_types[c] = t

    df = pd.read_csv(
        filename,
        dtype=data_types,
        usecols=[
            "draft_id",
            "draft_time",
            "won",
            "user_game_win_rate_bucket",
            "rank",
            "on_play",
            "num_turns",
            "num_mulligans",
            "opp_num_mulligans"
            # ...
        ]
        + draft_cols,
    )
    rename_cols = {
        "user_game_win_rate_bucket": "user_win_rate_bucket",
        "draft_time": "date",
    }
    df.columns = [
        x.lower() if x not in rename_cols else rename_cols[x] for x in df.columns
    ]
    df["won"] = df["won"].astype(float)
    df["date"] = pd.to_datetime(df["date"])
    card_col_prefixes = ["deck", "opening_hand", "drawn", "sideboard"]
    df = sort_cols_by_card_idxs(df, card_col_prefixes, cards)
    return df


def load_draft_data(filename, cards):
    COLUMN_REGEXES = {
        re.compile(r"user_game_win_rate_bucket"): "float16",
        re.compile(r"user_n_games_bucket"): "int8",
        re.compile(r"rank"): "str",
        re.compile(r"draft_id"): "str",
        re.compile(r"draft_time"): "str",
        re.compile(r"expansion"): "str",
        re.compile(r"event_type"): "str",
        re.compile(r"event_match_wins"): "int8",
        re.compile(r"event_match_losses"): "int8",
        re.compile(r"pack_number"): "int8",
        re.compile(r"pick_number"): "int8",
        re.compile(r"pick$"): "str",
        re.compile(r"pick_maindeck_rate"): "float16",
        re.compile(r"pick_sideboard_in_rate"): "float16",
        re.compile(r"pool_.*"): "int8",
        re.compile(r"pack_card_.*"): "int8",
    }
    col_names = pd.read_csv(filename, nrows=0).columns
    data_types = {}
    draft_cols = []
    for c in col_names:
        if c.startswith("pack_card_"):
            draft_cols.append(c)
        elif c == "pick":
            draft_cols.append(c)
        elif c.startswith("pool_"):
            draft_cols.append(c)
        for (r, t) in COLUMN_REGEXES.items():
            if r.match(c):
                data_types[c] = t

    df = pd.read_csv(
        filename,
        dtype=data_types,
        usecols=[
            "draft_id",
            "draft_time",
            "event_match_losses",
            "event_match_wins",
            "pack_number",
            "pick_number",
            "user_n_games_bucket",
            "user_game_win_rate_bucket",
            "rank"
            # ...
        ]
        + draft_cols,
    )
    rename_cols = {
        "user_game_win_rate_bucket": "user_win_rate_bucket",
        "draft_time": "date",
    }
    df.columns = [
        x.lower() if x not in rename_cols else rename_cols[x] for x in df.columns
    ]
    n_picks = df.groupby("draft_id")["pick"].count()
    t = n_picks.max()
    bad_draft_ids = n_picks[n_picks < t].index.tolist()
    df = df[~df["draft_id"].isin(bad_draft_ids)]
    df["pick"] = df["pick"].str.lower()
    df["date"] = pd.to_datetime(df["date"])
    df["won"] = (
        df["event_match_wins"] / (df["event_match_wins"] + df["event_match_losses"])
    ).fillna(0.0)
    card_col_prefixes = ["pack_card", "pool"]
    df = sort_cols_by_card_idxs(df, card_col_prefixes, cards)
    df["position"] = (
        df["pack_number"] * (df["pick_number"].max() + 1) + df["pick_number"]
    )
    df = df.sort_values(by=["draft_id", "position"])
    return df


def get_card_rating_data(expansion, endpoint=None, start=None, end=None, colors=None):
    if endpoint is None:
        endpoint = f"https://www.17lands.com/card_ratings/data?expansion={expansion.upper()}&format=PremierDraft"
        if start is not None:
            endpoint += f"&start_date={start}"
        if end is not None:
            endpoint += f"&end_date={end}"
        if colors is not None:
            endpoint += f"&colors={colors}"
    card_json = requests.get(endpoint).json()
    card_df = pd.DataFrame(card_json).fillna(0.0)
    numerical_cols = card_df.columns[card_df.dtypes != object]
    card_df["name"] = card_df["name"].str.lower()
    card_df = card_df.set_index("name")
    return card_df[numerical_cols]


def get_draft_json(draft_log_url, stream=False):
    if not stream:
        base_url = "https://www.17lands.com/data/draft?draft_id="
    else:
        base_url = "https://www.17lands.com/data/draft/stream/?draft_id="
    draft_ext = draft_log_url.split("/")[-1].strip()
    log_json_url = base_url + draft_ext
    response = requests.get(log_json_url, stream=stream)
    if not stream:
        response = response.json()
    return response
