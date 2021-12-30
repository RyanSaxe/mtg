import tensorflow as tf
import requests
import numpy as np
import pandas as pd
from matplotlib import colors
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.pyplot as plt
import warnings
import re


def print_deck(deck, cards, sort_by="name", return_str=False):
    cards = cards.sort_values(by=sort_by)
    output = ""
    for card_idx, card_name in cards[["idx", "name"]].to_numpy():
        count = deck[card_idx]
        if count > 0:
            print(count, card_name)
            if return_str:
                output += str(count) + " " + card_name + "\n"
    if return_str:
        return output


def print_counts(deck, cards, col="type_line"):
    col_counts = dict()
    cards = cards.set_index("idx")
    for card_idx, card_count in enumerate(deck):
        if card_count > 0:
            val = cards.loc[card_idx, col]
            if val in col_counts:
                col_counts[val] += card_count
            else:
                col_counts[val] = card_count
    for key, value in col_counts.items():
        print(key, ":", value)


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


def display_deck(pool, basics, spells, cards, return_url=False):
    pool = np.squeeze(pool)
    basics = np.squeeze(basics)
    spells = np.squeeze(spells)
    deck = np.concatenate([basics, spells])
    idx_to_name = cards.set_index("idx")["name"].to_dict()
    sb_text = "SIDEBOARD\n\n"
    deck_text = "DECK\n\n"
    deck_json = {"sideboard": [], "deck": []}
    for idx, count in enumerate(deck):
        name = idx_to_name[idx]
        if idx >= 5:
            sb_count = pool[idx - 5] - count
        else:
            sb_count = 0
        if sb_count > 0:
            sb_text += str(int(sb_count)) + " " + name + "\n"
            deck_json["sideboard"].append({"name": name, "count": int(sb_count)})
        if count == 0:
            continue
        deck_text += str(int(count)) + " " + name + "\n"
        deck_json["deck"].append({"name": name, "count": int(count)})
    if return_url:
        r = requests.post(url="https://www.sealeddeck.tech/api/pools", json=deck_json)
        r_js = r.json()
        output = r_js["url"]
    else:
        output = deck_text + "\n" + sb_text
    return output


def list_to_names(cards_json):
    if len(cards_json) > 0:
        return [x["name"].lower().split("//")[0].strip() for x in cards_json]
    else:
        return None


def names_to_array(names, mapping):
    names = list_to_names(names)
    idxs = [mapping[name] for name in names]
    unique, counts = np.unique(idxs, return_counts=True)
    arr = np.zeros(len(mapping))
    arr[unique] += counts
    return arr


def load_arena_ids(expansion):
    arena_id_file = "/content/drive/My Drive/mtg_data/card_list.csv"
    id_df = pd.read_csv(arena_id_file)
    id_df = id_df[(id_df["expansion"] == expansion) & (id_df["is_booster"])]
    id_df["name"] = id_df["name"].str.lower()
    return id_df.set_index("name")["id"].to_dict()


def names_to_arena_ids(names, expansion="VOW", mapping=None, return_mapping=False):
    if mapping is None:
        mapping = load_arena_ids(expansion)
    if not isinstance(names, list):
        names = [names]
    output = [mapping[x["name"].lower().split("//")[0].strip()] for x in names]
    if return_mapping:
        output = (output, mapping)
    return output


def read_mtgo(fname, name_to_idx=None, model=None, t=42):
    """
    process MTGO log file and convert it into tensors so the bot
    can say what it would do
    """
    ignore_cards = ["plains", "island", "swamp", "mountain", "forest"]
    with open(fname, "r") as f:
        lines = f.readlines()
    n_cards = len(name_to_idx.keys())
    packs = np.ones((t, n_cards), dtype=np.float32)
    picks = np.ones(t, dtype=np.int32) * n_cards
    pools = np.ones((t, n_cards), dtype=np.float32)
    positions = np.arange(t, dtype=np.int32)
    in_pack = False
    cur_pack = np.zeros(n_cards)
    cur_pick = np.zeros(n_cards)
    pool = np.zeros(n_cards)
    idx = 0
    for line in lines:
        match = re.findall(r"Pack \d pick \d+", line)
        if len(match) == 1:
            in_pack = True
            continue
        if in_pack:
            if len(line.strip()) == 0:
                in_pack = False
                if sum(cur_pick) != 0:
                    pick_idx = np.where(cur_pick != 0)[0]
                    assert len(pick_idx) == 1
                    packs[idx, :] = cur_pack
                    if idx + 1 < 41:
                        picks[idx + 1] = pick_idx[0]
                    pools[idx, :] = pool.copy()
                    pool[pick_idx[0]] += 1
                    idx += 1
                cur_pack = np.zeros(n_cards)
                cur_pick = np.zeros(n_cards)
                continue
            process = line.strip()
            if process.startswith("-"):
                cardname = process.split(" ", 1)[1].split("//")[0].strip().lower()
                if cardname in ignore_cards:
                    continue
                card_idx = name_to_idx[cardname]
                cur_pick[card_idx] = 1
            else:
                cardname = process.split("//")[0].strip().lower()
                if cardname in ignore_cards:
                    continue
                card_idx = name_to_idx[cardname]
            cur_pack[card_idx] = 1
    if idx < 42:
        packs[idx, :] = cur_pack
        pools[idx, :] = pool.copy()
    else:
        idx -= 1
    # draft_info = np.concatenate([packs, pools], axis=-1)
    model_input = (
        np.expand_dims(packs, 0),
        np.expand_dims(picks, 0),
        np.expand_dims(positions, 0),
    )
    if model is not None:
        predictions, att = model(model_input, training=False, return_attention=True)
        top3 = tf.math.top_k(predictions, k=3).indices.numpy()[0]
        idx_to_name = {v: k for k, v in name_to_idx.items()}
        print(
            "pick:",
            idx_to_name[top3[idx][0]],
            "(",
            predictions[0, idx, top3[idx][0]].numpy().round(3),
            ")",
            "--- followed by",
            idx_to_name[top3[idx][1]],
            "(",
            predictions[0, idx, top3[idx][1]].numpy().round(3),
            ")",
            "and",
            idx_to_name[top3[idx][2]],
            "(",
            predictions[0, idx, top3[idx][2]].numpy().round(3),
            ")",
        )
        hold = input()
        if hold == "stop":
            return model_input
        if idx < 41:
            return read_mtgo(fname, name_to_idx=name_to_idx, model=model, t=t)
    return model_input


# test
def draft_sim(
    expansion, model, t=None, idx_to_name=None, token="", build_model=None, cards=None
):
    name_to_idx = {v: k for k, v in idx_to_name.items()}
    seats = 8
    n_packs = 3
    n_cards = len(idx_to_name)
    n_picks = t // n_packs

    js = {
        idx: {"expansion": "VOW", "token": f"{token}", "picks": []}
        for idx in range(seats)
    }

    arena_mapping = load_arena_ids(expansion.expansion.upper())
    idx_to_js = {i: arena_mapping[idx_to_name[i]] for i in range(n_cards)}

    # index circular shuffle per iteration
    pack_shuffle_right = [7, 0, 1, 2, 3, 4, 5, 6]
    pack_shuffle_left = [1, 2, 3, 4, 5, 6, 7, 0]
    # initialize
    pick_data = np.ones((seats, t), dtype=np.int32) * n_cards
    pack_data = np.ones((seats, t, n_cards), dtype=np.float32)
    pool_data = np.ones((seats, t, n_cards), dtype=np.float32)
    final_pools = np.zeros((seats, n_cards), dtype=np.float32)
    positions = np.tile(np.arange(t, dtype=np.int32), [seats, 1])
    cur_pos = 0
    for pack_number in range(n_packs):
        # generate packs for this round
        packs = [
            expansion.generate_pack(name_to_idx=name_to_idx) for pack in range(seats)
        ]
        for pick_number in range(n_picks):
            pack_data[:, cur_pos, :] = np.vstack(packs)
            # draft_info = np.concatenate([pack_data, pool_data], axis=-1)
            for idx in range(seats):
                # model doesnt get serialized with 8 seats as an option so
                # we have to do it individually --- will ensure serialization
                # in the future
                data = (pack_data[[idx]], pick_data[[idx]], positions[[idx]])
                # make pick
                predictions, _ = model(data, training=False, return_attention=True)
                bot_pick = tf.math.argmax(predictions[0, cur_pos]).numpy()
                final_pools[idx][bot_pick] += 1
                if cur_pos + 1 < t:
                    pick_data[idx][cur_pos + 1] = bot_pick
                    pool_data[idx][cur_pos + 1][bot_pick] += 1
                pick_js = {
                    "pack_number": pack_number,
                    "pick_number": pick_number,
                    "pack_cards": [idx_to_js[x] for x in np.where(packs[idx] == 1)[0]],
                    "pick": idx_to_js[bot_pick],
                }
                js[idx]["picks"].append(pick_js)
                # the bot picked the card, so remove it from the pack for the next person
                packs[idx][bot_pick] = 0
            # pass the packs (left, right, left)
            if pack_number % 2 == 1:
                packs = [packs[idx] for idx in pack_shuffle_right]
            else:
                packs = [packs[idx] for idx in pack_shuffle_left]
            cur_pos += 1
    draft_logs = []
    for idx in range(seats):
        if build_model is not None:
            pool = np.expand_dims(final_pools[idx], 0)
            basics, spells, n_basics = build_model(pool, training=False)
            basics, spells = build_decks(basics, spells, n_basics, cards=cards)
            deck_url = display_deck(pool, basics, spells, cards, return_url=True)
        else:
            deck_url is None
        r = requests.post(url="https://www.17lands.com/api/submit_draft", json=js[idx])
        r_js = r.json()
        draft_id = r_js["id"]
        output = f"https://www.17lands.com/submitted_draft/{draft_id}"
        if deck_url is not None:
            output = (output, deck_url)
        draft_logs.append(output)
    return draft_logs


def draft_log_ai(
    draft_log_url,
    model,
    t=None,
    n_cards=None,
    idx_to_name=None,
    return_attention=False,
    return_style="df",
    batch_size=1,
    exchange_picks=-1,
    exchange_packs=-1,
    return_model_input=False,
    token="",
    build_model=None,
    cards=None,
    verbose=False,
):
    exchange_picks = (
        [exchange_picks] if isinstance(exchange_picks, int) else exchange_picks
    )
    exchange_packs = (
        [exchange_packs] if isinstance(exchange_packs, int) else exchange_packs
    )
    name_to_idx = {v: k for k, v in idx_to_name.items()}
    picks = get_draft_json(draft_log_url)["picks"]
    n_picks_per_pack = t / 3
    n_cards = len(name_to_idx)
    pool = np.zeros(n_cards, dtype=np.float32)
    draft_info = np.zeros((batch_size, t, n_cards * 2))
    positions = np.tile(
        np.expand_dims(np.arange(t, dtype=np.int32), 0), batch_size
    ).reshape(batch_size, t)
    actual_pick = []
    position_to_pxpy = dict()
    js = {"expansion": "VOW", "token": f"{token}", "picks": []}
    arena_id_mapping = None
    for pick in picks:
        arena_ids_in_pack, arena_id_mapping = names_to_arena_ids(
            pick["available"], mapping=arena_id_mapping, return_mapping=True
        )
        if pick["pick_number"] in exchange_picks:
            exchange = True
        else:
            exchange = False
        position = int(pick["pack_number"] * n_picks_per_pack + pick["pick_number"])
        if exchange and pick["pack_number"] in exchange_packs:
            correct_pick_options = [
                x["name"].lower().split("//")[0].strip()
                for x in pick["available"]
                if x["name"] != pick["pick"]["name"]
            ]
            correct_pick = np.random.choice(correct_pick_options)
            position_to_pxpy[position] = (
                "P" + str(pick["pack_number"] + 1) + "P*" + str(pick["pick_number"] + 1)
            )
        else:
            correct_pick = pick["pick"]["name"].lower().split("//")[0].strip()
            position_to_pxpy[position] = (
                "P" + str(pick["pack_number"] + 1) + "P" + str(pick["pick_number"] + 1)
            )
        pick_idx = name_to_idx[correct_pick]
        pack = names_to_array(pick["available"], name_to_idx)
        draft_info[0, position, :n_cards] = pack
        draft_info[0, position, n_cards:] = pool
        pool[pick_idx] += 1
        actual_pick.append(correct_pick)
        pick_js = {
            "pack_number": pick["pack_number"],
            "pick_number": pick["pick_number"],
            "pack_cards": arena_ids_in_pack,
            "pick": arena_id_mapping[correct_pick],
        }
        js["picks"].append(pick_js)
    # insert n_cards idx to shift the picks passed into the model to prevent seeing the correct pick
    np_pick = np.tile(
        np.expand_dims(
            np.asarray([n_cards] + [name_to_idx[name] for name in actual_pick[:-1]]), 0
        ),
        batch_size,
    ).reshape(batch_size, t)
    model_input = (
        tf.convert_to_tensor(draft_info[:, :, :n_cards], dtype=tf.float32),
        tf.convert_to_tensor(np_pick, dtype=tf.int32),
        tf.convert_to_tensor(positions, dtype=tf.int32),
    )
    if return_style == "input":
        return model_input
    # we get the first element in anything we return to handle the case where the model couldn't properly serialize
    # and we hence need to copy the data to be the same shape as the batch size in order to run a stored model
    if return_attention:
        output, attention = model(model_input, training=False, return_attention=True)
        output = output[0]
        # attention = tf.squeeze(attention)
    else:
        output = model(model_input, training=False)[0]
    if return_style == "output":
        if return_attention:
            return output, attention
        else:
            return output
    predictions = tf.math.top_k(output, k=3).indices.numpy()
    predicted_picks = [idx_to_name[pred[0]] for pred in predictions]
    if return_style == "df":
        df = pd.DataFrame()
        df["predicted_pick"] = predicted_picks
        df["human_pick"] = actual_pick
        df["second_choice"] = [idx_to_name[pred[1]] for pred in predictions]
        df["second_choice"].loc[
            [idx for idx in df.index if idx % n_picks_per_pack >= n_picks_per_pack - 1]
        ] = ""
        df["third_choice"] = [idx_to_name[pred[2]] for pred in predictions]
        df["third_choice"].loc[
            [idx for idx in df.index if idx % n_picks_per_pack >= n_picks_per_pack - 2]
        ] = ""
        df.index = [position_to_pxpy[idx] for idx in df.index]
        if return_attention:
            return df, attention
        return df
    for i, js_obj in enumerate(js["picks"]):
        js_obj["suggested_pick"] = arena_id_mapping[predicted_picks[i]]
    r = requests.post(url="https://www.17lands.com/api/submit_draft", json=js)
    r_js = r.json()
    if build_model is not None:
        pool = np.expand_dims(pool, 0)
        basics, spells, n_basics = build_decks_2(build_model, pool, cards=cards)
        deck_url = display_deck(pool, basics, spells, cards, return_url=True)
    else:
        deck_url = None
    try:
        draft_id = r_js["id"]
        output = f"https://www.17lands.com/submitted_draft/{draft_id}"
        if deck_url is not None:
            output = (output, deck_url)
        return output
    except:
        warnings.warn("Draft Log Upload Failed. Returning sent JSON to help debug.")
        return (js, r)


def display_draft(df, cmap=None, pack=None):
    if pack is not None:
        df = df.loc[[x for x in df.index if x.startswith("P" + str(pack))]]
    if cmap is None:
        cmap = LinearSegmentedColormap.from_list("gr", ["g", "w", "r"], N=256)
    cm = plt.cm.get_cmap(cmap)
    good_c = colors.rgb2hex(cm(int(cmap.N * 1 / 3)))
    bad_c = colors.rgb2hex(cm(int(cmap.N * 2 / 3)))
    human_picks = df["human_pick"].values
    anything_correct = np.zeros_like(human_picks)

    def f(dat, good_c="green", bad_c="red", human_col_val=None):
        output = []
        for i, pick in enumerate(dat):
            if human_col_val is not None:
                flag = pick == human_picks[i]
            else:
                flag = anything_correct[i]
                good_c = colors.rgb2hex(cm(int(cmap.N * (1 - flag))))
            if flag:
                output.append(f"background-color: {good_c}")
                if human_col_val is not None:
                    anything_correct[i] += human_col_val
            else:
                output.append(f"background-color: {bad_c}")
        return output

    style = df.style
    human_col_val_map = {
        "predicted_pick": 1.0,
        "second_choice": 2.0 / 3.0,
        "third_choice": 2.0 / 3.0,
    }
    for column in df.columns:
        if column == "human_pick":
            continue
        style = style.apply(
            f,
            axis=0,
            subset=column,
            good_c=good_c,
            bad_c=bad_c,
            human_col_val=human_col_val_map[column],
        )
    style = style.apply(f, axis=0, subset="human_pick", good_c=good_c, bad_c=bad_c)
    return style.set_properties(
        **{
            "text-align": "center",
            "padding": "10px",
            "border": "1px solid black",
            "margin": "0px",
        }
    )


def plot_attention_head(attention, pxpy):

    ax = plt.gca()
    ax.matshow(attention)
    ax.set_xticks(range(len(pxpy)))
    ax.set_yticks(range(len(pxpy)))

    ax.set_xticklabels(pxpy, rotation=90)

    ax.set_yticklabels(pxpy)


def plot_attention_weights(attention_heads):
    # pxpy = ["BIAS"]
    pxpy = []
    seq_l = attention_heads.shape[-1]
    n_picks = (seq_l) / 3
    for i in range(seq_l):
        pack = i // n_picks + 1
        pick = (i % n_picks) + 1
        pxpy.append("P" + str(int(pack)) + "P" + str(int(pick)))

    for h, head in enumerate(attention_heads):
        fig = plt.figure(figsize=(10, 30))
        plot_attention_head(head, pxpy)
        plt.scatter(range(seq_l), range(seq_l), color="red")
        plt.grid()
        plt.title(f"Head {h+1}")
        plt.tight_layout()
        plt.show()


def build_decks(basics, spells, n_basics, cards=None):
    n_basics = np.round(n_basics)
    n_spells = 40 - n_basics
    deck = np.concatenate([basics, spells], axis=-1)
    deck_out = np.zeros_like(deck)
    for i in range(0, 40):
        spell_argmax = np.squeeze(np.argmax(deck[:, 5:], axis=-1)) + 5
        basic_argmax = np.squeeze(np.argmax(deck[:, :5], axis=-1))
        card_to_add = np.where(np.squeeze(n_spells) > i, spell_argmax, basic_argmax,)
        idx = np.arange(deck.shape[0]), card_to_add
        deck[idx] -= 1
        deck_out[idx] += 1
    if cards is not None:
        deck_out = recalibrate_basics(np.squeeze(deck_out), cards)
        deck_out = deck_out[None, :]
    return deck_out[:, :5], deck_out[:, 5:]


def build_decks_2(model, pool, cards=None):
    pool = pool.copy()
    deck_out = np.zeros_like(pool)
    masked_flag = len(deck_out.shape) == 3
    spells_added = 0
    while True:
        basics, spells, n_non_basics = model((pool, deck_out), training=False)
        if np.round(n_non_basics) == 0:
            break
        spells = spells.numpy()
        basics = basics.numpy()
        n_non_basics = n_non_basics.numpy()[0][0]
        card_to_add = np.squeeze(np.argmax(spells, axis=-1))
        if not masked_flag:
            idx = np.arange(deck_out.shape[0]), card_to_add
        else:
            idx = (
                np.arange(deck_out.shape[0]),
                np.arange(deck_out.shape[0]),
                card_to_add,
            )
        deck_out[idx] += 1
        pool[idx] -= 1
        spells_added += 1
    basics_out = np.zeros((deck_out.shape[0], 5))
    for _ in range(40 - spells_added):
        card_to_add = np.squeeze(np.argmax(basics, axis=-1))
        if not masked_flag:
            idx = np.arange(deck_out.shape[0]), card_to_add
        else:
            idx = (
                np.arange(deck_out.shape[0]),
                np.arange(deck_out.shape[0]),
                card_to_add,
            )
        basics_out[idx] += 1
        basics[idx] -= 1
    deck_out = np.concatenate([basics_out, deck_out], axis=-1)
    if cards is not None:
        deck_out = recalibrate_basics(np.squeeze(deck_out), cards)
        deck_out = deck_out[None, :]
    return deck_out[:, :5], deck_out[:, 5:], 40 - spells_added


def recalibrate_basics(built_deck, cards, verbose=False):
    color_to_idx = (
        cards[cards["idx"] < 5]
        .set_index("idx")["produced_mana"]
        .apply(lambda x: x[0])
        .reset_index()
        .set_index("produced_mana")
        .to_dict()["idx"]
    )

    pip_count = {c: 0 for c in list("WUBRG")}
    # don't count a green mana dork that produces G as a G source, but if it produces other colors, it can count as a source
    basic_adds_extra_sources = {c: 0 for c in list("WUBRG")}
    splash_produces_count = {c: 0 for c in list("WUBRG")}
    for card_idx, count in enumerate(built_deck):
        if count == 0:
            continue
        card = cards[cards["idx"] == card_idx]
        basic_special_case_flag = (card["basic_land_search"]).iloc[0]
        mc = card["mana_cost"].iloc[0]
        splash_produce = (
            list(
                set(card["produced_mana"].iloc[0]) - {"C"} - set(card["colors"].iloc[0])
            )
            if not card["produced_mana"].isna().iloc[0]
            else []
        )
        for color in pip_count.keys():
            pip_count[color] += count * mc.count(color)
            if basic_special_case_flag:
                basic_count = built_deck[color_to_idx[color]]
                if basic_count == 0:
                    basic_adds_extra_sources[color] += count
                else:
                    splash_produces_count[color] += count
            elif color in splash_produce:
                splash_produces_count[color] += count
    min_produces_map = {
        0: 0,
        1: 3,
        2: 4,
        3: 4,
        4: 5,
    }

    add_basics_dict = {c: 0 for c in list("WUBRG")}

    cut_basics_dict = {c: 0 for c in list("WUBRG")}

    basic_cut_limit = {c: 0 for c in list("WUBRG")}

    for color in list("WUBRG"):
        pips = pip_count[color]
        if pips == 0:
            # ensure we cut basics that dont do anything
            idx_for_basic = color_to_idx[color]
            basic_count_in_deck = built_deck[idx_for_basic]
            cut_basics_dict[color] += basic_count_in_deck
        if pips > 0 and basic_adds_extra_sources[color] > 0:
            min_add = 1
        else:
            min_add = 0
        mana_req = min_produces_map.get(pips, 6)
        produces = splash_produces_count[color]
        produces_diff = produces - mana_req
        if produces_diff < 0:
            add_basics_dict[color] += (
                abs(produces_diff) - basic_adds_extra_sources[color]
            )
        else:
            basic_cut_limit[color] = max(produces_diff, 0)
        if add_basics_dict[color] < min_add:
            add_basics_dict[color] = min_add

    # now ad_basics_dict is the number of basics per color that needs to be added
    # the following logic determines what basics need to be cut
    # get number of basics in the deck, but if that basic is required to be added, don't allow it to be cut
    basics_that_can_be_cut = {
        c: min(built_deck[color_to_idx[c]], basic_cut_limit[c]) if n == 0 else 0
        for c, n in add_basics_dict.items()
    }
    # this is used for making swaps when adding basics. If we are forcing some basics to be cut, don't let them be added
    basics_that_can_be_cut = {
        c: np.clip(v - cut_basics_dict[c], 0, np.inf)
        for c, v in basics_that_can_be_cut.items()
    }
    total_basics_to_cut = sum([x for x in add_basics_dict.values()])
    if total_basics_to_cut > sum([x for x in basics_that_can_be_cut.values()]):
        if verbose:
            print("This manabase is not salvageable")
    cur_color_idx = 0
    colors_to_add = [c for c, n in add_basics_dict.items() if n > 0]
    check_bug = 0
    # if we are cutting lands, make sure to cut basics corresponding to
    # lower pips in the deck. This could have problems if there's already too
    # high an allocation to that, but empirically it seems more often balanced
    colors = sorted(list("WUBRG"), key=lambda color: pip_count[color])
    added_already = []
    while (
        sum([x for x in add_basics_dict.values()]) > 0
        or sum([x for x in cut_basics_dict.values()]) > 0
    ):
        if len(colors_to_add) == 0:
            if sum([x for x in add_basics_dict.values()]) > 0:
                colors_to_add = [c for c, n in add_basics_dict.items() if n > 0]
            else:
                if sum([x for x in cut_basics_dict.values()]) <= 0:
                    # nothing to add or cut!
                    break
                else:
                    colors_to_add = [
                        c
                        for c, n in basics_that_can_be_cut.items()
                        if n > 0 and c not in added_already
                    ]
                    if len(colors_to_add) == 0:
                        colors_to_add = [
                            c for c, n in basics_that_can_be_cut.items() if n > 0
                        ]
        if len(colors_to_add) == 0:
            if verbose:
                print("Nothing else is allowed to be cut, bad manabase")
            break
        c = colors[cur_color_idx % 5]
        # this is the actual idx in the deck built, not the fake one used to cycle through colors
        idx = color_to_idx[c]
        ad_c = colors_to_add[0]
        ad_idx = color_to_idx[ad_c]
        if sum([x for x in cut_basics_dict.values()]) > 0:
            if cut_basics_dict[c] > 0:
                built_deck[idx] -= 1
                built_deck[ad_idx] += 1
                basics_that_can_be_cut[c] -= 1
                cut_basics_dict[c] -= 1
                add_basics_dict[ad_c] -= 1
        else:
            if basics_that_can_be_cut[c] > 0:
                built_deck[idx] -= 1
                built_deck[ad_idx] += 1
                basics_that_can_be_cut[c] -= 1
                cut_basics_dict[c] -= 1
                add_basics_dict[ad_c] -= 1

        cur_color_idx += 1
        check_bug += 1
        if check_bug > 100:
            print("BUG")
            break
    return built_deck
