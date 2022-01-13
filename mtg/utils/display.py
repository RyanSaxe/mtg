import tensorflow as tf
import requests
import numpy as np
import matplotlib.pyplot as plt
import warnings
from mpl_toolkits.axes_grid1 import make_axes_locatable
import os
import pathlib
from mtg.utils.dataloading_utils import get_draft_json


def names_to_array(names, mapping):
    """
    convert json objects from 17lands to array the model can use
    """
    if len(names) > 0:
        names = [x["name"].lower().split("//")[0].strip() for x in names]
    else:
        names = None
    idxs = [mapping[name] for name in names]
    unique, counts = np.unique(idxs, return_counts=True)
    arr = np.zeros(len(mapping))
    arr[unique] += counts
    return arr


def display_deck(pool, basics, spells, cards, return_url=False):
    """
    given deckbuilder model output, return either the text of the build or a link
     to sealeddeck.tech
    """
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


def draft_sim(
    expansion, model, token="", build_model=None, basic_prior=True,
):
    """
    run a draft table with 8 copies of bots
    """
    t = expansion.t
    idx_to_name = expansion.get_mapping("idx", "name", include_basics=False)
    name_to_idx = expansion.get_mapping("name", "idx", include_basics=False)
    arena_mapping = expansion.get_mapping("name", "arena_id", include_basics=False)
    cards = expansion.cards.copy()
    seats = 8
    n_packs = 3
    n_cards = len(idx_to_name)
    n_picks = t // n_packs

    js = {
        idx: {
            "expansion": expansion.expansion.upper(),
            "token": f"{token}",
            "picks": [],
        }
        for idx in range(seats)
    }

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
            basics, spells, _ = build_decks(
                build_model, pool.copy(), cards=cards if basic_prior else None
            )
            deck_url = display_deck(pool, basics, spells, cards, return_url=True)
        else:
            deck_url = None
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
    expansion,
    batch_size=1,
    token="",
    build_model=None,
    mod_lookup=dict(),
    basic_prior=True,
    att_folder=None,
):
    """
    given a draft log, create a copy of that log that highlights what the bot would do

    att_folder: directory for storing attention visualizations
    basic_prior: heuristic update of manabase in deckbuilder
    mod_lookup: dictionary that lets you modify the data to prod the model and see if it
                changes decisions. Use it as such:

                {
                    'PxPy':{
                        'pack':{
                            #change cardA to cardB in PxPy
                            'cardA':'cardB'
                        },
                        #change the pick to cardC
                        'pick': 'cardC'
                    }
                    'pool':{
                        # remove two copies of cardD from the pool and replace
                        # it with a copy of cardE and a copy of cardD
                        'cardD':-2,
                        'cardE':1,
                        'cardF':1
                    }
                }
    """
    t = expansion.t
    idx_to_name = expansion.get_mapping("idx", "name", include_basics=False)
    name_to_idx = expansion.get_mapping("name", "idx", include_basics=False)
    arena_mapping = expansion.get_mapping("name", "arena_id", include_basics=False)
    cards = expansion.cards.copy()
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
    js = {"expansion": expansion.expansion.upper(), "token": f"{token}", "picks": []}
    for pick in picks:
        pxpy = "P" + str(pick["pack_number"] + 1) + "P" + str(pick["pick_number"] + 1)
        pack_mod = mod_lookup.get(pxpy, dict()).get("pack", dict())
        pick_mod = mod_lookup.get(pxpy, dict()).get("pick", None)
        for i, option in enumerate(pick["available"]):
            cardname = option["name"].lower().split("//")[0].strip()
            if cardname in pack_mod:
                pick["available"][i]["name"] = pack_mod[cardname]

        position = int(pick["pack_number"] * n_picks_per_pack + pick["pick_number"])
        if pick_mod is not None:
            correct_pick = pick_mod
        else:
            correct_pick = pick["pick"]["name"].lower().split("//")[0].strip()
        position_to_pxpy[position] = pxpy
        pick_idx = name_to_idx[correct_pick]
        arena_ids_in_pack = names_to_array(pick["available"], arena_mapping)
        pack = names_to_array(pick["available"], name_to_idx)
        draft_info[0, position, :n_cards] = pack
        draft_info[0, position, n_cards:] = pool
        pool[pick_idx] += 1
        actual_pick.append(correct_pick)
        pick_js = {
            "pack_number": pick["pack_number"],
            "pick_number": pick["pick_number"],
            "pack_cards": arena_ids_in_pack,
            "pick": arena_mapping[correct_pick],
        }
        js["picks"].append(pick_js)
    pool_mod = mod_lookup.get("pool", dict())
    for cardname, n_change in pool_mod.items():
        card_idx = name_to_idx[cardname]
        pool[card_idx] += n_change
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
    # we get the first element in anything we return to handle the case where the model couldn't properly serialize
    # and we hence need to copy the data to be the same shape as the batch size in order to run a stored model
    output, attention = model(model_input, training=False, return_attention=True)
    output = output[0]

    if att_folder is not None:
        draft_id = draft_log_url.split("/")[-1]
        location = os.path.join(att_folder, draft_id)
        att = {"pack": attention[0], "pick": attention[1][0], "final": attention[1][1]}
        for att_name, att_vec in att.items():
            # plot attention, shifted right if we're visualizing pick attention
            att_loc = os.path.join(location, att_name, shift=att_name == "pick")
            # index because shape is (1, n_heads, seq, seq)
            save_att_to_dir(att_vec[0], att_loc)

    predictions = tf.math.top_k(output, k=3).indices.numpy()
    predicted_picks = [idx_to_name[pred[0]] for pred in predictions]
    for i, js_obj in enumerate(js["picks"]):
        js_obj["suggested_pick"] = arena_mapping[predicted_picks[i]]
    r = requests.post(url="https://www.17lands.com/api/submit_draft", json=js)
    r_js = r.json()
    if build_model is not None:
        pool = np.expand_dims(pool, 0)
        basics, spells, _ = build_decks(
            build_model, pool.copy(), cards=cards if basic_prior else None
        )
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


def save_att_to_dir(attention, location, shift=False):
    """
    create and store images showing each attention heads activations for 
        the different places in models using attention. 

    This aligns the heads such that it's easier to recognize patterns related
        to which head learns to process what
    """
    pathlib.Path(location).mkdir(parents=True, exist_ok=True)
    if shift:
        pxpy = ["BIAS"]
    else:
        pxpy = []
    seq_l = attention.shape[-1]
    n_picks = (seq_l) / 3
    for i in range(seq_l):
        pack = i // n_picks + 1
        pick = (i % n_picks) + 1
        pxpy.append("P" + str(int(pack)) + "P" + str(int(pick)))
    if shift:
        # if we shift right, we exclude the last pick of pack 3
        pxpy = pxpy[:-1]
    for i, pick in enumerate(pxpy):
        img_loc = os.path.join(location, pick + ".png")
        attention_weights = attention[:, i, : i + 1]
        xlabels = pxpy[: i + 1]
        fig = plt.figure(figsize=(900 / 96, 600 / 96), dpi=96)
        plt.grid()
        ax = plt.gca()
        mat = ax.matshow(attention_weights)
        ax.set_xticks(range(attention_weights.shape[-1]))
        ax.set_yticks(range(attention_weights.shape[0]))
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        plt.colorbar(mat, cax=cax)
        ax.set_xticklabels(xlabels, rotation=90)
        plt.tight_layout()
        plt.savefig(img_loc)
        plt.clf()


def build_decks(model, pool, cards=None):
    """
    iteratively call the model to build the deck from a card pool
    """
    pool = np.expand_dims(pool, 0)
    deck_out = np.zeros_like(pool)
    masked_flag = len(deck_out.shape) == 3
    spells_added = 0
    while True:
        basics, spells, n_non_basics = model((pool, deck_out), training=False)
        if np.round(n_non_basics) <= spells_added:
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
    # overwrite basics prediction using the actual discrete deck
    # not continuous representation
    basics = model.basic_decoder(deck_out) * (40 - spells_added)
    basics = basics.numpy()
    basics_out = np.zeros((*deck_out.shape[: len(deck_out.shape) - 1], 5))
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
    else:
        deck_out = deck_out[0]
    return deck_out[:, :5], deck_out[:, 5:], 40 - spells_added


def recalibrate_basics(built_deck, cards, verbose=False):
    """
    heuristic modification of basics in deckbuild to avoid OOD yielding 
     weird manabases (e.g. basic that cant cast anything)

    --> eventually this will not be necessary, once deckbuilder improves
    """
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
