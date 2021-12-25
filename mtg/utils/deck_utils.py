def count_creature_producers(deck, cards):
    token_mask = cards["oracle_text"].str.lower().str.contains("create") & (
        cards["oracle_text"].str.lower().str.contains("token")
        | cards["oracle_text"].str.lower().str.contains("tokens")
    )
    creature_mask = cards["type_line"].str.lower().str.contains("creature")
    card_subset = cards[token_mask | creature_mask]["idx"].tolist()
    return deck[card_subset].sum()
