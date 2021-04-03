def print_deck(deck, card_df, sort_by="name"):
    card_df = card_df.sort_values(by=sort_by)
    for card_idx, card_name in card_df[['idx','name']].to_numpy():
        count = deck[card_idx]
        if count > 0:
            print(count,card_name)
    
def print_counts(deck, card_df, col="type_line"):
    col_counts = dict()
    card_df = card_df.set_index('idx')
    for card_idx, card_count in enumerate(deck):
        if card_count > 0:
            val = card_df.loc[card_idx,col]
            if val in col_counts:
                col_counts[val] += card_count
            else:
                col_counts[val] = card_count
    for key, value in col_counts.items():
        print(key,":",value) 

