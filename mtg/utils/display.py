def print_deck(deck, cards, sort_by="name", return_str=False):
    cards = cards.sort_values(by=sort_by)
    output = ""
    for card_idx, card_name in cards[['idx','name']].to_numpy():
        count = deck[card_idx]
        if count > 0:
            print(count,card_name)
            if return_str:
                output += str(count) + " " + card_name + "\n"
    if return_str:
        return output
    
def print_counts(deck, cards, col="type_line"):
    col_counts = dict()
    cards = cards.set_index('idx')
    for card_idx, card_count in enumerate(deck):
        if card_count > 0:
            val = cards.loc[card_idx,col]
            if val in col_counts:
                col_counts[val] += card_count
            else:
                col_counts[val] = card_count
    for key, value in col_counts.items():
        print(key,":",value) 

