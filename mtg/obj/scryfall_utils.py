def merge_card_faces(row):
    nans = row.isna()
    if nans["card_faces"]:
        return row
    card_faces = row["card_faces"]
    face_1_keys = card_faces[0].keys()
    face_2_keys = card_faces[1].keys()
    face = dict()
    for key in face_1_keys:
        if key in ["power", "toughness"]:
            try:
                val = int(card_faces[0][key])
            except:
                val = card_faces[0][key]
        else:
            val = card_faces[0][key]
        face[key] = val
    for key in face_2_keys:
        if key in ["power", "toughness"]:
            try:
                val = int(card_faces[1][key])
            except:
                val = card_faces[1][key]
        else:
            val = card_faces[1][key]
        if key not in face.keys():
            face[key] = val
        else:
            if key in ["oracle_text", "flavor_text", "type_line"]:
                face[key] = face[key] + "\n//\n" + val
            elif key == "colors":
                face[key] = list(set(face[key]).union(set(val)))
    for key, val in face.items():
        if key not in nans.index:
            continue
        if nans[key]:
            row[key] = val
    return row


def produce_for_splash(row):
    nans = row.isna()
    if nans["produced_mana"]:
        return []
    return list(set(row["produced_mana"]) - {"C"} - set(row["colors"]))

