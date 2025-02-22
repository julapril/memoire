import json


def formatter_structure(copule_vigne):
    copule_vigne.to_file("copule_vigne.json")

    with open("copule_vigne.json", "r") as f:
        loaded_json = json.load(f)
        structure = loaded_json.get("structure")

    d = structure['array']['d']
    sommets = list(range(d))
    ordre = [element - 1 for element in structure['order']]
    donnees = [element - 1 for element in structure['array']['data'][0]]
    aretes = [(ordre[i], ordre[donnees[i]]) for i in range(d - 1)]
    aretes_originales = aretes

    sommets_ordonnees = [aretes[0][0], aretes[0][1]]
    descendants = [aretes[0][1]]
    aretes_ordonnees = [aretes.pop(0)]

    for arrete in aretes:
        if sommets_ordonnees[0] in arrete:
            enfant = arrete[1] if arrete[0] == sommets_ordonnees[0] else arrete[0]
            sommets_ordonnees.append(enfant)
            aretes_ordonnees.append(arrete)
            descendants.append(enfant)
        aretes = [arrete for arrete in aretes if arrete not in aretes_ordonnees]

    while len(descendants) > 0:
        descendant = descendants.pop(0)
        for arrete in aretes:
            if descendant in arrete:
                enfant = arrete[1] if arrete[0] == descendant else arrete[0]
                sommets_ordonnees.append(enfant)
                aretes_ordonnees.append(arrete)
                descendants.append(enfant)
        aretes = [arrete for arrete in aretes if arrete not in aretes_ordonnees]

    sommets = range(len(sommets_ordonnees))
    # Remplacer les nombres dans la deuxième liste par leurs positions dans la première liste
    index_orig_ord = {num: idx for idx, num in enumerate(sommets_ordonnees)}
    index_ord_orig = {idx: num for idx, num in enumerate(sommets_ordonnees)}
    aretes = [(index_orig_ord[a], index_orig_ord[b]) for a, b in aretes_ordonnees]
    aretes = [(min(a, b), max(a, b)) for a, b in aretes]
    return {'d': d,
            'sommets': sommets,
            'index': index_ord_orig,
            'aretes': aretes,
            'aretes_originales': aretes_originales}
