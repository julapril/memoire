import numpy as np


def cardinalite_sous_arbres(Enfants):
    cardinalite = [[recursif_cardinalite_sous_arbres(Enfants, Enfants[enfant])
                    for enfant in enfants]
                   for enfants in Enfants]
    cum_cardinalite = [list(np.cumsum(card)) for card in cardinalite]
    return [[c + 1 for c in card] for card in cum_cardinalite]


def recursif_cardinalite_sous_arbres(Enfants, enfants):
    if len(enfants) == 0:
        return 1
    else:
        return 1 + sum(
            recursif_cardinalite_sous_arbres(Enfants, Enfants[enfant])
            for enfant in enfants)
