import numpy as np

def CardinalitéSousArbres(Enfants):
    cardinalité = [[RécursifCardinalitéSousArbres(Enfants, Enfants[enfant]) for enfant in enfants] for enfants in
                   Enfants]
    cum_cardinalité = [list(np.cumsum(card)) for card in cardinalité]
    return [[c + 1 for c in card] for card in cum_cardinalité]


def RécursifCardinalitéSousArbres(Enfants, enfants):
    if len(enfants) == 0:
        return 1
    else:
        return 1 + sum(RécursifCardinalitéSousArbres(Enfants, Enfants[enfant]) for enfant in enfants)
