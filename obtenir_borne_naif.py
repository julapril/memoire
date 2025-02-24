import numpy as np
from scipy.optimize import linprog


def obtenir_borne_naif(sommets, aretes,
                       probabilites_univariees, probabilites_bivariees,
                       nombre_de_defauts, minimiser=False):
    # sommets doit être une liste d'entiers consécutifs de 0 à n-1
    # aretes doit être une liste de tuples de sommets définissant un arbre
    # probabilites_univariees doit être une liste de flottants
    #   entre 0 et 1, de même longueur que sommets
    # probabilites_bivariees doit être une liste de flottants
    #   entre 0 et 1, de même longueur que aretes
    # nombre_de_defauts doit être un entier inférieur ou égal
    #   au nombre de sommets
    # minimiser doit être une valeur booléenne, False pour
    #   la borne supérieure et True pour la borne inférieure

    n = len(sommets)
    # Matrice dont les lignes représentent les supports de Bernoulli multivariés
    G = np.array([[(i // 2 ** j) % 2 for j in reversed(range(n))]
                  for i in range(2 ** n)], dtype=bool)
    # Matrice donnant les couples de variables aléatoires liés par une arete
    #   et simultanément égaux à 1
    GG = np.array([[G[i][e[0]] & G[i][e[1]] for e in aretes]
                   for i in range(2 ** n)], dtype=bool)
    # Vecteur de "1"
    uns = np.ones((2 ** n, 1), dtype=bool)
    un = np.ones(1)
    # Matrice avec des vecteurs dont la somme dépasse nombre_de_defauts
    chi = np.array(np.sum(G, axis=1) >= nombre_de_defauts, dtype=int)
    # On minimise l'objectif négatif au lieu de maximiser l'objectif
    obj = chi if minimiser else -chi
    bnd = [(0, np.inf)] * 2 ** n
    gauche_eq = np.block([[np.transpose(G)],
                          [np.transpose(GG)],
                          [np.transpose(uns)]])
    droite_eq = np.block([[probabilites_univariees],
                          [probabilites_bivariees],
                          [un]])

    opt = linprog(c=obj, A_eq=gauche_eq, b_eq=droite_eq, bounds=bnd)
    valeur_opt = opt.fun if minimiser else -opt.fun
    return valeur_opt
