import numpy as np

from CardinalitéSousArbres import CardinalitéSousArbres


def obtenir_probabilite_independance_conditionnelle(sommets, aretes,
                                                    probabilites_univariees, probabilites_bivariees,
                                                    nombre_de_defauts):
    # sommets doit être une liste d'entiers consécutifs de 0 à n-1
    # aretes doit être une liste de tuples de sommets donnant un arbre, avec parent < enfant
    # probabilites_univariees doit être une liste de flottants entre 0 et 1, de même longueur que sommets
    # probabilites_bivariees doit être une liste de flottants entre 0 et 1, de même longueur que aretes
    # nombre_de_defauts doit être un entier plus petit ou égal au nombre de sommets
    n = len(sommets)
    enfants = [[a[1] for a in aretes if a[0] == s] for s in sommets]
    d = [len(enfants) for enfants in enfants]  # Degrés sortants
    N = CardinalitéSousArbres(enfants)  # Cardinalité des premiers sous-arbres

    p_1 = probabilites_univariees
    p_0 = [1 - p_1[sommet] for sommet in sommets]

    pp_11 = probabilites_bivariees
    pp_00 = [1 - p_1[arete[0]]
             - p_1[arete[1]]
             + pp_11[j] for j, arete in enumerate(aretes)]
    pp_01 = [p_1[arete[1]]
             - pp_11[j] for j, arete in enumerate(aretes)]
    pp_10 = [p_1[arete[0]]
             - pp_11[j] for j, arete in enumerate(aretes)]

    w = np.zeros((n, max(d) + 1, 2, n + 1))

    for i in sommets:
        w[i, 0, 0, 0] = p_0[i]
        w[i, 0, 1, 1] = p_1[i]

    for i in reversed(sommets):
        if d[i] >= 1:
            i_1 = enfants[i][0]
            d_i_1 = d[i_1]
            N_i_1 = N[i][1 - 1]
            a_i_i1 = aretes.index((i, i_1))
            for t in range(0, N_i_1 - 1 + 1):
                if t <= N_i_1 - 2:
                    w[i, 1, 0, t] += w[i_1, d_i_1, 0, t] * pp_00[a_i_i1] / p_0[i_1]
                if t >= 1:
                    w[i, 1, 0, t] += w[i_1, d_i_1, 1, t] * pp_01[a_i_i1] / p_1[i_1]
            for t in range(1, N_i_1 + 1):
                if t <= N_i_1 - 1:
                    w[i, 1, 1, t] += w[i_1, d_i_1, 0, t - 1] * pp_10[a_i_i1] / p_0[i_1]
                if t >= 2:
                    w[i, 1, 1, t] += w[i_1, d_i_1, 1, t - 1] * pp_11[a_i_i1] / p_1[i_1]

        if d[i] >= 2:
            for s in range(2, d[i] + 1):
                i_s = enfants[i][s - 1]
                d_i_s = d[i_s]
                N_i_s = N[i][s - 1]
                N_i_s_prec = N[i][s - 1 - 1]
                N_is_dis = N_i_s - N_i_s_prec
                a_i_is = aretes.index((i, i_s))
                # y = 0
                for t in range(0, N_i_s - 1 + 1):
                    if t <= N_i_s - 2:
                        a_00_min = max(0, t - N_i_s_prec + 1)
                        a_00_max = min(N_is_dis - 1, t)
                        for a in range(a_00_min, a_00_max + 1):
                            w[i, s, 0, t] += (w[i, s - 1, 0, t - a]
                                              * w[i_s, d_i_s, 0, a]
                                              * pp_00[a_i_is]
                                              / p_0[i]
                                              / p_0[i_s])
                    if t >= 1:
                        a_01_min = max(1, t - N_i_s_prec + 1)
                        a_01_max = min(N_is_dis, t)
                        for a in range(a_01_min, a_01_max + 1):
                            w[i, s, 0, t] += (w[i, s - 1, 0, t - a]
                                              * w[i_s, d_i_s, 1, a]
                                              * pp_01[a_i_is]
                                              / p_0[i]
                                              / p_1[i_s])
                # y = 1
                for t in range(1, N_i_s + 1):
                    if t <= N_i_s - 1:
                        a_10_min = max(0, t - N_i_s_prec)
                        a_10_max = min(N_is_dis - 1, t - 1)
                        for a in range(a_10_min, a_10_max + 1):
                            w[i, s, 1, t] += (w[i, s - 1, 1, t - a]
                                              * w[i_s, d_i_s, 0, a]
                                              * pp_10[a_i_is]
                                              / p_1[i]
                                              / p_0[i_s])
                    if t >= 2:
                        a_11_min = max(1, t - N_i_s_prec)
                        a_11_max = min(N_is_dis, t - 1)
                        for a in range(a_11_min, a_11_max + 1):
                            w[i, s, 1, t] += (w[i, s - 1, 1, t - a]
                                              * w[i_s, d_i_s, 1, a]
                                              * pp_11[a_i_is]
                                              / p_1[i]
                                              / p_1[i_s])

    racine = sommets[0]
    prob = sum(w[racine, d[racine], 0, t] + w[racine, d[racine], 1, t] for t in range(nombre_de_defauts, n + 1))
    return prob
