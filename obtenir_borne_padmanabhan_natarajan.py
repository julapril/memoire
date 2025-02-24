from pulp import (LpMinimize,
                  LpProblem,
                  lpSum,
                  LpVariable,
                  value,
                  PULP_CBC_CMD)

from CardinalitéSousArbres import CardinalitéSousArbres


def obtenir_borne_padmanabhan_natarajan(sommets, aretes,
                                        probabilites_univariees,
                                        probabilites_bivariees,
                                        nombre_de_defauts,
                                        minimiser=False):
    # sommets doit être une liste d'entiers consécutifs de 0 à n-1
    # aretes doit être une liste de tuples de sommets donnant un arbre
    # probabilites_univariees doit être une liste de flottants
    #   entre 0 et 1, de même longueur que sommets
    # probabilites_bivariees doit être une liste de flottants
    #   entre 0 et 1, de même longueur que aretes
    # nombre_de_defauts doit être un entier plus petit ou égal
    #   au nombre de sommets
    # minimiser doit être une valeur booléenne, False pour
    #   la borne supérieure et True pour la borne inférieure
    n = len(sommets)
    if minimiser:
        minimiser_probabilites_univariees = \
            [1 - prob for prob in probabilites_univariees]
        minimiser_probabilites_bivariees = \
            [1 - probabilites_univariees[aretes[k][0]]
             - probabilites_univariees[aretes[k][1]]
             + probabilites_bivariees[k] for k in range(n - 1)]
        minimiser_nombre_de_defauts = n - nombre_de_defauts + 1
        return \
            (1 -
             obtenir_borne_padmanabhan_natarajan(
                 sommets=sommets,
                 aretes=aretes,
                 probabilites_univariees=minimiser_probabilites_univariees,
                 probabilites_bivariees=minimiser_probabilites_bivariees,
                 nombre_de_defauts=minimiser_nombre_de_defauts,
                 minimiser=False)
             )

    lp = LpProblem(name="lp-Padmanabhan-Natarajan", sense=LpMinimize)

    # Préparation des informations sur l'arbre
    enfants = [[a[1] for a in aretes if a[0] == s] for s in sommets]
    d = [len(enfants) for enfants in enfants]  # Degrés sortants
    N = CardinalitéSousArbres(enfants)  # Cardinalité des premiers sous-arbres

    # Initialisation des variables de décision
    lambdaa = LpVariable("lambda")
    alpha = LpVariable.dicts("alpha",
                             (i for i in range(n)))
    beta = LpVariable.dicts("beta",
                            (k for k in range(n - 1)))
    Delta = LpVariable.dicts("Delta",
                             (k for k in range(n - 1)),
                             lowBound=0)
    eta = LpVariable.dicts("eta",
                           (k for k in range(n - 1)),
                           lowBound=0)
    chi = LpVariable.dicts("chi",
                           (k for k in range(n - 1)),
                           lowBound=0)
    gamma = LpVariable.dicts("gamma",
                             (k for k in range(n - 1)),
                             lowBound=0)
    tau = LpVariable.dicts("tau",
                           (i for i in range(n)),
                           lowBound=0)
    x = LpVariable.dicts("x", ((i, s, y, t)
                               for i in range(n)
                               for s in range(d[i] + 1)
                               for y in [0, 1]
                               # (1 - y, N[i][s - 1] - (1 - y) + 1)
                               for t in range(n + 1)))
    z = LpVariable("z")

    # Fonction objectif
    lp += (lambdaa
           + lpSum(alpha[i] * probabilites_univariees[i]
                   for i in range(n))
           + lpSum(beta[k] * probabilites_bivariees[k]
                   for k in range(n - 1)))

    # Contraintes
    lp += (lambdaa
           - lpSum(Delta[k] + chi[k] for k in range(n - 1))
           - lpSum(tau[i] for i in range(n)) >= 0)
    for i in range(n):
        lp += (lpSum(Delta[k] - eta[k]
                     for k in range(n - 1) if aretes[k][0] == i)
               + lpSum(Delta[k] - gamma[k]
                       for k in range(n - 1) if aretes[k][1] == i)
               + tau[i] + alpha[i] >= 0)
    for k in range(n - 1):
        lp += eta[k] + gamma[k] - Delta[k] + chi[k] + beta[k] >= 0
    lp += lambdaa + z >= 1
    for t in range(nombre_de_defauts, n + 1):
        lp += x[0, d[0], 0, t] - z >= 0
        lp += x[0, d[0], 1, t] - z >= 0
    for i in range(n):
        for s in range(d[i] + 1):
            lp += x[i, s, 0, 0] == 0
            lp += x[i, s, 1, 1] - alpha[i] == 0
    for i in range(n):
        if d[i] >= 1:
            i_1 = enfants[i][0]
            d_i_1 = d[i_1]
            N_i_1 = N[i][0]
            for t in range(N_i_1 - 2 + 1):
                lp += x[i_1, d_i_1, 0, t] - x[i, 1, 0, t] >= 0
            for t in range(1, N_i_1 - 1 + 1):
                lp += x[i_1, d_i_1, 1, t] - x[i, 1, 0, t] >= 0
                lp += (x[i_1, d_i_1, 0, t - 1] - x[i, 1, 1, t]
                       + alpha[i] >= 0)
            for t in range(2, N_i_1 + 1):
                lp += (x[i_1, d_i_1, 1, t - 1] - x[i, 1, 1, t]
                       + alpha[i] + beta[aretes.index((i, i_1))] >= 0)
    for i in range(n):
        if d[i] >= 2:
            for s in range(2, d[i] + 1):
                i_s = enfants[i][s - 1]
                d_i_s = d[i_s]
                N_i_s = N[i][s - 1]
                N_i_s_prec = N[i][s - 1 - 1]
                N_is_dis = N_i_s - N_i_s_prec
                for t in range(N_i_s - 2 + 1):
                    a_1_min = max(0, t - N_i_s_prec + 1)
                    a_1_max = min(N_is_dis - 1, t)
                    for a in range(a_1_min, a_1_max + 1):
                        lp += (x[i, s - 1, 0, t - a]
                               + x[i_s, d_i_s, 0, a]
                               - x[i, s, 0, t] >= 0)
                for t in range(1, N_i_s - 1 + 1):
                    a_2_min = max(1, t - N_i_s_prec + 1)
                    a_2_max = min(N_is_dis, t)
                    for a in range(a_2_min, a_2_max + 1):
                        lp += (x[i, s - 1, 0, t - a]
                               + x[i_s, d_i_s, 1, a]
                               - x[i, s, 0, t] >= 0)
                    a_3_min = max(0, t - N_i_s_prec)
                    a_3_max = min(N_is_dis - 1, t - 1)
                    for a in range(a_3_min, a_3_max + 1):
                        lp += (x[i, s - 1, 1, t - a]
                               + x[i_s, d_i_s, 0, a]
                               - x[i, s, 1, t] >= 0)
                for t in range(2, N_i_s + 1):
                    a_4_min = max(1, t - N_i_s_prec)
                    a_4_max = min(N_is_dis, t - 1)
                    for a in range(a_4_min, a_4_max + 1):
                        lp += (x[i, s - 1, 1, t - a]
                               + x[i_s, d_i_s, 1, a]
                               - x[i, s, 1, t]
                               + beta[aretes.index((i, i_s))] >= 0)

    lp.solve(PULP_CBC_CMD(msg=False))
    return value(lp.objective)
