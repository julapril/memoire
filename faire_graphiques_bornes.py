import numpy as np
import matplotlib.pyplot as plt

from obtenir_borne_padmanabhan_natarajan import obtenir_borne_padmanabhan_natarajan
from obtenir_probabilite_independance_conditionnelle import obtenir_probabilite_independance_conditionnelle


def faire_graphiques_bornes(n_debut, n_fin, n_saut, regle_k, regle_k_nom, p_uni, p_bi, p_nom):
    nombre_de_familles_d_arbre = 4
    bornes = np.zeros((3 * nombre_de_familles_d_arbre + 1, (n_fin - n_debut + 1) // n_saut))
    bornes[0] = list(range(n_debut, n_fin, n_saut))

    for j, n in enumerate(range(n_debut, n_fin, n_saut)):
        # Sommets
        sommets = list(range(n))
        # Arbre en série
        aretes_serie = [(i, i + 1) for i in range(n - 1)]
        # Arbre binaire
        aretes_binaire = [(i // 2, i + 1) for i in range(n - 1)]
        # Arbre ternaire
        aretes_ternaire = [(i // 3, i + 1) for i in range(n - 1)]
        # Arbre en étoile
        aretes_etoile = [(0, i + 1) for i in range(n - 1)]
        # Une liste de différents ensembles d'aretes
        aretes = [aretes_serie, aretes_binaire, aretes_ternaire, aretes_etoile]

        # Probabilités univariées et bivariées
        p = [p_uni for _ in range(n)]
        pp = [p_bi for _ in range(n - 1)]

        # Nombre de défauts
        k = regle_k(n)

        for i, aretes in enumerate(aretes):
            borne_inf = obtenir_borne_padmanabhan_natarajan(sommets=sommets, aretes=aretes,
                                                            probabilites_univariees=p, probabilites_bivariees=pp,
                                                            nombre_de_defauts=k, minimiser=True)
            borne_sup = obtenir_borne_padmanabhan_natarajan(sommets=sommets, aretes=aretes,
                                                            probabilites_univariees=p, probabilites_bivariees=pp,
                                                            nombre_de_defauts=k, minimiser=False)
            prob_ind = obtenir_probabilite_independance_conditionnelle(sommets=sommets, aretes=aretes,
                                                                       probabilites_univariees=p,
                                                                       probabilites_bivariees=pp,
                                                                       nombre_de_defauts=k)
            bornes[3 * i + 1][j] = borne_inf
            bornes[3 * i + 2][j] = borne_sup
            bornes[3 * i + 3][j] = prob_ind

    # Graphiques
    titres = ['Arbre série', 'Arbre binaire', 'Arbre ternaire', 'Arbre étoile']

    for i in range(nombre_de_familles_d_arbre):
        plt.figure(figsize=(6, 4))
        n_values = bornes[0]
        borne_inf = bornes[3 * i + 1]
        borne_sup = bornes[3 * i + 2]
        prob_ind = bornes[3 * i + 3]

        plt.plot(n_values, borne_inf, label='Borne inférieure', marker='o')
        plt.plot(n_values, borne_sup, label='Borne supérieure', marker='x')
        plt.plot(n_values, prob_ind, label='Indépendance conditionnelle', marker='s')

        plt.xlabel('n')
        plt.ylabel('Probabilité')

        bornes_min = np.min(bornes[1:, :])
        bornes_max = np.max(bornes[1:, :])
        delta = bornes_max - bornes_min
        ylim_inf = bornes_min - delta * 0.05
        ylim_sup = bornes_max + delta * 0.05
        plt.ylim(ylim_inf, ylim_sup)

        plt.legend()
        plt.grid(True)

        filename = f"bornes_{regle_k_nom}_{p_nom}_{titres[i].replace(' ', '_').lower()}.png"
        plt.savefig(filename, dpi=1200)
        plt.show()
        plt.close()
