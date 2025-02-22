import numpy as np
import timeit
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display, Markdown

from obtenir_borne_naif import obtenir_borne_naif
from obtenir_borne_padmanabhan_natarajan import obtenir_borne_padmanabhan_natarajan


def analyse_temps_execution(n_debut_comp, n_fin_comp, n_valeurs_p_n, regle_k, nb_simulation, p_uni, p_bi):
    nombre_de_familles_d_arbre = 4

    # region Temps d'éxecution Naïf
    temps_d_execution_naif = np.zeros((1 + nombre_de_familles_d_arbre, n_fin_comp - n_debut_comp))
    temps_d_execution_naif[0] = list(range(n_debut_comp, n_fin_comp))

    for j, n in enumerate(range(n_debut_comp, n_fin_comp)):
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
        p = np.array([[p_uni] for _ in range(n)])
        pp = np.array([[p_bi] for _ in range(n - 1)])

        # Nombre de défauts
        k = regle_k(n)

        for i, aretes in enumerate(aretes):
            execution = lambda: obtenir_borne_naif(sommets=sommets, aretes=aretes,
                                                   probabilites_univariees=p, probabilites_bivariees=pp,
                                                   nombre_de_defauts=k, minimiser=False)
            temps_d_execution_naif[i + 1][j] = timeit.timeit(stmt=execution, globals=globals(),
                                                             number=nb_simulation) / nb_simulation

    colonnes = [f"n={int(n)}" for n in temps_d_execution_naif[0]]
    index = ["Arbre binaire", "Arbre ternaire", "Arbre en série", "Arbre en étoile"]
    df_naif = pd.DataFrame(temps_d_execution_naif[1:], columns=colonnes, index=index)

    # Afficher le DataFrame
    titre = "Temps d'exécution de l'algorithme naïf pour différentes structures et tailles d'arbres"
    display(Markdown(f"## {titre}"))
    display(df_naif)
    df_naif.to_excel('temps_d_execution_naif.xlsx', header=True)
    # endregion

    # region Temps d'exécution Padmanabhan Natarajan pour comparaison

    temps_d_execution_p_n_comp = np.zeros((1 + nombre_de_familles_d_arbre, n_fin_comp - n_debut_comp))
    temps_d_execution_p_n_comp[0] = list(range(n_debut_comp, n_fin_comp))

    for j, n in enumerate(range(n_debut_comp, n_fin_comp)):
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
            execution = lambda: obtenir_borne_padmanabhan_natarajan(sommets=sommets, aretes=aretes,
                                                                    probabilites_univariees=p, probabilites_bivariees=pp,
                                                                    nombre_de_defauts=k, minimiser=False)
            temps_d_execution_p_n_comp[i + 1][j] = timeit.timeit(stmt=execution, globals=globals(),
                                                                 number=nb_simulation) / nb_simulation

    colonnes = [f"n={int(n)}" for n in temps_d_execution_p_n_comp[0]]
    index = ["Arbre binaire", "Arbre ternaire", "Arbre en série", "Arbre en étoile"]
    df_p_n = pd.DataFrame(temps_d_execution_p_n_comp[1:], columns=colonnes, index=index)

    # Afficher le DataFrame
    titre = "Temps d'exécution de l'algorithme Padmanabhan-Natarajan pour différentes structures et tailles d'arbres"
    display(Markdown(f"## {titre}"))
    display(df_p_n)
    df_p_n.to_excel('temps_d_exécution_Padmanabhan-Natarajan_comparaison.xlsx', header=True)
    # endregion

    # region Temps d'exécution Padmanabhan Natarajan pour ordre

    temps_d_execution_p_n = np.zeros((1 + nombre_de_familles_d_arbre, len(n_valeurs_p_n)))
    temps_d_execution_p_n[0] = n_valeurs_p_n

    for j, n in enumerate(n_valeurs_p_n):
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
            execution = lambda: obtenir_borne_padmanabhan_natarajan(sommets=sommets, aretes=aretes,
                                                                    probabilites_univariees=p, probabilites_bivariees=pp,
                                                                    nombre_de_defauts=k, minimiser=False)
            temps_d_execution_p_n[i + 1][j] = timeit.timeit(stmt=execution, globals=globals(),
                                                            number=nb_simulation) / nb_simulation

    colonnes = [f"n={int(n)}" for n in temps_d_execution_p_n[0]]
    index = ["Arbre binaire", "Arbre ternaire", "Arbre en série", "Arbre en étoile"]
    df_p_n = pd.DataFrame(temps_d_execution_p_n[1:], columns=colonnes, index=index)

    # Afficher le DataFrame
    titre = "Temps d'exécution de l'algorithme Padmanabhan-Natarajan pour différentes structures et tailles d'arbres"
    display(Markdown(f"## {titre}"))
    display(df_p_n)
    df_p_n.to_excel('temps_d_exécution_Padmanabhan-Natarajan.xlsx', header=True)
    # endregion

    # region Temps d'exécution Graphiques
    titres = ['Arbre série', 'Arbre binaire', 'Arbre ternaire', 'Arbre étoile']

    # Graphiques comparatifs P-N vs Naif
    for i in range(nombre_de_familles_d_arbre):
        plt.figure(figsize=(6, 4))
        n_values = range(n_debut_comp, n_fin_comp)
        naif = temps_d_execution_naif[i + 1]
        padmanabhan_natarajan = temps_d_execution_p_n_comp[i + 1, range(n_fin_comp - n_debut_comp)]

        plt.plot(n_values, naif, label='Naïf', marker='o')
        plt.plot(n_values, padmanabhan_natarajan, label='Padmanabhan Natarajan', marker='x')

        plt.xlabel('n')
        plt.ylabel('Temps (s)')
        plt.legend()
        plt.grid(which="both")

        filename = f"temps_comparaison_{titres[i].replace(' ', '_').lower()}.png"
        plt.savefig(filename, dpi=1200)
        plt.show()
        plt.close()

    # Ordre P-N avec échelle loglog
    temps_min = np.min(temps_d_execution_p_n[1:, :])
    temps_max = np.max(temps_d_execution_p_n[1:, :])
    ylim_inf = temps_min / 2
    ylim_sup = temps_max * 2

    for i in range(nombre_de_familles_d_arbre):
        plt.figure(figsize=(6, 4))
        n_values = n_valeurs_p_n
        padmanabhan_natarajan = temps_d_execution_p_n[i + 1, :]

        plt.plot(n_values, padmanabhan_natarajan, label='Padmanabhan Natarajan', marker='o')
        plt.xscale('log', base=2)
        plt.yscale('log', base=2)

        plt.xlabel('n')
        plt.ylabel('Temps (s)')
        plt.ylim(ylim_inf, ylim_sup)
        plt.grid(which="both")

        filename = f"temps_PN_loglog_{titres[i].replace(' ', '_').lower()}.png"
        plt.savefig(filename, dpi=1200)
        plt.show()
        plt.close()

    # endregion
