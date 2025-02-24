import kagglehub

import os
import pandas as pd
import numpy as np
from scipy.stats import kendalltau
import pyvinecopulib as pv
import matplotlib.pyplot as plt
from IPython.display import display, Markdown

from faire_graphique_copule_en_vigne import vinecop_plot
from formatter_structure import formatter_structure
from obtenir_borne_padmanabhan_natarajan import obtenir_borne_padmanabhan_natarajan
from obtenir_probabilite_independance_conditionnelle import obtenir_probabilite_independance_conditionnelle
from obtenir_borne_univariee import obtenir_borne_univariee

os.environ["PATH"] += os.pathsep + 'C:/Program Files (x86)/Graphviz12.2.1/bin/'


def instacart(regle_k, granularite = "département", n_transactions=None):
    # region Importer et formatter les données
    folder_path = kagglehub.dataset_download("yasserh/instacart-online-grocery-basket-analysis-dataset")

    csv_files = [f for f in os.listdir(folder_path) if f.endswith(".csv")]

    instacart_donnees = {}

    for file in csv_files:
        file_path = os.path.join(folder_path, file)
        name = os.path.splitext(file)[0]
        instacart_donnees[name] = pd.read_csv(file_path)

    orders = (
        instacart_donnees["order_products__train"]
        .merge(instacart_donnees["products"], how="left", on="product_id")
        .merge(instacart_donnees["aisles"], how="left", on="aisle_id")
        .merge(instacart_donnees["departments"], how="left", on="department_id")
    )

    if n_transactions is not None:
        unique_orders = orders["order_id"].unique()[:n_transactions]  # n_transactions premiers order IDs
        orders = orders[orders["order_id"].isin(unique_orders)]

    # Indicateurs d'allée
    allees = pd.crosstab(orders["order_id"], orders["aisle_id"]).clip(upper=1).to_numpy()

    # Indicateurs de département
    dep = pd.crosstab(orders["order_id"], orders["department_id"]).clip(upper=1).to_numpy()

    p_allees = allees.mean(axis=0)
    p_dep = dep.mean(axis=0)

    num_allees = allees.shape[1]
    tau_allees = np.zeros((num_allees, num_allees))
    for i in range(num_allees):
        for j in range(i + 1, num_allees):
            tau_allees[i, j], _ = kendalltau(allees[:, i], allees[:, j])
            tau_allees[j, i] = tau_allees[i, j]
    allees = pd.DataFrame(allees)
    tau_allees = pd.DataFrame(tau_allees)

    num_dep = dep.shape[1]
    tau_dep = np.zeros((num_dep, num_dep))
    for i in range(num_dep):
        for j in range(i + 1, num_dep):
            tau_dep[i, j], _ = kendalltau(dep[:, i], dep[:, j])
            tau_dep[j, i] = tau_dep[i, j]
    dep = pd.DataFrame(dep)
    tau_dep = pd.DataFrame(tau_dep)
    # endregion

    # region Copule en vigne
    if granularite == "département":
        noms = instacart_donnees['departments']['department']
        data = dep.to_numpy()
        p = p_dep
    elif granularite == "allée":
        noms = instacart_donnees['aisles']['aisle']
        data = allees.to_numpy()
        p = p_allees
    else:
        return("granularité invalide")


    def bernoulli_fdr(u, proba_succes):
        return np.where(
            u == -1, 0,
            np.where(u == 0, 1 - proba_succes, 1)
        )

    u_disc = np.hstack(
        (bernoulli_fdr(data, p), bernoulli_fdr(data - 1, p))
    )  # Pseudo_observations discrètes

    # Puisque nos données originales sont discrètes (Bernoulli),
    # nous devons indiquer au modèle que chaque variable est discrète.
    # Ici, nous créons une liste contenant 'd' (pour discret) pour chaque colonne.
    var_types = ['d'] * data.shape[1]

    # Nous personnalisons les paramètres de l'ajustement.
    controls = pv.FitControlsVinecop(family_set=[pv.BicopFamily.gaussian], trunc_lvl=1)

    # Modéliser la copule en vigne (vine copula) avec les pseudo-observations.
    # Le constructeur Vinecop accepte les pseudo-observations, les paramètres d'ajustement
    # et les types de variables.
    copule_vigne = pv.Vinecop.from_data(u_disc, controls=controls, var_types=var_types)

    # Afficher un résumé du modèle
    print(str(copule_vigne))
    structure = formatter_structure(copule_vigne)
    d = structure['d']
    sommets = structure['sommets']
    index = structure['index']
    aretes = structure['aretes']


    # Seulement la granularité département permet une visualisation agréable
    if granularite == "département":
        vinecop_plot(copule_vigne, add_edge_labels=False, vars_names=noms)
        filename = "graphiques/copule_vigne_arbre.png"
        plt.savefig(filename, dpi=1200)
        plt.show()

    # endregion

    # region Probabilités

    # Nombre de défauts
    k = regle_k(d)

    p = [p[index.get(sommet)] for sommet in sommets]
    pp = [np.mean(data[:, index.get(i)] * data[:, index.get(j)]) for i, j in aretes]

    prob_empirique = np.mean(data.sum(axis=1) >= k)

    prob_ind_cond = obtenir_probabilite_independance_conditionnelle(sommets=sommets, aretes=aretes,
                                                                    probabilites_univariees=p,
                                                                    probabilites_bivariees=pp,
                                                                    nombre_de_defauts=k)

    borne_inf_ind = obtenir_borne_univariee(probabilites_univariees=p,
                                            nombre_de_defauts=k,
                                            minimiser=True)
    borne_sup_ind = obtenir_borne_univariee(probabilites_univariees=p,
                                            nombre_de_defauts=k,
                                            minimiser=False)

    borne_inf = obtenir_borne_padmanabhan_natarajan(sommets=sommets, aretes=aretes,
                                                    probabilites_univariees=p, probabilites_bivariees=pp,
                                                    nombre_de_defauts=k, minimiser=True)
    borne_sup = obtenir_borne_padmanabhan_natarajan(sommets=sommets, aretes=aretes,
                                                    probabilites_univariees=p, probabilites_bivariees=pp,
                                                    nombre_de_defauts=k, minimiser=False)

    # Création d'un DataFrame
    resultats = pd.DataFrame({
        "Méthode": ["Probabilité empirique", "Probabilité indépendance conditionnelle",
                    "Borne inf indépendance", "Borne sup indépendance",
                    "Borne inf Padmanabhan-Natarajan", "Borne sup Padmanabhan-Natarajan"],
        "Valeur": [prob_empirique, prob_ind_cond, borne_inf_ind, borne_sup_ind, borne_inf, borne_sup]
    })

    # Affichage en Markdown
    display(Markdown("## Résultats des calculs"))
    display(resultats)

    # Exportation vers Excel
    resultats.to_excel(f"tableaux/resultats_instacart_{granularite}.xlsx", index=False, header=True)
    # endregion