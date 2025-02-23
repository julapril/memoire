from itertools import chain

from faire_graphiques_bornes import faire_graphiques_bornes
from analyse_temps_execution import analyse_temps_execution
from instacart import instacart

if __name__ == '__main__':
    def k_3(n):
        return 3


    def k_div_par_3(n):
        return n // 3


    n_fin = 30
    p_uni = 0.1
    p_bi_pos = p_uni ** 2 * 5
    p_bi_neg = p_uni ** 2 / 5

    faire_graphiques_bornes(n_debut=3, n_fin=n_fin, n_saut=3, regle_k=k_3, regle_k_nom="k_3",
                            p_uni=p_uni, p_bi=p_bi_pos, p_nom="dép_pos")
    faire_graphiques_bornes(n_debut=3, n_fin=n_fin, n_saut=3, regle_k=k_div_par_3, regle_k_nom="k_div_par_3",
                            p_uni=p_uni, p_bi=p_bi_pos, p_nom="dép_pos")
    faire_graphiques_bornes(n_debut=3, n_fin=n_fin, n_saut=3, regle_k=k_3, regle_k_nom="k_3",
                            p_uni=p_uni, p_bi=p_bi_neg, p_nom="dép_nég")
    faire_graphiques_bornes(n_debut=3, n_fin=n_fin, n_saut=3, regle_k=k_div_par_3, regle_k_nom="k_div_par_3",
                            p_uni=p_uni, p_bi=p_bi_neg, p_nom="dép_nég")

    n_valeurs_p_n = list(chain(
        range(8, 16, 2),
        range(16, 32, 4),
        range(32, 64, 8),
        range(64, 129, 16)
    ))

    analyse_temps_execution(n_debut_comp=3, n_fin_comp=15, n_valeurs_p_n=n_valeurs_p_n,
                            regle_k=k_div_par_3, nb_simulation=10,
                            p_uni=p_uni, p_bi=p_bi_pos)


    def regle_k(n):
        return n // 10

    instacart(
        regle_k=regle_k
        , granularite="département"
    )

    instacart(
        regle_k=regle_k
        , granularite="allée"
    )
