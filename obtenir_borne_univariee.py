def obtenir_borne_univariee(probabilites_univariees,
                            nombre_de_defauts,
                            minimiser=False):
    # probabilites_univariees doit être une liste de flottants
    #   entre 0 et 1, de même longueur que sommets
    # nombre_de_defauts doit être un entier plus petit
    #   ou égal au nombre de sommets
    # minimiser doit être une valeur booléenne, False pour
    #   la borne supérieure et True pour la borne inférieure
    n = len(probabilites_univariees)
    if minimiser:
        minimiser_probabilites_univariees = \
            [1 - prob for prob in probabilites_univariees]
        minimiser_nombre_de_defauts = (
                n - nombre_de_defauts + 1)
        return \
            (1 -
                obtenir_borne_univariee(
                    probabilites_univariees=minimiser_probabilites_univariees,
                    nombre_de_defauts=minimiser_nombre_de_defauts,
                    minimiser=False)
             )

    p = sorted(probabilites_univariees)

    probas = [sum(p[0:n - r]) / (nombre_de_defauts - r)
              for r in range(nombre_de_defauts)]
    probas.append(1)
    prob = min(probas)

    return prob
