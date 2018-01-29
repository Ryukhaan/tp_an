# -*- coding: utf-8 -*-
import Sonar
import random
import copy


# Fonction qui extrait les données et renvoie un tableau contenant toutes les données structurées.
def data_extract(path_to_data):
    res = []
    with open(path_to_data, 'r') as data_file:
        # on lit la première ligne
        read_line = data_file.readline()
        # on teste si la ligne lue n'est pas vide.
        while read_line != "":
            # on met la data de la ligne dans l'array résultat.
            res.append(Sonar.SonarData(read_line.split(';')))
            # on lit la prochaine ligne.
            read_line = data_file.readline()
        data_file.close()
    # print(res)
    return res


if __name__ == '__main__':
    # --- PARAMETERS ---

    k_bornes = (1, 10)
    blocks_nbr = 5
    datafile = "sonar_data.txt"

    # end parameters

    # --- INITIALISATION ---

    # on commence par chager les données.
    all_data = data_extract(datafile)
    # on mélange nos données.
    random.shuffle(all_data)
    # on créer tout les blocks
    all_blocks = []
    n = len(all_data)
    for b in range(blocks_nbr):
        new_block = all_data[int(n * b / blocks_nbr): int(n * (b + 1) / blocks_nbr)]
        all_blocks.append(new_block)

    # on regarde si ça a fonctionné.
    for block in all_blocks:
        print(block)

    print(all_blocks[0][0])

    # On initialise le dictionnaire qui contient tout les résultats.
    final_res = {}

    # end initialisation

    # --- START MAIN ---
    print "Start main...", "Number of bases: {0}".format(blocks_nbr + 1)

    # On parcours les blocs de test.
    for b_t_i in range(blocks_nbr):
        print "--- Testing base: {0} ---".format(b_t_i)
        # on parcours les k.
        b_t = all_blocks[b_t_i]
        for k in range(k_bornes[0], k_bornes[1] + 1):
            print "Testing k='{0}'...".format(k)
            # on parcours les blocks de validations (qui doivent être différents des blocks de tests)
            for b_v_i in range(blocks_nbr):
                # On teste la différence.
                if b_v_i != b_t_i:
                    b_v = all_blocks[b_v_i]
                    # on créer une copy facilement modifiable de 'all_block'
                    all_blocks_temp = copy.copy(all_blocks)
                    # on enlève les blocks de test et de validation de cette base. On commence par le plus grand pour
                    # éviter les décalage
                    all_blocks_temp.remove(all_blocks_temp[max(b_t_i,b_v_i)])
                    all_blocks_temp.remove(all_blocks_temp[min(b_t_i,b_v_i)])
                    # on créer la base d'apprentissage
                    b_a = []
                    for a_b_t_i in all_blocks_temp:
                        for elem in a_b_t_i:
                            b_a.append(elem)

                    # -- ETAPE TEST --
                    # On arrive à l'étape de test, on initialise les paramètres de test.
                    good_result = 0.0
                    iterations = 0.0

                    # maintenant, on parcours les éléments de la base de validation.
                    for b_v_elem in b_v:
                        # on initilise l'array des k plus proche éléments.
                        k_nearest = []
                        # on parcours les éléments de la base d'apprentissage.
                        for b_a_elem in b_a:
                            if len(k_nearest) < k:
                                # Si la liste n'est pas complète, on la complète.
                                k_nearest.append((b_a_elem, b_a_elem.calc_distance(b_v_elem)))
                            else:
                                # si la liste est pleine, on calcule la distance.
                                dist = b_a_elem.calc_distance(b_v_elem)
                                replace_index = -1
                                replace_value = 0
                                # on compare la distance obtenue par rapport aux valeurs déjà présente.
                                for k_n_e_i in range(k):
                                    k_n_elem = k_nearest[k_n_e_i]
                                    # On cherche à trouver la plus grande valeur à remplacer dans les k plus proches.
                                    if dist < k_n_elem[1] and replace_value < k_n_elem[1]:
                                        replace_index = k_n_e_i
                                        replace_value = k_n_elem[1]
                                # Si on a trouvé à remplacer, on remplace.
                                if replace_index != -1:
                                    k_nearest[replace_index] = (b_a_elem, dist)
                        # une fois les k plus proches voisins trouvés, on créer le dico des classes pour la p-moyenne
                        compa_dict = {}
                        # On compte le nombre d'occurence de chaque classe.
                        compteur_dict = {}
                        for k_n_elem in k_nearest:
                            key = k_n_elem[0].get_sonar_class()
                            try:
                                compteur_dict[key] += 1
                            except KeyError:
                                compteur_dict[key] = 1
                            try:
                                compa_dict[key] += k_n_elem[1]
                            except KeyError:
                                compa_dict[key] = k_n_elem[1]

                        # On pondère maintenant la distance par le nombre d'occurence de chaque classe.
                        for key in compteur_dict.keys():
                            compa_dict[key] /= compteur_dict[key]

                        nearest = min(compa_dict.keys(),
                                      key=(lambda key1: compa_dict[key1]))  # renvoie la classe en str
                        # Si il est bien classé, on ajoute un bon résultat.
                        good_result += 1 if b_v_elem.is_same_sonar(nearest) else 0
                        # on a fait une itération.
                        iterations += 1
                    # dès qu'on a parcourut tous les b_v_elem, on note le taux de réussite.
                    final_res[(b_t_i, b_v_i, k)] = good_result / iterations

    # On obtient le dictionnaire final.
    print(final_res)
    # on cherche maintenant le meilleur k dans le dictionnaire.
    final_k = {}
    for key, value in final_res.items():
        try:
            final_k[key[2]] += value
        except KeyError:
            final_k[key[2]] = value

    max_key = max(final_k.keys(), key=(lambda key_k: final_k[key_k]))
    max_value = final_k[max_key] / (blocks_nbr ** 2)
    print "Le meilleur k pour ce jeux de données est k='{0}' avec {1}".format(max_key, max_value)
    # end main
