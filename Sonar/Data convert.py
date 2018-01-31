# -*- coding: utf-8 -*-
"""
Ce fichier à pour but de transformer les données du sonnar pour l'importation avec GinNet.
"""
import csv

# Params

input_file = "sonar.data"
output_file = "sonar_data.txt"
input_numbers = 60
output_number = 1

# Parsing Params

header_separator = "$TRAIN\n\n"
data_set_separator = '$TEST'
data_block_separator = '\n\n'
output_separator = '=>'
line_separator = '\n'
input_separator = ','

# end Params


def data_parsing(datas, caracterisation):
    res = []
    # On récupère le contenu du block
    datablock_list = datas.split(data_block_separator)
    for db in datablock_list:
        if db.strip() != '':
            # On sépare caractéristique de résultat.
            data_carac_block, data_class = db.strip().split(output_separator)
            # On initilise le résultat de cette ligne, avec la caractérisation
            data_carac = [caracterisation]
            for data_carac_block_line in data_carac_block.split(line_separator):
                if data_carac_block_line.strip() != '':
                    for data_input in data_carac_block_line.strip().split(input_separator):
                        if data_input.strip() != '':
                            data_carac.append(float(data_input.strip()))
            # print data_carac, data_class.strip()
            data_carac.append(data_class.strip())
            res.append(data_carac)
            # print data_carac.__len__()
    return res


if __name__ == '__main__':

    # MAIN
    with open(input_file) as datafile:
        datafile_content = datafile.read()
        datafile.close()

        content = datafile_content.split(header_separator)[1]

        datas_training, datas_tests  = content.strip().split(data_set_separator)
        # print datas_training, '\n\n ------- \n\n', datas_tests
        training_data_list = data_parsing(datas_training, 'TRAIN')
        tests_data_list = data_parsing(datas_tests, 'TEST')

        print training_data_list, tests_data_list

        with open(output_file, 'wb') as finalfile:
            writerbuffer = csv.writer(finalfile, delimiter=';')
            for td in training_data_list:
                writerbuffer.writerow(td)
            for tstd in tests_data_list:
                writerbuffer.writerow(tstd)
