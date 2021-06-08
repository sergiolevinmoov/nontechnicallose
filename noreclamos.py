import pandas as pd
import numpy as np
import math
import csv



def drop_columnas_no_consumo(consumo_historico):
    consumo_historico.drop(labels=['periodo-fraude','fraude'], inplace=True)
    return consumo_historico

def get_ultimo_periodo_loc(historico, periodo_fraude, is_fraude):
    if not is_fraude or np.isnan(is_fraude):
        ultimo_periodo = historico.last_valid_index()
    else:
        ultimo_periodo = int(periodo_fraude)
        periodo_fraude_loc = historico.index.get_loc(ultimo_periodo)
        historico_fraude = historico[:periodo_fraude_loc]
        ultimo_periodo = historico_fraude.last_valid_index()
    return historico.index.get_loc(ultimo_periodo)



def cambio_consumo(consumo_historico:pd.Series, is_fraude, periodo_fraude:int=None, periodo_a_comparar:int=6, cantidad_periodos:int=3):
    """
    Retorna la relacion entre el promedio de los ultimos cantidad_periodos comparado con <periodo_a_comparar>
    periodos atras. Si periodo fraunde no es null, se cosidera el periodo fraude como ultimo periodo.
    IE: si el promedio de los ultimos 2 periodos es igual al promedio de hace  7 y 8 periodos atras, se retorna 1.
    :param consumo_historico
    :param periodo_fraude:
    :param periodo_a_comparar: Cuantos periodos atras
    :param cantidad_periodos:
    :return:
    """
    try:
        consumo_historico = drop_columnas_no_consumo(consumo_historico)
        ultimo_periodo_loc= get_ultimo_periodo_loc(consumo_historico, periodo_fraude, is_fraude)
        ultimos_periodos = consumo_historico[ultimo_periodo_loc + 1 - cantidad_periodos:ultimo_periodo_loc + 1]
        avg_ultimos_periodos = np.nanmean(ultimos_periodos)
        periodos_previos = consumo_historico[ultimo_periodo_loc - periodo_a_comparar - cantidad_periodos + 1: ultimo_periodo_loc - periodo_a_comparar + 1]
        avg_periodos_previos = np.nanmean(periodos_previos)
        return avg_ultimos_periodos/avg_periodos_previos
    except Exception as e:
        print(e)

def coefficient_of_variation (consumo_historico:pd.Series,is_fraude,periodo_fraude:int=None, cantidad_periodos:int=12):
    """
    Retorna la relacion entre la varianza y la media de los ultimos cantidad_periodos
    :param consumo_historico
    :param periodo_fraude:
    :param cantidad_periodos:
    :return:
    """
    try:
        consumo_historico = drop_columnas_no_consumo(consumo_historico)
        ultimo_periodo_loc= get_ultimo_periodo_loc(consumo_historico, periodo_fraude,is_fraude)
        ultimos_periodos = consumo_historico[ultimo_periodo_loc + 1 - cantidad_periodos:ultimo_periodo_loc + 1]
        std = np.nanstd(ultimos_periodos)
        mean = np.nanmean(ultimos_periodos)
        return std, mean, std/mean
    except Exception as e:
        return None,None,None


def correlation_with_population (consumo_historico:pd.Series, population_average:pd.Series, is_fraude, periodo_fraude:int=None, cantidad_periodos:int=12):
    """
    Retorna la relacion entre la varianza y la media de los ultimos cantidad_periodos
    :param consumo_historico
    :param periodo_fraude:
    :param cantidad_periodos:
    :return:
    """
    try:
        consumo_historico = drop_columnas_no_consumo(consumo_historico)
        ultimo_periodo_loc= get_ultimo_periodo_loc(consumo_historico, periodo_fraude,is_fraude)
        ultimos_periodos = consumo_historico[ultimo_periodo_loc + 1 - cantidad_periodos:ultimo_periodo_loc + 1]
        population_average = population_average[ultimo_periodo_loc + 1 - cantidad_periodos:ultimo_periodo_loc + 1]
        correlation = ultimos_periodos.corr(population_average, method='pearson')
        return correlation
    except Exception as e:
        print(e)

def avg_demora_pago (pagos_historico:pd.Series,is_fraude, periodo_fraude:int=None, cantidad_periodos:int=6):
    """
    Retorna la relacion entre la varianza y la media de los ultimos cantidad_periodos
    :param consumo_historico
    :param periodo_fraude:
    :param cantidad_periodos:
    :return:
    """
    try:
        pagos_historico = drop_columnas_no_consumo(pagos_historico)
        ultimo_periodo_loc = get_ultimo_periodo_loc(pagos_historico, periodo_fraude,is_fraude)
        ultimos_periodos = pagos_historico[ultimo_periodo_loc + 1 - cantidad_periodos:ultimo_periodo_loc + 1]
        mean = np.nanmean(ultimos_periodos)
        return mean
    except Exception as e:
        print(e)

def generate_consumos(consumos_csv, fraudes, merge_type:str='inner'):
    consumos = consumos_csv.pivot(index='cuenta', columns='periodo', values='consumo')
    if fraudes is not None:
        consumos = consumos.merge(fraudes, left_index=True, right_index=True, how=merge_type)
        consumos['fraude'] = np.where(np.isnan(consumos['periodo-fraude']), 0, 1)
    else:
        consumos['fraude'] = np.nan
        consumos['periodo-fraude'] = np.nan
    ##Dont Uncomment consumos['cambio-consumo1-1'] = consumos.apply(lambda row: cambio_consumo(row, row['periodo-fraude'], 1, 1), axis=1)
    consumos['cambio-consumo1-6'] = consumos.apply(lambda row: cambio_consumo(row,row['fraude'], row['periodo-fraude'], 1, 6), axis=1)
    ##consumos['cambio-consumo3'] = consumos.apply(lambda row: cambio_consumo(row,row['fraude'], row['periodo-fraude'], 3), axis=1)
    consumos['cambio-consumo6'] = consumos.apply(lambda row: cambio_consumo(row,row['fraude'], row['periodo-fraude'], 6), axis=1)
    try:
        ##consumos['std12'], consumos['mean12'], consumos['coeficiente-variacion12'] = consumos.apply(lambda row: coefficient_of_variation(row,row['fraude'], row['periodo-fraude']), axis=1, result_type='expand').T.values
        consumos['std6'], consumos['mean6'], consumos['coeficiente-variacion6'] = consumos.apply(
            lambda row: coefficient_of_variation(row, row['fraude'],row['periodo-fraude'], 6), axis=1, result_type='expand').T.values
        consumos['std3'], consumos['mean3'], consumos['coeficiente-variacion3'] = consumos.apply(
            lambda row: coefficient_of_variation(row,row['fraude'], row['periodo-fraude'], 3), axis=1, result_type='expand').T.values
    except Exception as e :
        print(e)
    population_mean = consumos.mean(axis=0)
    consumos.to_csv(data_path + 'consumos-training.csv', encoding='Windows-1252', quoting=csv.QUOTE_NONNUMERIC)
    # consumos['correlacion-con-promedio']=consumos.apply(lambda row:correlation_with_population(row, population_mean, row['periodo-fraude']), axis=1)
    consumos = consumos.loc[:, 'cambio-consumo1-6':]
    return consumos

def generate_pagos(pagos, fraudes,merge_type:str='inner'):
    pagos = pagos.pivot(index='cuenta', columns='periodo', values='demora')
    pagos = pagos.merge(fraudes, left_index=True, right_index=True, how=merge_type)
    pagos['promedio-demora-pago'] = pagos.apply(lambda row: avg_demora_pago(row,row['fraude'], row['periodo-fraude']), axis=1)
    # pagos['promedio-demora-pago3']=pagos.apply(lambda row:avg_demora_pago(row,row['fraude'], row['periodo-fraude'],3), axis=1)
    pagos.to_csv(data_path + 'pagos-training.csv', encoding='Windows-1252', quoting=csv.QUOTE_NONNUMERIC)
    pagos = pagos.loc[:, 'promedio-demora-pago':]
    return pagos

def merge_all(consumos, pagos, fraudes, ipf, output_file, for_scoring:False):
    training_data=consumos
    #training_data = consumos.merge(pagos, left_index=True, right_index=True, how='left')
    if fraudes is not None:
        training_data = training_data.merge(fraudes, left_index=True, right_index=True, how='left')
        training_data['fraude'] = np.where(np.isnan(training_data['periodo-fraude']), 0, 1)
        del training_data['periodo-fraude']
    #training_data = training_data.merge(ipf, left_index=True, right_index=True, how='left')
    # training_data['fraude'] = np.where(np.isnan(training_data['periodo-fraude']), 0, 1)
    training_data.dropna(inplace=True)
    training_data.index = training_data.index.map(str)
    training_data.index = training_data.index.map(lambda i: 'C-' + i)
    training_data.to_csv(data_path + output_file, encoding='Windows-1252', quoting=csv.QUOTE_NONNUMERIC)

def get_periodos(consumo_historico:pd.Series,  periodo_fraude:int=None,cantidad_periodos:int=6):
    try:
        consumo_historico.drop(labels=['periodo-fraude'], inplace=True)
        ultimo_periodo_loc= get_ultimo_periodo_loc(consumo_historico, periodo_fraude, True)
        ultimos_periodos = consumo_historico[ultimo_periodo_loc + 1 - cantidad_periodos:ultimo_periodo_loc + 1]
        if ultimos_periodos.size == 6:
            ultimos_periodos.index = [1,2,3,4,5,6]
            return ultimos_periodos
        else:
            return pd.Series([np.nan, np.nan, np.nan, np.nan, np.nan, np.nan])
    except Exception as e:
        return pd.Series([np.nan, np.nan, np.nan, np.nan, np.nan, np.nan])

def generate_periodos(consumos:pd.Series, f):
    consumos = consumos_csv.pivot(index='cuenta', columns='periodo', values='consumo')
    consumos = consumos.merge(f, left_index=True, right_index=True, how='left')
    consumos = consumos.drop(consumos[np.isnan(consumos['periodo-fraude'])].index)
    rows_list = []
    for index, c in consumos.iterrows():
        periodo = get_periodos(c, c['periodo-fraude'], 6)
        rows_list.append(periodo)
    periodos = pd.concat(rows_list, axis=1)
    periodos = periodos.transpose()
    #consumos.apply(lambda row: get_periodos(row, row['periodo-fraude'], 6), axis=1, result_type='expand')
    periodos.to_csv(data_path + 'periodos.csv', encoding='Windows-1252', quoting=csv.QUOTE_NONNUMERIC)


#periodo_base = 20121
data_path:str = 'C:/work/mooving/Predictive Analytics/NTL/noreclamos/'
# Read  Data
fraudes = pd.read_csv(data_path+'fraude.csv')
fraudes.set_index('cuenta',inplace=True)
#ipf = pd.read_csv(data_path + 'ipf.csv')
#ipf = ipf.replace(np.nan, '', regex=True)
#ipf.set_index('cuenta', inplace=True)

consumos_fraude = pd.read_csv(data_path + 'consumo_fraude.csv')
consumos_nofraude = pd.read_csv(data_path + 'consumo_nofraude.csv')
consumos_csv = pd.concat([consumos_fraude,consumos_nofraude])
#generate_periodos(consumos_fraude,fraudes)
consumos = generate_consumos(consumos_csv,fraudes,'left')
##pagos_csv = pd.read_csv(data_path + 'pagos.csv')
##pagos = generate_pagos(pagos_csv,fraudes)
merge_all(consumos, None, fraudes, None, 'training.csv', False)

#consumos_scoring_csv = pd.read_csv(data_path + 'consumo_scoring.csv')
#consumos_scoring = generate_consumos(consumos_scoring_csv,None,'left')
##pagos_scoring_csv = pd.read_csv(data_path + 'pagos_scoring.csv')
##pagos_scoring = generate_pagos(pagos_scoring_csv,fraudes,'left')
#merge_all(consumos_scoring, None, None, None, 'scoring_data.csv', True)

print('Process Finished')


