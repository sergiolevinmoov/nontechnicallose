import pandas as pd
import numpy as np
import math
import csv



def drop_columnas_no_consumo(consumo_historico):
    consumo_historico.drop(labels=['periodo-fraude','fraude','AGUSCODPOS'], inplace=True)
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


def correlation_with_population (consumo_historico:pd.Series, population_average, periodo_fraude=None, cantidad_periodos=6, categoria='RES'):
    """
    Retorna la relacion entre la varianza y la media de los ultimos cantidad_periodos
    :param consumo_historico
    :param periodo_fraude:
    :param cantidad_periodos:
    :return:
    """
    try:
        is_fraude = ~np.isnan(periodo_fraude)
        if categoria == 'RES':
            localidad_average = population_average.loc[int(consumo_historico['AGUSCODPOS'])]
        else:
            localidad_average = population_average
        consumo_historico = drop_columnas_no_consumo(consumo_historico)
        ultimo_periodo_loc= get_ultimo_periodo_loc(consumo_historico, periodo_fraude,is_fraude)
        ultimos_periodos = consumo_historico[ultimo_periodo_loc + 1 - cantidad_periodos:ultimo_periodo_loc + 1]
        localidad_average_periodo = localidad_average[ultimo_periodo_loc + 1 - cantidad_periodos:ultimo_periodo_loc + 1]
        correlation = ultimos_periodos.corr(localidad_average_periodo, method='pearson')
        return correlation
    except Exception as e:
        print(e)

def add_correlacion_cp(consumos: pd.DataFrame, categoria):
    if categoria == 'RES':
        population_mean = consumos.groupby(['AGUSCODPOS']).mean()
    else:
        population_mean = consumos.mean()
    consumos['correlacion-con-promedio6'] = consumos.apply(lambda row: correlation_with_population(row, population_mean, row['periodo-fraude'], 6, categoria), axis=1)
    #result['correlacion-con-promedio3'] = result.apply(lambda row: correlation_with_population(row, population_mean, row['periodo-fraude'], 3), axis=1)
    return consumos

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

def generate_variables_consumos(consumos, fraudes, merge_type:str= 'inner'):
    if fraudes is not None:
        consumos = consumos.merge(fraudes, left_index=True, right_index=True, how=merge_type)
        consumos['fraude'] = np.where(np.isnan(consumos['periodo-fraude']), 0, 1)
    else:
        consumos['fraude'] = np.nan
        consumos['periodo-fraude'] = np.nan
    result = consumos.copy()
    ##Dont Uncomment consumos['cambio-consumo1-1'] = consumos.apply(lambda row: cambio_consumo(row, row['periodo-fraude'], 1, 1), axis=1)
    result['cambio-consumo1-6'] = consumos.apply(lambda row: cambio_consumo(row,row['fraude'], row['periodo-fraude'], 6, 1), axis=1)
    ##consumos['cambio-consumo3'] = consumos.apply(lambda row: cambio_consumo(row,row['fraude'], row['periodo-fraude'], 3), axis=1)
    result['cambio-consumo3-6'] = consumos.apply(lambda row: cambio_consumo(row,row['fraude'], row['periodo-fraude'], 6,3), axis=1)
    result['cambio-consumo6-6'] = consumos.apply(lambda row: cambio_consumo(row, row['fraude'], row['periodo-fraude'], 6, 6), axis=1)
    ##consumos['std12'], consumos['mean12'], consumos['coeficiente-variacion12'] = consumos.apply(lambda row: coefficient_of_variation(row,row['fraude'], row['periodo-fraude']), axis=1, result_type='expand').T.values
    result['std6'], result['mean6'], result['coeficiente-variacion6'] = consumos.apply(
        lambda row: coefficient_of_variation(row, row['fraude'],row['periodo-fraude'], 6), axis=1, result_type='expand').T.values
    result['std3'], result['mean3'], result['coeficiente-variacion3'] = consumos.apply(
        lambda row: coefficient_of_variation(row,row['fraude'], row['periodo-fraude'], 3), axis=1, result_type='expand').T.values
    result['std12'], result['mean12'], result['coeficiente-variacion12'] = consumos.apply(
        lambda row: coefficient_of_variation(row, row['fraude'], row['periodo-fraude'], 3), axis=1,result_type='expand').T.values
    consumos = add_correlacion_cp(consumos,categoria)
    result['correlacion-con-promedio6'] = consumos['correlacion-con-promedio6']
    #result['correlacion-con-promedio3'] = consumos['correlacion-con-promedio3']
    result.to_csv(data_path + 'consumos-training.csv', encoding='Windows-1252', quoting=csv.QUOTE_NONNUMERIC)
    result = result.loc[:, 'cambio-consumo1-6':]
    return result

def generate_variables_pagos(pagos, fraudes, merge_type:str= 'inner'):
    pagos = pagos.pivot(index='cuenta', columns='periodo', values='demora')
    pagos = pagos.merge(fraudes, left_index=True, right_index=True, how=merge_type)
    pagos['promedio-demora-pago'] = pagos.apply(lambda row: avg_demora_pago(row,row['fraude'], row['periodo-fraude']), axis=1)
    # pagos['promedio-demora-pago3']=pagos.apply(lambda row:avg_demora_pago(row,row['fraude'], row['periodo-fraude'],3), axis=1)
    pagos.to_csv(data_path + 'pagos-training.csv', encoding='Windows-1252', quoting=csv.QUOTE_NONNUMERIC)
    pagos = pagos.loc[:, 'promedio-demora-pago':]
    return pagos

def merge_all(consumos, pagos, fraudes, ipf, periodos, output_file, for_scoring:False):
    training_data=consumos
    if fraudes is not None:
        training_data = training_data.merge(fraudes, left_index=True, right_index=True, how='left')
        training_data['fraude'] = np.where(np.isnan(training_data['periodo-fraude']), 0, 1)
        del training_data['periodo-fraude']
    if ipf is not None:
        training_data = training_data.merge(ipf, left_index=True, right_index=True, how='left')
    #training_data['fraude'] = np.where(np.isnan(training_data['periodo-fraude']), 0, 1)
    training_data.dropna(inplace=True)
    training_data = training_data[~training_data.index.duplicated(keep='first')]
    #training_data = training_data.merge(periodos, left_index=True, right_index=True, how='left')
    training_data.index = training_data.index.map(str)
    training_data.index = training_data.index.map(lambda i: 'C-' + i)
    training_data.to_csv(data_path + output_file, encoding='Windows-1252', quoting=csv.QUOTE_NONNUMERIC)

def get_periodos(consumo_historico:pd.Series,  periodo_fraude:int=None,cantidad_periodos:int=6):
    try:
        consumo_historico.drop(labels=['periodo-fraude'], inplace=True)
        ultimo_periodo_loc= get_ultimo_periodo_loc(consumo_historico, periodo_fraude, ~np.isnan(periodo_fraude))
        ultimos_periodos = consumo_historico[ultimo_periodo_loc + 1 - cantidad_periodos:ultimo_periodo_loc + 1]
        if ultimos_periodos.size == cantidad_periodos:
            ultimos_periodos.index = list(range(1, cantidad_periodos+1))
            return ultimos_periodos
        else:
            return pd.Series(list(range(1, cantidad_periodos)))
    except Exception as e:
        return pd.Series(list(range(1, cantidad_periodos)))

def generate_periodos(consumos_csv:pd.Series, fraudes):
    consumos = consumos_csv.pivot(index='cuenta', columns='periodo', values='consumo')
    consumos = consumos.merge(fraudes, left_index=True, right_index=True, how='left')
    #consumos = consumos.drop(consumos[np.isnan(consumos['periodo-fraude'])].index)
    rows_list = []
    for index, c in consumos.iterrows():
        periodo = get_periodos(c, c['periodo-fraude'], 12)
        rows_list.append(periodo)
    periodos = pd.concat(rows_list, axis=1)
    periodos = periodos.transpose()
    #consumos.apply(lambda row: get_periodos(row, row['periodo-fraude'], 6), axis=1, result_type='expand')
    periodos.to_csv(data_path + 'periodos.csv', encoding='Windows-1252', quoting=csv.QUOTE_NONNUMERIC)
    return periodos


categoria = 'RES'
data_path:str = 'C:/work/mooving/Predictive Analytics/NTL/noreclamos/'
# Read  Data
fraudes = pd.read_csv(data_path+'fraude.csv')
fraudes.set_index('cuenta',inplace=True)


ipf = pd.read_csv(data_path + 'ipf.csv')
ipf = ipf.replace(np.nan, '', regex=True)
ipf.set_index('cuenta', inplace=True)
#ipf = ipf[['categoria','AGUSCODPOS']]

consumos_fraude = pd.read_csv(data_path + 'consumo_fraude.csv')
consumos_nofraude = pd.read_csv(data_path + 'consumo_nofraude.csv')
consumos_csv = pd.concat([consumos_fraude,consumos_nofraude])
consumos_csv = consumos_csv.pivot(index='cuenta', columns='periodo', values='consumo')
cat_cp = ipf[['categoria','AGUSCODPOS']]
consumos_csv = training_data = consumos_csv.merge(cat_cp, left_index=True, right_index=True, how='left')
consumos_csv = consumos_csv.loc[consumos_csv['categoria'] == categoria]
del consumos_csv['categoria']
#periodos = generate_periodos(consumos_csv,fraudes)
consumos = generate_variables_consumos(consumos_csv, fraudes, 'left')
#pagos_csv = pd.read_csv(data_path + 'pagos.csv')
#pagos = generate_variables_pagos(pagos_csv,fraudes)
merge_all(consumos, None, fraudes, ipf, None, 'training'+categoria+'.csv', False)
'''

consumos_scoring_csv = pd.read_csv(data_path + 'consumo_scoring.csv')
consumos_scoring = generate_variables_consumos(consumos_scoring_csv,None,'left')
ipf = pd.read_csv(data_path + 'ipf_scoring.csv')
ipf = ipf.replace(np.nan, '', regex=True)
ipf.set_index('cuenta', inplace=True)
##pagos_scoring_csv = pd.read_csv(data_path + 'pagos_scoring.csv')
##pagos_scoring = generate_pagos(pagos_scoring_csv,fraudes,'left')
merge_all(consumos_scoring, None, None, ipf, None, 'scoring_data.csv', True)
'''
print('Process Finished')


