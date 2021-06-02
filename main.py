import pandas as pd
import numpy as np
import math
import csv

def get_ultimo_periodo_loc(historico, periodo_fraude):
    if periodo_fraude is None or math.isnan(periodo_fraude):
        ultimo_periodo = historico.last_valid_index()
    else:
        ultimo_periodo = int(periodo_fraude)
        periodo_fraude_loc = historico.index.get_loc(ultimo_periodo) - 1
        historico_fraude = historico[:periodo_fraude_loc]
        ultimo_periodo = historico_fraude.last_valid_index()
    return historico.index.get_loc(ultimo_periodo) - 1


def cambio_consumo(consumo_historico:pd.Series,periodo_fraude:int=None, periodo_a_comparar:int=6, cantidad_periodos:int=3):
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
        consumo_historico.drop(labels='periodo-fraude', inplace=True)
        ultimo_periodo_loc= get_ultimo_periodo_loc(consumo_historico, periodo_fraude)
        ultimos_periodos = consumo_historico[ultimo_periodo_loc + 1 - cantidad_periodos:ultimo_periodo_loc + 1]
        avg_ultimos_periodos = np.nanmean(ultimos_periodos)
        periodos_previos = consumo_historico[ultimo_periodo_loc - periodo_a_comparar - cantidad_periodos + 1: ultimo_periodo_loc - periodo_a_comparar + 1]
        avg_periodos_previos = np.nanmean(periodos_previos)
        return avg_ultimos_periodos/avg_periodos_previos
    except Exception as e:
        print(e)

def coefficient_of_variation (consumo_historico:pd.Series,periodo_fraude:int=None, cantidad_periodos:int=12):
    """
    Retorna la relacion entre la varianza y la media de los ultimos cantidad_periodos
    :param consumo_historico
    :param periodo_fraude:
    :param cantidad_periodos:
    :return:
    """
    try:
        consumo_historico.drop(labels='periodo-fraude', inplace=True)
        ultimo_periodo_loc= get_ultimo_periodo_loc(consumo_historico, periodo_fraude)
        ultimos_periodos = consumo_historico[ultimo_periodo_loc + 1 - cantidad_periodos:ultimo_periodo_loc + 1]
        std = np.nanstd(ultimos_periodos)
        mean = np.nanmean(ultimos_periodos)
        return std, mean, std/mean
    except Exception as e:
        print(e)


def correlation_with_population (consumo_historico:pd.Series, population_average:pd.Series, periodo_fraude:int=None, cantidad_periodos:int=12):
    """
    Retorna la relacion entre la varianza y la media de los ultimos cantidad_periodos
    :param consumo_historico
    :param periodo_fraude:
    :param cantidad_periodos:
    :return:
    """
    try:
        consumo_historico.drop(labels='periodo-fraude', inplace=True)
        ultimo_periodo_loc= get_ultimo_periodo_loc(consumo_historico, periodo_fraude)
        ultimos_periodos = consumo_historico[ultimo_periodo_loc + 1 - cantidad_periodos:ultimo_periodo_loc + 1]
        population_average = population_average[ultimo_periodo_loc + 1 - cantidad_periodos:ultimo_periodo_loc + 1]
        correlation = ultimos_periodos.corr(population_average, method='pearson')
        return correlation
    except Exception as e:
        print(e)

def avg_demora_pago (pagos_historico:pd.Series,periodo_fraude:int=None, cantidad_periodos:int=6):
    """
    Retorna la relacion entre la varianza y la media de los ultimos cantidad_periodos
    :param consumo_historico
    :param periodo_fraude:
    :param cantidad_periodos:
    :return:
    """
    try:
        pagos_historico.drop(labels='periodo-fraude', inplace=True)
        ultimo_periodo_loc = get_ultimo_periodo_loc(pagos_historico, periodo_fraude)
        ultimos_periodos = pagos_historico[ultimo_periodo_loc + 1 - cantidad_periodos:ultimo_periodo_loc + 1]
        mean = np.nanmean(ultimos_periodos)
        return mean
    except Exception as e:
        print(e)


periodo_base = 20121
data_path:str = 'C:/work/mooving/Predictive Analytics/NTL/'
# Read  Data
fraudes = pd.read_csv(data_path+'fraude.csv')
fraudes.set_index('cuenta',inplace=True)
fraudes = fraudes[fraudes['periodo-fraude'] >= periodo_base]
consumos_csv = pd.read_csv(data_path + 'consumo.csv')
#consumos_csv.set_index('cuenta',inplace=True)
consumos_csv = consumos_csv[consumos_csv['periodo'] >= periodo_base]

# Variables de Consumo
consumos = consumos_csv.pivot(index='cuenta', columns='periodo', values='consumo')
consumos = consumos.merge(fraudes, left_index=True, right_index=True, how='left')
consumos['cambio-consumo1']=consumos.apply(lambda row:cambio_consumo(row,row['periodo-fraude'],1), axis=1)
consumos['cambio-consumo3']=consumos.apply(lambda row:cambio_consumo(row,row['periodo-fraude'],3), axis=1)
consumos['std12'],consumos['mean12'],consumos['coeficiente-variacion12']=consumos.apply(lambda row:coefficient_of_variation(row, row['periodo-fraude']), axis=1,result_type='expand').T.values
consumos['std6'],consumos['mean6'],consumos['coeficiente-variacion6']=consumos.apply(lambda row:coefficient_of_variation(row, row['periodo-fraude'],6), axis=1,result_type='expand').T.values
consumos['std3'],consumos['mean3'],consumos['coeficiente-variacion3']=consumos.apply(lambda row:coefficient_of_variation(row, row['periodo-fraude'],3), axis=1,result_type='expand').T.values
population_mean = consumos.mean(axis=0)
#consumos['correlacion-con-promedio']=consumos.apply(lambda row:correlation_with_population(row, population_mean, row['periodo-fraude']), axis=1)
consumos = consumos.loc[:,'cambio-consumo1':]


# Variables de Demora de Pago
pagos_csv = pd.read_csv(data_path + 'pagos.csv')
pagos_csv = pagos_csv[pagos_csv['periodo'] >= periodo_base]
pagos = pagos_csv.pivot(index='cuenta', columns='periodo', values='demora')
pagos = pagos.merge(fraudes, left_index=True, right_index=True, how='left')
pagos['promedio-demora-pago']=pagos.apply(lambda row:avg_demora_pago(row, row['periodo-fraude']), axis=1)
#pagos['promedio-demora-pago3']=pagos.apply(lambda row:avg_demora_pago(row, row['periodo-fraude'],3), axis=1)
pagos = pagos.loc[:,'promedio-demora-pago':]

# Datos demograficos (carenciado, localidad, categoria,
# Juntar todo

training_data = consumos.merge(pagos, left_index=True, right_index=True, how='left')
training_data = training_data.merge(fraudes, left_index=True, right_index=True, how='left')
training_data['fraude'] = np.where(np.isnan(training_data['periodo-fraude']), 0, 1)
del training_data['periodo-fraude']
training_data.dropna(inplace=True)
training_data.index = training_data.index.map(str)
training_data.index = training_data.index.map(lambda i: 'C-'+i)
training_data.to_csv(data_path+'training.csv', encoding='Windows-1252',quoting= csv.QUOTE_NONNUMERIC)
print('Process Finished')


