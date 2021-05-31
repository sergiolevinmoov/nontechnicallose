import pandas as pd
import numpy as np
import math



def cambio_consumo(consumo_historico:pd.Series,periodo_fraude:int=None, periodo_a_comparar:int=6, cantidad_periodos:int=3):
    """
    Retorna la relacion entre el promedio de los ultimos cantidad_periodos comparado con <periodo_a_comparar>
    periodos atras. Si periodo fraunde no es null, se cosidera el periodo fraude como ultimo periodo.
    IE: si el promedio de los ultimos 2 periodos es igual al promedio de hace  7 y 8 periodos atras, se retorna 1.
    :param consumo_historico
    :param periodo_fraude:
    :param periodo_a_comparar:
    :param cantidad_periodos:
    :return:
    """
    try:
        consumo_historico.drop(labels='periodo-fraude', inplace=True)
        if periodo_fraude is None or math.isnan(periodo_fraude):
            ultimo_periodo = consumo_historico.last_valid_index()
        else:
            ultimo_periodo = int(periodo_fraude)
        ultimo_periodo_indexloc = consumo_historico.index.get_loc(ultimo_periodo) - 1
        ultimos_periodos = consumo_historico[ultimo_periodo_indexloc + 1 - cantidad_periodos:ultimo_periodo_indexloc + 1]
        avg_ultimos_periodos = np.nanmean(ultimos_periodos)
        periodos_previos = consumo_historico[ultimo_periodo_indexloc - periodo_a_comparar - cantidad_periodos + 1: ultimo_periodo_indexloc - periodo_a_comparar + 1]
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
        if periodo_fraude is None or math.isnan(periodo_fraude):
            ultimo_periodo = consumo_historico.last_valid_index()
        else:
            ultimo_periodo = int(periodo_fraude)
        ultimo_periodo_indexloc = consumo_historico.index.get_loc(ultimo_periodo)-1
        ultimos_periodos = consumo_historico[ultimo_periodo_indexloc + 1 - cantidad_periodos:ultimo_periodo_indexloc + 1]
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
        if periodo_fraude is None or math.isnan(periodo_fraude):
            ultimo_periodo = consumo_historico.last_valid_index()
        else:
            ultimo_periodo = int(periodo_fraude)
        ultimo_periodo_indexloc = consumo_historico.index.get_loc(ultimo_periodo)-1
        ultimos_periodos = consumo_historico[ultimo_periodo_indexloc + 1 - cantidad_periodos:ultimo_periodo_indexloc + 1]
        population_average = population_average[ultimo_periodo_indexloc + 1 - cantidad_periodos:ultimo_periodo_indexloc + 1]
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
        if periodo_fraude is None or math.isnan(periodo_fraude):
            ultimo_periodo = pagos_historico.last_valid_index()
        else:
            ultimo_periodo = int(periodo_fraude)
        ultimo_periodo_indexloc = pagos_historico.index.get_loc(ultimo_periodo)-1
        ultimos_periodos = pagos_historico[ultimo_periodo_indexloc + 1 - cantidad_periodos:ultimo_periodo_indexloc + 1]
        mean = np.nanmean(ultimos_periodos)
        return mean
    except Exception as e:
        print(e)

data_path:str = 'C:/work/mooving/Predictive Analytics/NTL/'
# Read  Data
fraudes = pd.read_csv(data_path+'fraude.csv')
fraudes.set_index('cuenta',inplace=True,)

consumos_csv = pd.read_csv(data_path + 'consumo.csv')
consumos_csv.set_index('cuenta')

# Variables de Consumo
consumos = consumos_csv.pivot(index='cuenta', columns='periodo', values='consumo')
consumos = consumos.merge(fraudes, left_index=True, right_index=True, how='left')
consumos['cambio-consumo']=consumos.apply(lambda row:cambio_consumo(row,row['periodo-fraude']), axis=1)
consumos['std'],consumos['mean'],consumos['coeficiente-variacion']=consumos.apply(lambda row:coefficient_of_variation(row, row['periodo-fraude']), axis=1,result_type='expand').T.values
population_mean = consumos.mean(axis=0)
consumos['correlacion-con-promedio']=consumos.apply(lambda row:correlation_with_population(row, population_mean, row['periodo-fraude']), axis=1)

# Variables de Demora de Pago
pagos_csv = pd.read_csv(data_path + 'pagos.csv')
pagos_csv.set_index('cuenta')
pagos = pagos_csv.pivot(index='cuenta', columns='periodo', values='demora')
pagos = pagos.merge(fraudes, left_index=True, right_index=True, how='left')
pagos['promedio-demora-pago']=pagos.apply(lambda row:avg_demora_pago(row, row['periodo-fraude']), axis=1)

# Datos demograficos (carenciado, localidad, categoria,
# Juntar todo
consumos = consumos.loc[:,'cambio-consumo':]
pagos = pagos.loc[:,'promedio-demora-pago':]
training_data = consumos.merge(pagos, left_index=True, right_index=True, how='left')
training_data = training_data.merge(fraudes, left_index=True, right_index=True, how='left')
training_data['fraude'] = np.where(np.any(np.isnan(training_data['periodo-fraude'])), 1, 0)
training_data.to_csv(data_path+'training.csv')
print(training_data)


