import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 

def graphDurations(title,dataset,chars):

    if chars:
        labels = ['0-60 chars', '60-240 chars', '240-360 chars', '360-480 chars', '480-600 chars', '600 chars+']
    else:
        labels = ['0-1 min', '1-4 min', '4-6 min', '6-8 min', '8-10 min', '10 min+']

    bins = [0, 60, 240, 360, 480, 600, float('inf')]
    dataset['duracion_rango'] = pd.cut(dataset['duracion'], bins=bins, labels=labels, right=False)

    counts = dataset['duracion_rango'].value_counts().sort_index()

    # Calcular el porcentaje que representa cada categoría dentro del dataset total

    total_calls = len(dataset)

    percentages = (counts / total_calls) * 100

    plt.figure(figsize=(10, 6))

    bars = plt.bar(counts.index, counts.values)
    if chars:
        plt.xlabel('Duración de interacciones')
        plt.ylabel('Cantidad de letras')
    else:
        plt.xlabel('Duración de llamada')
        plt.ylabel('Cantidad de llamadas')

    plt.title(title)

    plt.xticks(rotation=45)

    # Agregar etiquetas con los porcentajes en cada barra

    for bar, percentage in zip(bars, percentages):

        plt.text(bar.get_x() + bar.get_width() / 2, 

                bar.get_height(), 

                f'{percentage:.1f}%',

                ha='center', va='bottom', fontsize=12, color='black')

    plt.show()

def combine(dataset):
    dataset['Telefono_final'] = (
    dataset['Telefono']
    .combine_first(dataset['Phone'])
    .combine_first(dataset['Movil'])
    .combine_first(dataset['Mobile'])
    )


    dataset['Telefono_final'] = dataset['Telefono_final'].astype(str)
    dataset['Telefono_final'] = dataset['Telefono_final'].str.replace(r'\.0$', '', regex=True)

    # Sobrescribir la columna con los últimos 6 dígitos
    dataset['Telefono_final'] = dataset['Telefono_final'].str[-7:]

def filterBeforeCreatedTime(dataset):
    dataset["fechaHora"] = pd.to_datetime(dataset["fechaHora"],format="mixed")

    dataset["Created Time"] = pd.to_datetime(dataset["Created Time"],format="mixed")

    # Filtrar registros donde "fechaHora" esté por debajo de "Created Time"

    efectivasFiltrados = dataset[dataset["fechaHora"] <= dataset["Created Time"]]


    return efectivasFiltrados

def successRate(dataset):
    # Contar reuniones realizadas y totales por agente
    success_count = dataset[dataset["Estado Reunión"] == "REALIZADO"].groupby("agente").size()

    total_count = dataset.groupby("agente").size()

    # Calcular el success rate por agente
    success_rate = (success_count / total_count).fillna(0) * 100

    # Crear un DataFrame con los resultados
    success_rate_df = pd.DataFrame({

        "Total": total_count,

        "Visitas": success_count,

        "Success Rate (%)": success_rate

    }).reset_index()

    # Calcular estadísticas de duración por agente
    duracion_stats_agente = dataset.groupby("agente")["duracion"].agg(["mean", "median", lambda x: x.mode().iloc[0] if not x.mode().empty else None])


    # Renombrar columnas
    duracion_stats_agente.columns = ["Duración Promedio", "Duración Mediana", "Duración Moda"]


    # Unir con el dataframe de success rate
    successRate = success_rate_df.merge(duracion_stats_agente, on="agente", how="left")

    return successRate

def hipotesis(datasetAgenda,datasetSinAgenda,minimosAgenda,minimosSinAgenda):
    datasetAgenda["hipotesis"] = np.where(
    ((datasetAgenda["weekend"] == "Saturday") & (datasetAgenda["lead_created"].dt.hour >= 18)) |
    ((datasetAgenda["weekend"] == "Sunday") & (datasetAgenda["lead_created"].dt.hour <= 14)),
    "Dentro del rango horario",
    "Fuera del rango horario"
    )

    datasetSinAgenda["hipotesis"] = np.where(
        ((datasetSinAgenda["weekend"] == "Saturday") & (datasetSinAgenda["lead_created"].dt.hour >= 18)) |
        ((datasetSinAgenda["weekend"] == "Sunday") & (datasetSinAgenda["lead_created"].dt.hour <= 14)),
        "Dentro del rango horario",
        "Fuera del rango horario"
    )
    cantidadUsuariosPerdidos = datasetAgenda["hipotesis"].value_counts(sort=False) + datasetSinAgenda["hipotesis"].value_counts(sort=False)
    print(cantidadUsuariosPerdidos.values[1])

    total = len(minimosAgenda) + len(minimosSinAgenda)

    print(total)

    porcentaje = (cantidadUsuariosPerdidos.values[1] / total) * 100
    print(f"Perderiamos un {porcentaje:.2f}% de las entradas")

    conAgendaVisita = datasetAgenda[datasetAgenda['Estado Reunión'] == 'REALIZADO'].copy()
    conAgendaNoVisita = datasetAgenda[datasetAgenda['Estado Reunión'] != 'REALIZADO'].copy()

    # Contar las ocurrencias en cada grupo para ambos conjuntos
    countAgendaVisita = conAgendaVisita['hipotesis'].value_counts(sort=False)
    countAgendaNoVisita = conAgendaNoVisita['hipotesis'].value_counts(sort=False)
    countNoAgenda = datasetSinAgenda['hipotesis'].value_counts(sort=False)

    dataFindeHipotesis = pd.DataFrame({
        'Con Agenda (VISITO)': countAgendaVisita,
        'Con Agenda (NO VISITO)': countAgendaNoVisita,
        'Sin Agenda': countNoAgenda
    })

    # Graficar el histograma apilado
    dataFindeHipotesis.plot(kind='bar', stacked=True, figsize=(12, 6), edgecolor='black')
    plt.title('Distribución dentro y fuera del periodo de 18hs del sabado y 14hs del domingo', fontsize=16)
    plt.xlabel('Horario', fontsize=12)
    plt.ylabel('Frecuencia', fontsize=12)
    plt.xticks(rotation=0)
    plt.legend(title="Estado", fontsize=10)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.show()

    dataFindeHipotesis['Total'] = dataFindeHipotesis['Con Agenda (VISITO)'] + dataFindeHipotesis['Con Agenda (NO VISITO)'] + dataFindeHipotesis['Sin Agenda']

    # Calcular los porcentajes
    dataFindeHipotesis['% Con Agenda'] = ((dataFindeHipotesis['Con Agenda (VISITO)'] + dataFindeHipotesis['Con Agenda (NO VISITO)']) / dataFindeHipotesis['Total']) * 100
    dataFindeHipotesis['% Sin Agenda'] = (dataFindeHipotesis['Sin Agenda'] / dataFindeHipotesis['Total']) * 100

    dataFindeHipotesis["totalAgenda"] = dataFindeHipotesis['Con Agenda (VISITO)'] + dataFindeHipotesis['Con Agenda (NO VISITO)']
    dataFindeHipotesis[" % visitaron" ] = (dataFindeHipotesis['Con Agenda (VISITO)'] /dataFindeHipotesis["Total"] )*100
    dataFindeHipotesis[" % NO visitaron" ] = (dataFindeHipotesis['Con Agenda (NO VISITO)'] /dataFindeHipotesis["Total"] )*100

    print(dataFindeHipotesis)


def analizar_horarios_por_pares(datasetAgenda, datasetSinAgenda):
    # Asegurar que 'lead_created' es datetime
    datasetAgenda["lead_created"] = pd.to_datetime(datasetAgenda["lead_created"])
    datasetSinAgenda["lead_created"] = pd.to_datetime(datasetSinAgenda["lead_created"])

    # Definir los pares de días
    pares_dias = [("Monday", "Tuesday"), ("Tuesday", "Wednesday"), ("Wednesday", "Thursday"), 
                  ("Thursday", "Friday"), ("Friday", "Saturday"), ("Saturday", "Sunday"), ("Sunday", "Monday")]

    # Mapear horarios de interés por día
    horarios_interes = {
        "Monday": (18, 23),
        "Tuesday": (18, 23),
        "Wednesday": (18, 23),
        "Thursday": (18, 23),
        "Friday": (18, 23),
        "Saturday": (18, 23),
        "Sunday": (0, 14)
    }

    # Crear gráficos para cada par de días
    for dia1, dia2 in pares_dias:
        # Filtrar registros de los dos días
        subsetAgenda = datasetAgenda[(datasetAgenda["lead_created"].dt.strftime("%A") == dia1) | 
                                     (datasetAgenda["lead_created"].dt.strftime("%A") == dia2)].copy()
        subsetSinAgenda = datasetSinAgenda[(datasetSinAgenda["lead_created"].dt.strftime("%A") == dia1) | 
                                           (datasetSinAgenda["lead_created"].dt.strftime("%A") == dia2)].copy()

        # Función para clasificar cada fila en "Dentro" o "Fuera"
        def clasificar_horario(row):
            dia = row["lead_created"].strftime("%A")
            hora_inicio, hora_fin = horarios_interes[dia]
            return "Dentro del rango horario" if hora_inicio <= row["lead_created"].hour <= hora_fin else "Fuera del rango horario"

        # Aplicar clasificación
        subsetAgenda["hipotesis"] = subsetAgenda.apply(clasificar_horario, axis=1)
        subsetSinAgenda["hipotesis"] = subsetSinAgenda.apply(clasificar_horario, axis=1)

        # Contar cantidad de registros en cada categoría
        countAgendaVisita = subsetAgenda[subsetAgenda['Estado Reunión'] == 'REALIZADO']['hipotesis'].value_counts().fillna(0)
        countAgendaNoVisita = subsetAgenda[subsetAgenda['Estado Reunión'] != 'REALIZADO']['hipotesis'].value_counts().fillna(0)
        countNoAgenda = subsetSinAgenda['hipotesis'].value_counts().fillna(0)

        # Crear dataframe con los conteos
        dataHorarios = pd.DataFrame({
            'Con Agenda (VISITO)': countAgendaVisita,
            'Con Agenda (NO VISITO)': countAgendaNoVisita,
            'Sin Agenda': countNoAgenda
        }).fillna(0)
        
        dataHorarios.plot(kind='bar', stacked=True, figsize=(12, 6), edgecolor='black')
        # Crear gráfico para este par de días
        plt.title(f'Distribución dentro y fuera del periodo de horarios ({dia1} - {dia2})', fontsize=16)
        plt.xlabel('Horario', fontsize=12)
        plt.ylabel('Frecuencia', fontsize=12)
        plt.xticks(rotation=0)
        plt.legend(title="Estado", fontsize=10)
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.show()

        # Mostrar el dataframe en consola
        print(f"\nAnálisis para {dia1} - {dia2}")
        dataHorarios['Total'] = dataHorarios['Con Agenda (VISITO)'] + dataHorarios['Con Agenda (NO VISITO)'] + dataHorarios['Sin Agenda']

        # Calcular los porcentajes
        dataHorarios['% Con Agenda'] = ((dataHorarios['Con Agenda (VISITO)'] + dataHorarios['Con Agenda (NO VISITO)']) / dataHorarios['Total']) * 100
        dataHorarios['% Sin Agenda'] = (dataHorarios['Sin Agenda'] / dataHorarios['Total']) * 100

        dataHorarios["totalAgenda"] = dataHorarios['Con Agenda (VISITO)'] + dataHorarios['Con Agenda (NO VISITO)']
        dataHorarios[" % visitaron" ] = (dataHorarios['Con Agenda (VISITO)'] /dataHorarios["Total"] )*100
        dataHorarios[" % NO visitaron" ] = (dataHorarios['Con Agenda (NO VISITO)'] /dataHorarios["Total"] )*100

        print(dataHorarios)
