#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np

import streamlit as st
from io import BytesIO
import requests


# # <b><font color="navy">1. Promedio Movil Simple PMS</span></font></b>

# ## 1.1 Cargar datos

# In[4]:


def cargar_datos_desde_github(url):
    response = requests.get(url)
    response.raise_for_status()  # Check if the request was successful
    return pd.read_csv(BytesIO(response.content), encoding='utf-8')


# ## 1.2. Pre-procesamiento de datos

# In[5]:


def preprocesamiento_1(df):
    # Asignar formato datetime a columna Fecha
    df['FECHA'] = pd.to_datetime(df['FECHA'], format='%d-%m-%y')
    
    # Establecer FECHA como datetime index
    df.set_index('FECHA', inplace=True)
    
    # Eliminar columna COD_ALMACEN
    df = df.drop(columns = 'COD_ALMACEN')
    
    # Filtrar por fechas posteriores a 2021-01-01
    df = df[df.index >= '2021-01-01'].copy()
    
    # Usar función GROUP BY para agrupar demanda por semana, comenzando los lunes y terminando los domingos
    df_sem = df.groupby(['COD_SKU', 'DESC_SKU']).resample('W-SUN').sum(numeric_only=True)
    
    # Resetear Index para aplanar la tabla
    df_sem.reset_index(inplace=True)
    
    # Colocar semanas como columnas con una tabla dinamica
    df_sem_td = df_sem.pivot(index=['COD_SKU', 'DESC_SKU'], columns='FECHA', values='DEMANDA').fillna(0)
    
    return df_sem_td


# df_sem_td = preprocesamiento_1(df)

# ## 1.3. Funciones para calcular PMS

# In[7]:


def extraer_datos_demanda(df_sem_td):
    # Seleccionar las columnas que comienzan por '202' (las de demanda)
    columnas_dem = df_sem_td.filter(like='202')
    
    # Llevar valores de demanda a una lista
    series_tiempo = columnas_dem.values.tolist()

    return series_tiempo


# In[8]:


def promedio_movil(demanda, extra_periods, n):
    
    # Determina el numero de datos de la demanda
    largo_demanda = len(demanda) 
    # Adiciona el numero de periodos que queremos pronosticar
    demanda = np.append(demanda, [np.nan] * extra_periods)
    # Crea un arreglo para posteriormente escribir los pronosticos de largo demanda + los perdiodos extras
    forecast = np.full(largo_demanda + extra_periods, np.nan)  
    
    # Itera entre n y el largo de la serie de demanda
    for t in range(n, largo_demanda):
        # Promedia los datos de la demanda correspondientes segun n y los datos disponibles
        forecast[t] = np.mean(demanda[t-n:t]) 
    for u in range(1,extra_periods + 1):
        # Pronostica el periodo siguiente
        forecast[t+u] = np.mean(demanda[t-n+1:t+1]) 
    
    # Crea el data frame con los pronósticos
    df = pd.DataFrame.from_dict({'DEMANDA': demanda, 'FORECAST': forecast, 'ERROR': demanda-forecast,}) 
    
    # Calcula 'Error Porcentual' usando función lambda 
    df['ERROR_PORCENTUAL'] = df.apply(
                    lambda fila: 2 if (fila['DEMANDA'] <= 0 and pd.notna(fila['FORECAST'])) else abs(fila['DEMANDA'] 
                        - fila['FORECAST']) / fila['DEMANDA'], axis=1)
  
    # Regresa df como resultado   
    return df


# In[9]:


def generacion_mejor_promedio_movil(series_tiempo, extra_periods, n_min, n_max, barra_progreso_pms=None):
    
    # Crea una lista vacia para acumular el pronostico del periodo siguiente 
    forecast_siguiente = [] 
    # Crea una lista vacia para acumular el wmape correspondiente de cada referencia
    mejor_error = []
    # Crea una lista vacia para acumular el n con menor error
    mejor_n = []
    # Crea una lista vacia para acumular el n con menor error
    rmse_mejor_n = []

    # Para calculo del error total
    total_error_abs = []
    total_demanda = []

    total_series = len(series_tiempo)
    # Itera por cada una de las series de tiempo
    for i, serie in enumerate(series_tiempo): 
        params = []
        KPIs = []
        dfs = []
        RMSE = []
        error_abs = []
        suma_demanda = []
        
        for n in range(n_min,n_max):        
            # Aplica funcion de promedio por cada n
            df_prom_mov =  promedio_movil(serie, extra_periods = extra_periods, n = n)
            # Acumula los parametros
            params.append(n)
            # Acumula las tablas
            dfs.append(df_prom_mov)
            # Suma la demanda que aplica para calcular el error
            sum_dem = df_prom_mov.loc[df_prom_mov['ERROR'].notnull(), 'DEMANDA'].sum()
            sum_len = df_prom_mov.loc[df_prom_mov['ERROR'].notnull(), 'DEMANDA'].count()
            sum_error_abs = df_prom_mov['ERROR'].abs().sum()
            # Calcula el wmape, devuelve 200% si demanda igual o menor que 0
            mae_porc = 2 if sum_dem <= 0 else sum_error_abs / sum_dem
            rmse = np.sqrt((df_prom_mov['ERROR']**2).sum() / sum_len)
            # Acumula KPI
            KPIs.append(mae_porc)
            RMSE.append(rmse)
            error_abs.append(sum_error_abs)
            suma_demanda.append(sum_dem)
            
        # Seleeciona el minimo de cada n
        minimo = np.argmin(KPIs)
        # Seleciona el n correspondiente al mejor error
        mejor_param_n = params[minimo]
        # Selecciona el wmape minimo
        mae_porc_minimo = np.min(KPIs)
        # Selecciona la suma del error absoluto correspondiente al mejor n 
        sum_error_abs_min = error_abs[minimo]
        # Selecciona la suma de la demanda correspondiente al mejor n 
        sum_dem_min = suma_demanda[minimo]
        # Selecciona el rmse correspondiente al mejor n (no necesariamente es el menor rmse)
        rmse_n = RMSE[minimo]
        # Selecciona el pronostico de la tabla con mejor wmape
        forecast =  dfs[minimo]['FORECAST'].iloc[-1]
        # Acumula el pronostico
        forecast_siguiente.append(forecast)
        # Acumula el wmape
        mejor_error.append(mae_porc_minimo)
        # Acumula el mejor n con base en el indice +1
        mejor_n.append(mejor_param_n)
        # Acumula el rmse correspondiente al mejor n
        rmse_mejor_n.append(rmse_n)
        #Calcula error global
        total_error_abs.append(sum_error_abs_min)
        total_demanda.append(sum_dem_min)

        if barra_progreso_pms:
            barra_progreso_pms.progress((i + 1) / total_series)
    # Calcula el mae% de todos los sku
    error_global = np.sum(total_error_abs) / np.sum(total_demanda) if np.sum(total_demanda) != 0 else float('inf')
    
    return forecast_siguiente, mejor_n, rmse_mejor_n, error_global


# In[10]:


def entregable_pms(forecast_siguiente, mejor_n, rmse_mejor_n, df_sem_td): 
    df_return_pms =  pd.DataFrame({'n_OPTIMO':mejor_n,'PRONOSTICO_PMS':forecast_siguiente , 'RMSE_PMS':rmse_mejor_n, 
                          }, index = df_sem_td.index )
    return df_return_pms   


# # <b><font color="navy">2. Suavización Exponencial SE</span></font></b>

# ### 2.1. Funcion para calcular SE Simple por cada Sku

# In[11]:


def suavizacion_exp_simple(demanda, extra_periods, alfa):

    # periodos historicos
    largo_demanda = len(demanda)

    # Acumular np.nan en el arreglo de demanda para cubrir periodos  futuros
    d = np.append(demanda, [np.nan]*extra_periods)

    # Arreglo para el pronostico
    f = np.full(largo_demanda +  extra_periods, np.nan)
    # Inicializar modelo
    f[1] = d[0]

    #Crear todos los pronosticos t+1 hasta el final de los periodos historicos
    for t in range(2, largo_demanda+1):
        f[t] = alfa*d[t-1]+(1-alfa)*f[t-1]

    # Pronostico para los pediodos extra
    for t in range(largo_demanda+1, largo_demanda+extra_periods):
        # Actualizar el pronostico con el pronostico anterior
        f[t] = f[t-1]

    df = pd.DataFrame.from_dict({'DEMANDA':d,'FORECAST':f,'ERROR':d-f})
    
    return df
    


# ### 2.2 Función para aplicar mejor alfa a todos los Sku

# In[12]:


def generacion_mejor_suavizacion_exp(series_tiempo, extra_periods, alfa_min, alfa_max, barra_progreso_se=None):
    
    rango_alfa = np.arange(alfa_min, alfa_max+0.01, 0.01)
    # Crea una lista vacia para acumular el pronostico del periodo siguiente 
    forecast_siguiente_se = [] 
    # Crea una lista vacia para acumular el wmape correspondiente de cada referencia
    mejor_error_se = []
    # Crea una lista vacia para acumular el alfa con menor error
    mejor_alfa = []
    # Crea una lista vacia para acumular el alfa con menor error
    rmse_mejor_alfa = []

    # Para calculo del error total
    total_error_abs = []
    total_demanda = []
    
    total_series = len(series_tiempo)
    # Itera por cada una de las series de tiempo
    for i, serie in enumerate(series_tiempo):
        parametros = []
        KPIs = []
        dfs_se = []
        RMSE_se = []
        error_abs = []
        suma_demanda = []
        
        for alfa in rango_alfa:
                  
            # Aplica funcion de promedio por cada alfa
            df =  suavizacion_exp_simple(demanda=serie, extra_periods = extra_periods, alfa = alfa)
            # Acumula los parametros
            parametros.append(alfa)
            # Acumula las tablas
            dfs_se.append(df)
            # Suma la demanda que aplica para calcular el error
            sum_dem = df.loc[df['ERROR'].notnull(), 'DEMANDA'].sum()
            sum_len = df.loc[df['ERROR'].notnull(), 'DEMANDA'].count()
            sum_error_abs = df['ERROR'].abs().sum()
            # Calcula el wmape, devuelve 200% si demanda igual o menor que 0
            mae_porc = 2 if sum_dem <= 0 else sum_error_abs / sum_dem
            rmse = np.sqrt((df['ERROR']**2).sum() / sum_len)
            # Acumula KPI
            KPIs.append(mae_porc)
            RMSE_se.append(rmse)
            error_abs.append(sum_error_abs)
            suma_demanda.append(sum_dem)
            
        # Seleeciona el minimo de cada alfa
        minimo = np.argmin(KPIs)
        # Selecciona el alfa correspondiente al menor error
        mejor_param_alfa = parametros[minimo]
        # Selecciona el wmape minimo
        mae_porc_minimo = np.min(KPIs)
        # Selecciona la suma del error absoluto correspondiente al mejor n 
        sum_error_abs_min = error_abs[minimo]
        # Selecciona la suma de la demanda correspondiente al mejor n 
        sum_dem_min = suma_demanda[minimo]
        # Selecciona el rmse correspondiente al mejor n (no necesariamente es el menor rmse)
        rmse_alfa = RMSE_se[minimo]
        # Selecciona el pronostico de la tabla con mejor wmape
        forecast =  dfs_se[minimo]['FORECAST'].iloc[-1]
        # Acumula el pronostico
        forecast_siguiente_se.append(forecast)
        # Acumula el wmape
        mejor_error_se.append(mae_porc_minimo)
        # Acumula el mejor n con base en el indice +1
        mejor_alfa.append(mejor_param_alfa)
        # Acumula el rmse correspondiente al mejor n
        rmse_mejor_alfa.append(rmse_alfa)
        #Calcula error global
        total_error_abs.append(sum_error_abs_min)
        total_demanda.append(sum_dem_min)

        if barra_progreso_se:
            barra_progreso_se.progress((i + 1) / total_series)
        
    error_global_se = np.sum(total_error_abs) / np.sum(total_demanda) if np.sum(total_demanda) != 0 else float('inf')
    
    return forecast_siguiente_se, mejor_alfa, rmse_mejor_alfa, error_global_se


# ## Archivo Final

# In[13]:


def entregable_se(forecast_siguiente_se, mejor_alfa, rmse_mejor_alfa, df_sem_td): 
    df_return_se =  pd.DataFrame({'alfa_OPTIMO':mejor_alfa,'PRONOSTICO_SE':forecast_siguiente_se , 'RMSE_SE':rmse_mejor_alfa
                          }, index = df_sem_td.index )
    return df_return_se                    


# In[14]:


def entregable(forecast_siguiente, mejor_n, rmse_mejor_n, forecast_siguiente_se, mejor_alfa, rmse_mejor_alfa, df_sem_td): 
    df_return =  pd.DataFrame({'n_OPTIMO':mejor_n,'PRONOSTICO_PMS':forecast_siguiente , 'RMSE_PMS':rmse_mejor_n, 'alfa_OPTIMO':mejor_alfa,'PRONOSTICO_SE':forecast_siguiente_se , 'RMSE_SE':rmse_mejor_alfa
                          }, index = df_sem_td.index )
    return df_return      


# # <b><font color="navy">3. Ejecución APP</span></font></b>

# In[15]:


def main():
    # Split layout into two columns
    st.title("App para Cálculo de Pronósticos")

    st.sidebar.title('Flujo de Datos')
    seccion = st.sidebar.radio('⬇️ ir a:', ('Carga de datos', 'Forecasting', 'Descargar Resultados'))

    if 'df_orig' not in st.session_state:
        st.session_state.df_orig = None
        
    if seccion == 'Carga de datos':

        github_url = 'https://raw.githubusercontent.com/wgutierr/planeacion_demanda/main/dataset/demanda_dia.csv'
        
        if st.button("Cargar datos desde GitHub"):
            st.session_state.df_orig = cargar_datos_desde_github(github_url)          
              
        if st.session_state.df_orig is not None:
            st.success('Archivo Cargado Exitosamente')
            col1, buffer, col2 = st.columns([5, 1, 2])
            
            with col1:
                st.write("Información Cargada:")
                st.dataframe(st.session_state.df_orig)
            with col2:
                st.metric(label='Filas', value=len(st.session_state.df_orig))
                st.metric(label='Columnas', value=len(st.session_state.df_orig.columns))
                
            df_sem_td = preprocesamiento_1(st.session_state.df_orig)
            st.session_state.df_sem_td = df_sem_td
            
            col_1_1, col_1_2 = st.columns([3, 1])
            
            with col_1_1:
                st.write("Demanda agrupada por semana por SKU")
                st.dataframe(df_sem_td)
            with col_1_2:
                st.metric(label='Filas', value=len(df_sem_td))
                st.metric(label='Columnas', value=len(df_sem_td.columns))
                
            series_tiempo = extraer_datos_demanda(df_sem_td)
            st.session_state.series_tiempo = series_tiempo
            
            # Initialize session state variables if they don't exist
            if 'extra_periods' not in st.session_state:
                st.session_state.extra_periods = []
            if 'forecast_siguiente' not in st.session_state:
                st.session_state.forecast_siguiente = []
            if 'mejor_n' not in st.session_state:
                st.session_state.mejor_n = []
            if 'rmse_mejor_n' not in st.session_state:
                st.session_state.rmse_mejor_n = []
            if 'error_global_pms' not in st.session_state:
                st.session_state.error_global_pms = None  # Initialize error_global_pms
        else:
            st.error('No se ha cargado el archivo de demanda o no se cargó correctamente. Verifique el formato y vuelva a intentarlo.')
        
            
        
    elif seccion == 'Forecasting':
        
        tabs = st.tabs(['Promedio Movil Simple', 'Suavizacion Exponencial'])
    
        with tabs[0]:
            
            st.title("Promedio Movil Simple")
            
            st.write("Por favor completar parámetros para la optimización por promedio móvil simple")
            
            extra_periods_pms = st.number_input("Períodos adicionales PMS", min_value=1, step=1)
            n_min_pms = st.number_input("Valor mínimo de n", min_value=1, max_value=52, step=1, value=3)
            n_max_pms = st.number_input("Valor máximo de n", min_value=1, max_value=53, step=1, value=16)
            
            if st.button('Generar Mejor PMS'):
                with st.spinner('Calculando Pronosticos y Errores en Unidades...'):    
                    barra_progreso_pms = st.progress(0)
                    
                    forecast_siguiente, mejor_n, rmse_mejor_n, error_global = generacion_mejor_promedio_movil(
                        st.session_state.series_tiempo, extra_periods_pms, n_min_pms, n_max_pms, barra_progreso_pms
                    )
                    
                    barra_progreso_pms.progress(100)
                    
                    st.session_state.forecast_siguiente = forecast_siguiente
                    st.session_state.mejor_n = mejor_n
                    st.session_state.rmse_mejor_n = rmse_mejor_n
                    st.session_state.extra_periods = extra_periods_pms
                    st.session_state.error_global_pms = error_global  # Store error_global for PMS
                    
                    col3, buffer2, col4 = st.columns([4, 1, 2])
                    with col3:               
                        df_resultado_pms = entregable_pms(
                        st.session_state.forecast_siguiente, st.session_state.mejor_n, st.session_state.rmse_mejor_n,
                        st.session_state.df_sem_td)
                        
                        st.dataframe(df_resultado_pms)
                        st.session_state.df_resultado_pms = df_resultado_pms
                    with col4:
                        st.metric(label='MAE% Global PMS', value="{:.2%}".format(error_global), delta = 'en  unidades')
             
                
        with tabs[1]:

            st.title("Suavización Exponencial")
            
            st.write("Por favor completar parámetros para la optimización por suavización exponencial")
            extra_periods_se = st.number_input("Períodos adicionales SE", min_value=1, step=1)
            alfa_min = st.number_input("Valor mínimo de alfa", min_value=0.01, max_value=0.98, step=0.01, value=0.2)
            alfa_max = st.number_input("Valor máximo de alfa", min_value=0.02, max_value=0.99, step=0.01, value=0.6)
    
            if st.button("Calcular SE"):
                with st.spinner('Calculando Pronosticos y Errores en Unidades...'):
                    barra_progreso_se = st.progress(0)
                    forecast_siguiente_se, mejor_alfa, rmse_mejor_alfa, error_global_se = generacion_mejor_suavizacion_exp(
                        st.session_state.series_tiempo, extra_periods_se, alfa_min, alfa_max, barra_progreso_se
                    )
                    barra_progreso_se.progress(100)
                    st.session_state.forecast_siguiente_se = forecast_siguiente_se
                    st.session_state.mejor_alfa = mejor_alfa
                    st.session_state.rmse_mejor_alfa = rmse_mejor_alfa
                    col5, buffer3, col6 = st.columns([4, 1, 2])
                    with col5:               
                        df_resultado_se = entregable_se(
                        st.session_state.forecast_siguiente_se, st.session_state.mejor_alfa, st.session_state.rmse_mejor_alfa,
                        st.session_state.df_sem_td 
                        )
                        
                        st.dataframe(df_resultado_se)
                        st.session_state.df_resultado_se = df_resultado_se
                        
                    with col6:
                        st.metric(label='MAE% Global SE', value="{:.2%}".format(error_global_se))
                    # Display PMS error if it was previously calculated            

    elif seccion == 'Descargar Resultados':
        st.title("Descargar Resultados")
        if 'df_resultado_pms' in st.session_state and 'df_resultado_se' in st.session_state:
       
            
            df_resultado_final = entregable(
                    st.session_state.forecast_siguiente, st.session_state.mejor_n, st.session_state.rmse_mejor_n,
                    st.session_state.forecast_siguiente_se, st.session_state.mejor_alfa, st.session_state.rmse_mejor_alfa,            st.session_state.df_sem_td,
                                        )
                
            st.session_state.df_resultado_final = df_resultado_final
        
            buffer = BytesIO()
            st.session_state.df_resultado_final.to_excel(buffer, index=True)
            buffer.seek(0)

            st.download_button(
                label="Descargar resultado final como Excel",
                data=buffer,
                file_name='resultado_final.xlsx',
                mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
            )
        else:
            st.write("No hay resultados disponibles para descargar")

if __name__ == '__main__':
    main()

