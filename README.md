# planeacion_demanda
 App simple de calculo de PMS y SE

 Esta app carga información histórica de demanda con frecuencia día, y la transforma a frecuencia semanal. Esto en la pestaña 'Cargar Datos'.
 En la pestaña 'Forecasting' aplica dos modelos de pronósticos. El primero es un PMS. El algoritmo evalúa cada SKU y encuentra el n (número de periodos historicos a promediar) que genera el menor error (MAE%). Selecciona este n y cálcula el RMSE para ser utilizado posteriormente en cálculos de stock de seguridad.
 Puede escogerese entre 1 y 52 periodos a promediar. No se habilitan más para no impactar el tiempo de ejecución, adicional no se genera un impacto mayor al aumentar las iteraciones de n.
 El segundo modelo es una suavización exponencial simple. El proceso es similar, solo que se optimiza el valor de alfa (coeficiente de suavización) en lugar de n.
 Finalemnte se cuenta con una opción de descargar los pronósticos generados a excel. Se descarga tanto el pronóstico, como el mejor n en el caso de PMS, el mejor alga en el caso de SE y el RMSE para ambos.
