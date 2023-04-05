# Acerca de Simulation_methods
A continuación veremos en detalle cada una de las funciones presente en este archivo que entregan como output la simulación de una variable aleatoria 
en especifico, veremos en detalle cual es esta, con que método se obteine, y cuales son las variables de entrada, ademas de dar con que función se puede 
dar una visualizacion del valor obtenido

### Metropolis_Hasting_XY(N, n, x_0, beta)
Esta función recibe como input las variables
- N:  el numero de pasos de la cadena asociada 
- n: el radio de la grilla, si se tiene n como radio $(2*n+1)^2$ es el tamaño de esta
- x_0: un valor inicial de la cadena, mediante la función "todos_ordenados", se obtiene un valor inicial para esta función
- beta: el valor de inverso a la temperatura
como output entrega una realización (aproximada) de un modelo XY

### Swendsen_Wang_Villain(N, radius,beta, x0 =str(1))
Esta función tiene de input
- N: el numero de iteraciones deseada 
- n: el radio de la grilla
- beta: el valor inverso de la temperatura
- x0: un valor inicial para el algoritmo, si no se da este valor, el algoritmo procede como si fueran todos los valores iguales a 0 radianes

como output entrega  una realización del modelo de Villain.
tambien se dispone de Swendsen_Wang_Villain_H(N, radius,beta, x0 =str(1)), que la unica diferencia del anterior, es que en esta el output ademas de ser 
una simulación del modelo de Villain, tambien entrega un historial del proceso de clusterización y los valores de cada iteración 

### Swendsen_Wang_Villain_b0(N,n,beta, x_0 = str(1)):
Esta función tiene de input
- N: el numero de iteraciones deseada 
- n: el radio de la grilla
- beta: el valor inverso de la temperatura
- x0: un valor inicial para el algoritmo, si no se da este valor, el algoritmo procede como si fueran todos los valores iguales a 0 radianes

La diferencia con el caso anterior es que en esta, el modelo de Villain simulado, tiene una condición de borde dada, esta se da desde el valor x0, 
pues en todo el proceso nunca modifica tales valores

### def gas_de_coulumb(N, n, beta):
esta función como input tiene
- N: numero de iteraciones deseada
- n: radio de la grilla
- beta: valor del inverso de la temperatura
como output, esta función entrega 2 matrices, la primera, es una simulación de Villain utilizada, la segunda es el gas de Coulumb
