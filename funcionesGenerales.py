# deriv_RK implementa el método de Runge-Kutta.
# La derivada debe admitir los parámetros x, t y, opcionalmente, parámetros constantes
import math
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import least_squares

argLeastSquares = dict(ftol=1e-13, xtol=1e-13, gtol=1e-13,
                       verbose=True, kwargs={})

""" leeFichero
Entradas: 
    - nombrFich, una cadena de caracteres con el nombre del fichero. 
    - colValLabel, una cadena que identifica la línea del fichero que contiene las etiquetas
      de las columnas (valor por defecto,'Wavelength')
    - intercaladas, un booleano que indica la estructura de los datos. 
      Si vale True, los datos son pares del tipo {(etiqueta, valor)}. 
      Si no, la etiqueta es el primer número de la línea, común a todos los valores de la misma. 
      Valor por defecto: True
    - separador, el carácter que separa los valores almacenados (valor por defecto, ',')

La función lee el fichero indicado fila a fila. Devuelve una lista con dos componentes:
•	colValues, un vector de reales con las etiquetas que identifican cada columna.
•	Measures, una matriz con las medidas guardadas en el fichero.
La matriz Measures está formada por todas las líneas que contienen sólo números
El vector colValues se construye con todos los valores reales de la fila que empieza por la cadena colValLabel.
"""
def leeFichero(nombrFich, colValLabel='Wavelength nm.', intercaladas=True, separador=','):
  # Abrimos el fichero
  with open(nombrFich,'r') as fichero:
    # Inicializamos las variables
    col_names = []
    data_rows = []
    #row_names = []

    # Iteramos por las lineas del fichero
    for linea in fichero:
      row = []

      # Si esta línea es la linea que contiene la colValLabel, extraemos los nombres/valores de las columna
      if linea.split(separador)[0] == colValLabel:
        col_names = [dato.strip() for dato in linea.split(separador) if dato.strip()]
      else:
          # Procesamos cada fila de datos
          values=[dato.strip() for dato in linea.split(separador) if dato.strip()]

          if values:
            #row_name = values[0] # El primer valor es el nombre de la fila (Wavelength)
            row_values = values # El resto de los valores lo ponemos como datos

            # Si intercaladas es cierto, tomamos los valores en posiiones pares de row_values
            if intercaladas:
              row_values=row_values[::2]

            # Intentamos convertir cada valor en float, pero nos quedamos con string si no es possible
            processed_row=[]
            for dato in row_values:
              try:
                processed_row.append(float(dato)) # Intentamos convertir en float
              except ValueError:
                processed_row.append(dato) # Nos quedamos el string si no es in float
            
            #row_names.append(row_name) # Recogemos los nombres de las filas
            data_rows.append(processed_row) # Recogemos los datos de las filas

  # Creamos un DataFrame para los datos recogidos
  df = pd.DataFrame(data_rows)

  # Ponemos los col_names como nombres de las columna, asegurandonos que su numero coincide con el
  # número de columnas en los datos.
  if len(col_names)== len(df.columns) :
    df.columns = col_names # Nos saltamos la primera etiqueta (que es para los nombre de las filas) REVISAR
  else:
    print(f"Warning: The number of columns in the data does not match the header size")
  return df


"""
Dada una lista, lectura, devuelve una matriz de dos columnas:
i) La primera columna almacena los valores de lectura['colValues']
ii) La componente i-ésima de la segunda columna es la diferencia 
    entre los valores máximo y mínimo de la columna i+2 de la matriz lectura['colValues']
"""
def preprocesaAbsorbance(df):
  #df=datos
  # Extraemos los valores de la primera columna (Wavelength nm.)
  wavelength= df.iloc[:,0] # La primer columna contiene los valores de longitud de onda

  # Extramemos las medidas de absorbancia (todas las columnas menos la primera)
  absorbance= df.iloc[:,1:]

  # Extraemos los nombres de las columnas a partir de la tercera (incluida)
  volume= df.columns[2:]
  volume = pd.to_numeric(volume)

  # Inicializamos las listas para almacenar los mínimos, máximos y sus respectivas longitudes de onda
  min_vals = []
  max_vals = []
  min_wave = []
  max_wave = []

  # Iteramos sobre cada columna (excepto la primera y la segunda, longitud de onda y baseline respectivamente)
  for col in df.columns[2:]: 
    # Encontramos el valor mínimo, máximo y su correspondiente longitud de onda
      min_val = df[col].min()
      max_val = df[col].max()

      min_wl = df[df[col] == min_val]['Wavelength nm.'].iloc[0]
      max_wl = df[df[col] == max_val]['Wavelength nm.'].iloc[0]

      # Almacenamos en las listas
      min_vals.append(min_val)
      max_vals.append(max_val)
      min_wave.append(min_wl)
      max_wave.append(max_wl)


  #Calculamos el primedio de las longitudes de onda correpondientes a los mínimos y máximos
  avg_min_wave = np.mean(min_wave)
  avg_max_wave = np.mean(max_wave)

  # Redondeamos las longitudes al incremente más cercano de 0.5 nm
  avg_min_wave = round(avg_min_wave *2)/2 # Asegurando incrementos de 0.5 nm
  avg_max_wave = round(avg_max_wave *2)/2 



  # Encontramos las filas en el DataFrame que corresponden a las longitudes de onda redondeadas
  min_abs = df[df['Wavelength nm.'] == avg_min_wave].iloc[:,2:].reset_index(drop=True).T
  max_abs = df[df['Wavelength nm.'] == avg_max_wave].iloc[:,2:].reset_index(drop=True).T

  # Calculamos deltaAbs como la diferencia entre los absorbancias en las longitudes de onda redondeadas
  deltaAbs = max_abs - min_abs

  # Creamos un DataFrame de salida con los resultados
  out_df = deltaAbs
  out_df.insert(0, column ='',value=volume)
  out_df.reset_index(drop=True, inplace=True)
  out_df.columns= ['Volume ul', 'DeltaAbs']


  # Imprimimos los valores calculados
  print(f'Mínimo promedio:  ({avg_min_wave} nm)\t Máximo promedio:  ({avg_max_wave} nm)')
  print(f'  {out_df}\n')

  return out_df

# Algoritmo Runge-Kutta 4º orden
def deriv_RK (fDeriv,x,t,deltaT,**paramDeriv):

    k1 = fDeriv ( x, t, **paramDeriv)
    k2 = fDeriv ( x+k1*deltaT/2, t+ deltaT/2, **paramDeriv)
    k3 = fDeriv ( x+k2*deltaT/2, t+ deltaT/2, **paramDeriv)
    k4 = fDeriv ( x+k3*deltaT, t+ deltaT, **paramDeriv)

    return (k1+2*k2+2*k3+k4)/6

def Bolzano (f, xMin, xMax, *param, epsilon = 1e-10 , **kwargs):
    fa = f(xMin, *param, **kwargs)
    if fa * f(xMax, *param, **kwargs) > 0:
        sol = np.nan
    else:
        while xMax - xMin > epsilon:
            xMed = (xMax + xMin)/2
            if f(xMed, *param, **kwargs) * fa > 0:
                xMin = xMed
            else:
                xMax = xMed
        sol = (xMax + xMin)/2
    return sol

def guarda(directorioOut, nombrFich, nombres, valores):
    fichero = open(directorioOut+'\\'+nombrFich+'.txt',"w",encoding='utf-8')
    for nombr in nombres:
        fichero.write(nombr+'\t')
    fichero.write('\n')
    for linea in valores:
        for nombr in linea:
            if isinstance (nombr, str):
                fichero.write('%s\t'%nombr)
            else:
                fichero.write('%g\t'%nombr)
        fichero.write('\n')
    fichero.close()

def derivada(x,y):
    return (y[2:]-y[:-2])/(x[2:]-x[:-2])

def ajusta (fResiduo, parVar, **argLS):
    nombrParVar, parFijos = [argLS['kwargs'][nombr] for nombr in ['nombrParVar','parFijos']]

    ajuste = least_squares(fun=fResiduo, x0=parVar, **argLS)

    parAjustados = dict(zip(nombrParVar, ajuste['x']), **parFijos)

    """
    Estimamos el error en los parámetros, siguiendo a 
    Gavin, H. P. (2019). The Levenberg-Marquardt algorithm for nonlinear least squares curve-fitting problems. 
    Department of civil and environmental engineering, Duke University, 19.
    ecuaciones 22 y 25
    """
    m = len(ajuste['fun'])
    n = len(ajuste['x'])
    #resVariance = np.sum(fResiduo(ajuste['x'], **argClave)**2)/(m-n+1)
    resVariance = np.linalg.norm(ajuste['fun'])**2 /(m-n+1)
    #print('ajuste: ', ajuste['fun'], np.linalg.norm(ajuste['fun']), m, n, resVariance)

    try:
        sd = np.sqrt(np.diag(np.linalg.inv(np.transpose(ajuste['jac']).dot(ajuste['jac'])) * resVariance))
    except:
        sd = [np.inf]*len(nombrParVar)
    print ('\n parAjustados:', parAjustados,'\n sd=', sd,'\n')
    """
    Cálculo del coeficiente de determinación R2:
    https: // www.mathworks.com / help / stats / coefficient - of - determination - r - squared_es.html
    """
    SSE = sum(ajuste['fun']**2)
    Y = argLS['kwargs']['Y']
    Y = Y - np.mean(Y)
    SST = sum(Y**2)
    R2 = 1 - SSE/SST

    sdPar = dict ( zip ([nombr+'_std' for nombr in nombrParVar],sd))
    return dict(parAjustados=parAjustados, sdPar = sdPar, R2=R2,detalles=ajuste)

def redefineALS (ALS, paramF,**dict):
    for nombr in paramF:
        ALS['kwargs'][nombr] = paramF[nombr]
    for nombr in dict:
        ALS[nombr] = dict[nombr]

def residualsLS (param, **kwargs):
    nombrParVar, parFijos, f, fKwargs,Y = \
        [kwargs[nombr] for nombr in ['nombrParVar','parFijos','f','fKwargs','Y']]
    parametros = dict ( zip(nombrParVar, param), **parFijos)
    sol = f (parametros, **fKwargs) - Y
    print('\t||residuals|| = ' + str(np.linalg.norm(sol)))
    #print(paramModelo)
    return sol

def procesa (**dictIn):
    argLeastSquares,nombrParVar, dictParEstim, f, fKwargs, Y= \
        [dictIn[nombr] for nombr in ['argLeastSquares','nombrParVar','dictParEstim','f','fKwargs','Y']]
    if 'bounds' in dictIn:
        bounds = dictIn['bounds']
    else:
        bounds = ([0 for nombr in nombrParVar], [np.inf for nombr in nombrParVar])

    nombrParFijos = [nombr for nombr in dictParEstim if nombr not in nombrParVar]
    valoresParFijos = [dictParEstim[nombr] for nombr in nombrParFijos]
    parFijos = dict(zip(nombrParFijos, valoresParFijos))
    dictResiduo =dict ( parFijos = parFijos,
                        nombrParVar= nombrParVar,
                        f = f,
                        fKwargs = fKwargs,
                        Y = Y)
    redefineALS(    argLeastSquares,
                    paramF=dictResiduo,
                    bounds=bounds
                )

    estim = [dictParEstim[nombr] for nombr in nombrParVar]
    sol = ajusta(residualsLS, estim, **argLeastSquares)
    return sol

