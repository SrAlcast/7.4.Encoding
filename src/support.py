# Tratamiento de datos
# -----------------------------------------------------------------------
import pandas as pd

# Configuración
# -----------------------------------------------------------------------
pd.set_option('display.max_columns', None) # para poder visualizar todas las columnas de los DataFrames

# Ignorar los warnings
# -----------------------------------------------------------------------
import warnings
warnings.filterwarnings('ignore')

import numpy as np

# Para la visualización 
# -----------------------------------------------------------------------
import matplotlib.pyplot as plt
import seaborn as sns

# Otros objetivos
# -----------------------------------------------------------------------
import math
from itertools import combinations

# Para pruebas estadísticas
# -----------------------------------------------------------------------
from scipy import stats
from statsmodels.stats.multicomp import pairwise_tukeyhsd

# Para la codificación de las variables numéricas
# -----------------------------------------------------------------------
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, LabelEncoder, TargetEncoder # para poder aplicar los métodos de OneHot, Ordinal,  Label y Target Encoder 

import math
import matplotlib.pyplot as plt
import seaborn as sns

def crear_boxplot(dataframe, lista_variables, variable_respuesta, whis=1.5, color="blue", tamano_grafica_base=(20, 5)):
    """
    Crea un boxplot para cada variable categórica en el conjunto de datos con una visualización mejorada.

    Parámetros:
    - dataframe: DataFrame que contiene los datos.
    - lista_variables: Lista de variables categóricas para generar los boxplots.
    - variable_respuesta: Variable respuesta para graficar en el eje y.
    - whis: El ancho de los bigotes. Por defecto es 1.5.
    - color: Color de los boxplots. Por defecto es "blue".
    - tamano_grafica_base: Tamaño base de cada fila de gráficos. Por defecto es (20, 5).
    """
    num_variables = len(lista_variables)
    num_filas = math.ceil(num_variables / 2)
    
    # Ajustar el tamaño de la figura dinámicamente
    tamano_grafica = (tamano_grafica_base[0], tamano_grafica_base[1] * num_filas)
    
    fig, axes = plt.subplots(num_filas, 2, figsize=tamano_grafica)
    axes = axes.flat

    for indice, columna in enumerate(lista_variables):
        sns.boxplot(
            y=variable_respuesta,
            x=columna,
            data=dataframe,
            color=color,
            ax=axes[indice],
            whis=whis,
            flierprops={'markersize': 4, 'markerfacecolor': 'orange'}
        )
        axes[indice].set_title(f'Boxplot: {columna}', fontsize=12)  # Título de cada subgráfico
        axes[indice].tick_params(axis='x', rotation=45)  # Rotar etiquetas del eje X

    # Ocultar los ejes restantes si hay un número impar de gráficos
    for ax in axes[num_variables:]:
        ax.axis('off')

    # Ajustar diseño general
    fig.tight_layout()
    plt.show()


def crear_barplot(dataframe, lista_variables, variable_respuesta, paleta="viridis", tamano_grafica_base=(20, 10)):
    """
    Crea un barplot para cada variable categórica en el conjunto de datos con una visualización mejorada.

    Parámetros:
    - dataframe: DataFrame que contiene los datos.
    - lista_variables: Lista de variables categóricas para generar los barplots.
    - variable_respuesta: Variable respuesta para calcular la media en cada categoría.
    - paleta: Paleta de colores para el barplot. Por defecto es "viridis".
    - tamano_grafica_base: Tamaño base de cada fila de gráficos. Por defecto es (20, 5).
    """
    num_variables = len(lista_variables)
    num_filas = math.ceil(num_variables / 2)
    
    # Ajustar tamaño de la figura dinámicamente
    tamano_grafica = (tamano_grafica_base[0], tamano_grafica_base[1] * num_filas)
    
    fig, axes = plt.subplots(num_filas, 2, figsize=tamano_grafica)
    axes = axes.flat

    for indice, columna in enumerate(lista_variables):
        # Calcular la media agrupada por categorías
        categoria_mediana = (
            dataframe.groupby(columna)[variable_respuesta]
            .mean()
            .reset_index()
            .sort_values(by=variable_respuesta)
        )

        # Crear el barplot
        sns.barplot(
            x=categoria_mediana[columna],
            y=categoria_mediana[variable_respuesta],
            palette=paleta,
            ax=axes[indice],
            errorbar='ci'
        )

        # Agregar títulos y ajustar etiquetas
        axes[indice].set_title(f"Media de {variable_respuesta} por {columna}", fontsize=12)
        axes[indice].tick_params(axis='x', rotation=45)

    # Ocultar los ejes sobrantes si el número de gráficos es impar
    for ax in axes[num_variables:]:
        ax.axis('off')

    # Ajustar diseño general
    fig.tight_layout()
    plt.show()



import pandas as pd
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, LabelEncoder
from category_encoders import TargetEncoder

def one_hot_encoding(dataframe, columns):
    """
    Realiza codificación one-hot en las columnas especificadas.

    Args:
        dataframe (pd.DataFrame): DataFrame de pandas.
        columns (list): Lista de columnas a codificar.

    Returns:
        pd.DataFrame: DataFrame con codificación one-hot aplicada.
    """
    one_hot_encoder = OneHotEncoder()
    trans_one_hot = one_hot_encoder.fit_transform(dataframe[columns])
    oh_df = pd.DataFrame(trans_one_hot.toarray(), columns=one_hot_encoder.get_feature_names_out(columns))
    dataframe = pd.concat([dataframe.reset_index(drop=True), oh_df.reset_index(drop=True)], axis=1)
    dataframe.drop(columns=columns, inplace=True)
    return dataframe


def get_dummies_encoding(dataframe, columns, prefix=None, prefix_sep="_"):
    """
    Realiza codificación get_dummies en las columnas especificadas.

    Args:
        dataframe (pd.DataFrame): DataFrame de pandas.
        columns (list): Lista de columnas a codificar.
        prefix (str o dict, opcional): Prefijo para las columnas codificadas.
        prefix_sep (str): Separador entre el prefijo y la columna original.

    Returns:
        pd.DataFrame: DataFrame con codificación get_dummies aplicada.
    """
    df_dummies = pd.get_dummies(dataframe[columns], dtype=int, prefix=prefix, prefix_sep=prefix_sep)
    dataframe = pd.concat([dataframe.reset_index(drop=True), df_dummies.reset_index(drop=True)], axis=1)
    dataframe.drop(columns=columns, inplace=True)
    return dataframe


def ordinal_encoding(dataframe, columns, categories):
    """
    Realiza codificación ordinal en las columnas especificadas.

    Args:
        dataframe (pd.DataFrame): DataFrame de pandas.
        columns (list): Lista de columnas a codificar.
        categories (list of list): Lista de listas con las categorías en orden.

    Returns:
        pd.DataFrame: DataFrame con codificación ordinal aplicada.
    """
    ordinal_encoder = OrdinalEncoder(categories=categories, dtype=float, handle_unknown="use_encoded_value", unknown_value=np.nan)
    dataframe[columns] = ordinal_encoder.fit_transform(dataframe[columns])
    return dataframe


def label_encoding(dataframe, columns):
    """
    Realiza codificación label en las columnas especificadas.

    Args:
        dataframe (pd.DataFrame): DataFrame de pandas.
        columns (list): Lista de columnas a codificar.

    Returns:
        pd.DataFrame: DataFrame con codificación label aplicada.
    """
    label_encoder = LabelEncoder()
    for col in columns:
        dataframe[col] = label_encoder.fit_transform(dataframe[col])
    return dataframe


def target_encoding(dataframe, columns, target):
    """
    Realiza codificación target en las columnas especificadas.

    Args:
        dataframe (pd.DataFrame): DataFrame de pandas.
        columns (list): Lista de columnas a codificar.
        target (str): Nombre de la variable objetivo.

    Returns:
        pd.DataFrame: DataFrame con codificación target aplicada.
    """
    target_encoder = TargetEncoder(cols=columns)
    dataframe[columns] = target_encoder.fit_transform(dataframe[columns], dataframe[target])
    return dataframe


def frequency_encoding(dataframe, columns):
    """
    Realiza codificación de frecuencia en las columnas especificadas.

    Args:
        dataframe (pd.DataFrame): DataFrame de pandas.
        columns (list): Lista de columnas a codificar.

    Returns:
        pd.DataFrame: DataFrame con codificación de frecuencia aplicada.
    """
    for col in columns:
        freq_map = dataframe[col].value_counts(normalize=True)
        dataframe[col] = dataframe[col].map(freq_map)
    return dataframe
