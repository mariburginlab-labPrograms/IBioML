"""
IbioML - Toolkit de Machine Learning para experimentos de neurodecodificación.

Este paquete proporciona herramientas especializadas para el análisis de datos
neuronales y la implementación de experimentos de neurodecodificación.
"""

__version__ = "0.1.2"
__author__ = "Juan Ignacio Ponce"
__email__ = "jiponce@ibioba-mpsp-conicet.gov.ar"

# Imports principales
from . import models
from . import preprocess_data  
from . import tuner
from . import trainer
from . import plots

# Hacer disponibles las clases principales
from .models import MLPModel, RNNModel, LSTMModel, GRUModel

__all__ = [
    # Módulos
    "models", 
    "preprocess_data", 
    "tuner", 
    "trainer", 
    "plots",
    # Clases principales
    "MLPModel", 
    "RNNModel", 
    "LSTMModel", 
    "GRUModel",
]