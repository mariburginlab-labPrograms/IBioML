#!/usr/bin/env python3
"""
Ejemplos de uso de parámetros de capas exclusivas en model_space.

Este script muestra cómo configurar num_exclusive_layers y exclusive_hidden_size
tanto con valores fijos como con rangos de búsqueda (optimizables).

NOTA: Este es un script de documentación. Los model_space aquí son ejemplos
      de cómo estructurar tu configuración, no se ejecutan modelos reales.
"""

# ========================================
# PARTE 1: Configuraciones fijas
# ========================================

print("\n" + "="*80)
print("PARTE 1: CONFIGURACIONES FIJAS en model_space")
print("="*80)

# Ejemplo 1: Sin capas exclusivas (comportamiento clásico)
model_space_1 = {
    "model_class": "DualOutputMLPModel",  # O usar la clase directamente
    "output_size": 1,
    "hidden_size": 128,
    "num_layers": 3,
    "dropout": 0.2,
    "lr": 0.001,
    "batch_size": 64,
    "num_epochs": 10,
    "es_patience": 3,
    "reg_type": None,
    "lambda_reg": 1e-4,
    "device": "cpu",
    
    # Parámetros de capas exclusivas
    "num_exclusive_layers": 0,  # 0 = sin capas exclusivas
}

print("\n📝 Ejemplo 1: Sin capas exclusivas")
print("   num_exclusive_layers: 0")
print("   → Se traduce a use_exclusive=False")

# Ejemplo 2: Con 1 capa exclusiva del mismo tamaño (default mejorado)
model_space_2 = {
    "model_class": "DualOutputMLPModel",
    "output_size": 1,
    "hidden_size": 128,
    "num_layers": 3,
    "dropout": 0.2,
    "lr": 0.001,
    "batch_size": 64,
    "num_epochs": 10,
    "es_patience": 3,
    "reg_type": None,
    "lambda_reg": 1e-4,
    "device": "cpu",
    
    # Parámetros de capas exclusivas
    "num_exclusive_layers": 1,  # 1 capa exclusiva
    # exclusive_hidden_size no especificado = usa hidden_size (128)
}

print("\n📝 Ejemplo 2: Con 1 capa exclusiva del mismo tamaño")
print("   num_exclusive_layers: 1")
print("   exclusive_hidden_size: (no especificado, usa hidden_size=128)")

# Ejemplo 3: Con capas exclusivas más pequeñas
model_space_3 = {
    "model_class": "DualOutputMLPModel",
    "output_size": 1,
    "hidden_size": 128,
    "num_layers": 3,
    "dropout": 0.2,
    "lr": 0.001,
    "batch_size": 64,
    "num_epochs": 10,
    "es_patience": 3,
    "reg_type": None,
    "lambda_reg": 1e-4,
    "device": "cpu",
    
    # Parámetros de capas exclusivas
    "num_exclusive_layers": 1,
    "exclusive_hidden_size": 64,  # Más pequeño que hidden_size
}

print("\n📝 Ejemplo 3: Con capas exclusivas más pequeñas")
print("   num_exclusive_layers: 1")
print("   exclusive_hidden_size: 64 (< hidden_size=128)")
print("   → Menos parámetros, más rápido")

# Ejemplo 4: Con múltiples capas exclusivas
model_space_4 = {
    "model_class": "DualOutputMLPModel",
    "output_size": 1,
    "hidden_size": 128,
    "num_layers": 3,
    "dropout": 0.2,
    "lr": 0.001,
    "batch_size": 64,
    "num_epochs": 10,
    "es_patience": 3,
    "reg_type": None,
    "lambda_reg": 1e-4,
    "device": "cpu",
    
    # Parámetros de capas exclusivas
    "num_exclusive_layers": 3,  # 3 capas exclusivas por output
    "exclusive_hidden_size": 96,
}

print("\n📝 Ejemplo 4: Con múltiples capas exclusivas")
print("   num_exclusive_layers: 3")
print("   exclusive_hidden_size: 96")
print("   → Redes más profundas en las ramas exclusivas")

# ========================================
# PARTE 2: Configuraciones optimizables (búsqueda de hiperparámetros)
# ========================================

print("\n\n" + "="*80)
print("PARTE 2: CONFIGURACIONES OPTIMIZABLES (búsqueda con Optuna)")
print("="*80)

# Ejemplo 5: Optimizar si usar capas exclusivas o no
model_space_5 = {
    "model_class": "DualOutputMLPModel",
    "output_size": 1,
    "hidden_size": 128,
    "num_layers": 3,
    "dropout": 0.2,
    "lr": (float, 1e-4, 1e-2, True),  # Optimizable (log scale)
    "batch_size": 64,
    "num_epochs": 10,
    "es_patience": 3,
    "reg_type": None,
    "lambda_reg": 1e-4,
    "device": "cpu",
    
    # Parámetros de capas exclusivas optimizables
    "num_exclusive_layers": (int, 0, 2),  # Buscar entre 0, 1, 2 capas
}

print("\n📝 Ejemplo 5: Optimizar número de capas exclusivas")
print("   num_exclusive_layers: (int, 0, 2)")
print("   → Optuna probará 0 (sin exclusive), 1 o 2 capas")

# Ejemplo 6: Optimizar tamaño de capas exclusivas
model_space_6 = {
    "model_class": "DualOutputMLPModel",
    "output_size": 1,
    "hidden_size": (int, 64, 256, 64),  # Buscar: 64, 128, 192, 256
    "num_layers": 3,
    "dropout": 0.2,
    "lr": (float, 1e-4, 1e-2, True),
    "batch_size": 64,
    "num_epochs": 10,
    "es_patience": 3,
    "reg_type": None,
    "lambda_reg": 1e-4,
    "device": "cpu",
    
    # Parámetros de capas exclusivas optimizables
    "num_exclusive_layers": 1,  # Fijo en 1
    "exclusive_hidden_size": (int, 32, 128, 32),  # Buscar: 32, 64, 96, 128
}

print("\n📝 Ejemplo 6: Optimizar tamaño de capas exclusivas")
print("   num_exclusive_layers: 1 (fijo)")
print("   exclusive_hidden_size: (int, 32, 128, 32)")
print("   → Optuna probará tamaños 32, 64, 96, 128")

# Ejemplo 7: Optimizar ambos parámetros simultáneamente
model_space_7 = {
    "model_class": "DualOutputMLPModel",
    "output_size": 1,
    "hidden_size": (int, 128, 256, 64),
    "num_layers": (int, 2, 4),
    "dropout": (float, 0.0, 0.5),
    "lr": (float, 1e-4, 1e-2, True),
    "batch_size": 64,
    "num_epochs": 10,
    "es_patience": 3,
    "reg_type": None,
    "lambda_reg": 1e-4,
    "device": "cpu",
    
    # Ambos parámetros optimizables
    "num_exclusive_layers": (int, 0, 3),  # 0 a 3 capas
    "exclusive_hidden_size": (int, 64, 256, 64),  # 64 a 256 en pasos de 64
}

print("\n📝 Ejemplo 7: Optimizar ambos parámetros")
print("   num_exclusive_layers: (int, 0, 3)")
print("   exclusive_hidden_size: (int, 64, 256, 64)")
print("   → Optuna explorará todas las combinaciones posibles")

# ========================================
# PARTE 3: Casos especiales y consideraciones
# ========================================

print("\n\n" + "="*80)
print("PARTE 3: CASOS ESPECIALES Y CONSIDERACIONES")
print("="*80)

print("\n💡 Consideraciones importantes:")
print()
print("1. num_exclusive_layers = 0:")
print("   • Desactiva completamente las capas exclusivas")
print("   • Se traduce internamente a use_exclusive=False")
print("   • El modelo es más ligero (menos parámetros)")
print()
print("2. exclusive_hidden_size no especificado:")
print("   • Por defecto usa el mismo valor que hidden_size")
print("   • Útil cuando querés mantener la misma capacidad")
print()
print("3. num_exclusive_layers > 1:")
print("   • Crea redes más profundas en las ramas exclusivas")
print("   • Aumenta la capacidad de especialización")
print("   • Aumenta el número total de parámetros")
print()
print("4. Trade-offs:")
print("   • Más capas exclusivas = más especialización pero más parámetros")
print("   • Capas más pequeñas = menos parámetros pero menos capacidad")
print("   • Sin exclusive (0) = más rápido pero menos especialización")

# ========================================
# PARTE 4: Uso con run_study
# ========================================

print("\n\n" + "="*80)
print("PARTE 4: EJEMPLO COMPLETO CON run_study")
print("="*80)

print("""
from ibioml.tuner import run_study
import numpy as np

# Datos sintéticos de ejemplo
X = np.random.randn(1000, 50)
y = np.random.randn(1000, 2)  # 2 columnas: posición y velocidad
T = np.repeat(np.arange(10), 100)  # 10 trials de 100 muestras cada uno

# Configuración del model_space con parámetros de capas exclusivas
model_space = {
    "model_class": "DualOutputMLPModel",
    "output_size": 1,
    "device": "cpu",
    "num_epochs": 50,
    "es_patience": 5,
    "reg_type": None,
    "lambda_reg": 1e-4,
    "batch_size": 64,
    
    # Hiperparámetros a optimizar
    "hidden_size": (int, 64, 256, 64),      # 64, 128, 192, 256
    "num_layers": (int, 2, 4),              # 2, 3, 4
    "dropout": (float, 0.0, 0.5),           # 0.0 a 0.5
    "lr": (float, 1e-4, 1e-2, True),        # 0.0001 a 0.01 (log scale)
    
    # Parámetros de capas exclusivas (optimizables)
    "num_exclusive_layers": (int, 0, 3),    # 0 (sin), 1, 2, 3 capas
    "exclusive_hidden_size": (int, 64, 192, 64),  # 64, 128, 192
}

# Ejecutar búsqueda
results = run_study(
    X, y, T,
    model_space=model_space,
    num_trials=20,          # 20 trials de Optuna por fold
    outer_folds=5,          # 5 folds externos
    save_path="results",
    study_name="mlp_exclusive_optimization"
)
""")

print("\n" + "="*80)
print("✅ RESUMEN")
print("="*80)
print("""
Ahora podés controlar completamente las capas exclusivas desde model_space:

• Valores fijos: num_exclusive_layers=0, num_exclusive_layers=1, etc.
• Rangos optimizables: (int, low, high, [step])
• Combinaciones: Mezclar fijos y optimizables según necesites

La búsqueda de hiperparámetros encontrará la mejor configuración
de capas exclusivas para tus datos automáticamente!
""")
print("="*80 + "\n")
