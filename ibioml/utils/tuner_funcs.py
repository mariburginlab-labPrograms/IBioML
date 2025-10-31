from ibioml.utils.trainer_funcs import create_dataloaders
from ibioml.utils.data_scaler import scale_data
import copy

def split_model_space(model_space):
    """
    Separa los hiperparámetros fijos de los optimizables (tuplas) en model_space.
    Devuelve (fixed_space, search_space)
    
    Soporta parámetros especiales para capas exclusivas:
    - num_exclusive_layers: int o (int, low, high, [step]) - Si es 0, desactiva capas exclusivas
    - exclusive_hidden_size: int o (int, low, high, [step]) - Tamaño de capas exclusivas
    """
    fixed_keys = {
        "model_class", "output_size", "device", "num_epochs",
        "es_patience", "reg_type", "lambda_reg", "batch_size"
    }
    fixed = {k: v for k, v in model_space.items() if k in fixed_keys or not isinstance(v, tuple)}
    search_space = {}
    for k, v in model_space.items():
        if isinstance(v, tuple):
            # INT: (int, low, high, [step])
            if v[0] == int:
                if len(v) == 4:
                    search_space[k] = {"type": "int", "low": v[1], "high": v[2], "step": v[3]}
                else:
                    search_space[k] = {"type": "int", "low": v[1], "high": v[2]}
            # FLOAT: (float, low, high, [log])
            elif v[0] == float:
                if len(v) == 4:
                    search_space[k] = {"type": "float", "low": v[1], "high": v[2], "log": v[3]}
                else:
                    search_space[k] = {"type": "float", "low": v[1], "high": v[2]}
    return fixed, search_space

# Keep original functions for backward compatibility
def initialize_config(config, X_train, y_train, X_val, y_val, get_scaler=False):
    """
    Inicializa la configuración para el entrenamiento.
    Maneja automáticamente modelos de salida única y dual.
    Soporta configuración de capas exclusivas vía num_exclusive_layers y exclusive_hidden_size.
    """
    # Detectar automáticamente la dimensión de salida
    if y_train.ndim > 1 and y_train.shape[1] == 2:
        y_dim = 2
    else:
        y_dim = 1
    
    # Escalar datos
    X_train_scaled, y_train_scaled, X_val_scaled, y_val_scaled, scaler = scale_data(
        X_train, y_train, X_val, y_val, return_scaler=True
    )
    
    # Crear DataLoaders
    train_loader, val_loader = create_dataloaders(
        X_train_scaled, y_train_scaled, X_val_scaled, y_val_scaled, config["batch_size"]
    )
    
    # Configurar modelo y otros parámetros
    train_config = copy.deepcopy(config)
    train_config["input_size"] = X_train.shape[X_train.ndim-1]
    train_config["train_loader"] = train_loader
    train_config["val_loader"] = val_loader
    train_config["scaler"] = scaler  # Store scaler in config
    
    # Configuración específica según la dimensión de salida
    if y_dim == 2:
        train_config["output_size"] = 1  # Cada cabeza predice un valor
    else:
        train_config["y_scaler"] = scaler.target_scaler
    
    # Manejar parámetros de capas exclusivas (solo para modelos dual)
    if y_dim == 2:
        # Convertir num_exclusive_layers=0 a use_exclusive=False
        if "num_exclusive_layers" in train_config:
            if train_config["num_exclusive_layers"] == 0:
                train_config["use_exclusive"] = False
                # No pasar num_exclusive_layers ni exclusive_hidden_size si use_exclusive=False
                train_config.pop("num_exclusive_layers", None)
                train_config.pop("exclusive_hidden_size", None)
            else:
                train_config["use_exclusive"] = True
                # num_exclusive_layers y exclusive_hidden_size se pasan tal cual
        # Si no está especificado, usar el comportamiento por defecto (use_exclusive=True, 1 capa)
    
    if get_scaler:
        return train_config, scaler
    else:
        return train_config


def make_serializable(obj):
    """Convierte un objeto complejo a una forma serializable en JSON."""
    if isinstance(obj, dict):
        return {k: make_serializable(v) for k, v in obj.items() 
                if not callable(v) and k not in ['device', 'model_class', 'train_loader', 'val_loader', 'y_scaler', 'scaler']}
    elif isinstance(obj, list):
        return [make_serializable(item) for item in obj]
    elif not isinstance(obj, (str, int, float, bool, type(None))):
        return str(obj)
    else:
        return obj
