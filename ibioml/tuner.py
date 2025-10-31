import numpy as np
import torch
import datetime
import optuna
import os
import json
import copy
from ibioml.trainer import train_model
from ibioml.utils.model_factory import create_model_class
from ibioml.utils.evaluators import create_evaluator, ModelEvaluator
from ibioml.utils.splitters import TrialKFold, trial_train_test_split
from ibioml.utils.tuner_funcs import *
from sklearn.model_selection import KFold, train_test_split


class Tuner:
    """
    Afinador de hiperparámetros con validación cruzada anidada.
    
    Esta clase maneja la optimización de hiperparámetros usando Optuna y CV anidada.
    Soporta modelos de salida única y dual.
    """
    
    def __init__(self, model_space=None, evaluator: ModelEvaluator = None):
        """Inicializa el afinador con una configuración de modelo opcional y un evaluador."""
        self.model_space = model_space
        self.evaluator = evaluator
        self.save_path = "results"
        self.study_name = None
        self.num_trials = 5
        self.outer_folds = 5
        self.inner_folds = 5
        self.search_alg = "bayes"
        self.search_space = None
    
    def set_model_space(self, model_space):
        """Establece la configuración del espacio de modelos."""
        self.model_space = model_space
        return self
    
    def set_study_params(self, save_path="results", study_name=None, num_trials=5, 
                        outer_folds=5, inner_folds=5, search_alg="bayes", 
                        search_space=None):
        """Establece todos los parámetros del estudio a la vez."""
        self.save_path = save_path
        self.study_name = study_name
        self.num_trials = num_trials
        self.outer_folds = outer_folds
        self.inner_folds = inner_folds
        self.search_alg = search_alg
        self.search_space = search_space
        return self
    
    def set_evaluator(self, evaluator: ModelEvaluator):
        """Establece la instancia del evaluador."""
        self.evaluator = evaluator
        return self
    
    def run(self, X, y, T):
        """Ejecuta la optimización de hiperparámetros con CV anidada."""
        # Setup directories
        experiment_dir, training_results_dir = self._setup_directories()
        
        # Initialize results storage
        results = self._initialize_results()
        
        # Run outer CV loop
        self._run_outer_cv_loop(X, y, T, results, training_results_dir)
        
        # Save final results
        self._save_final_results(results, experiment_dir)
        
        # Print summary
        self._print_cv_summary(results)
        
        return results
    
    def _setup_directories(self):
        """Crea la estructura de directorios para los resultados del experimento."""
        if self.study_name:
            experiment_dir = os.path.join(self.save_path, self.study_name)
        else:    
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            experiment_dir = os.path.join(self.save_path, f"study_{timestamp}")
        
        training_results_dir = os.path.join(experiment_dir, "training_results")
        
        os.makedirs(experiment_dir, exist_ok=True)
        os.makedirs(training_results_dir, exist_ok=True)
        
        return experiment_dir, training_results_dir
    
    def _initialize_results(self):
        """Inicializa el diccionario de resultados usando el evaluador."""
        return self.evaluator.initialize_results()
    
    def _run_outer_cv_loop(self, X, y, T, results, training_results_dir):
        """Ejecuta el bucle externo de validación cruzada."""
        outer_cv = TrialKFold(n_splits=self.outer_folds, shuffle=True, random_state=42)
        
        for fold_idx, (train_val_idx, test_idx) in enumerate(outer_cv.split(trial_markers=T)):
            print(f"\n===== Outer Fold {fold_idx+1}/{self.outer_folds} =====")
            
            # Divide los datos para este fold
            X_train_val, X_test = X[train_val_idx], X[test_idx]
            y_train_val, y_test = y[train_val_idx], y[test_idx]
            T_train_val = T[train_val_idx]
            
            # Crea un directorio para el fold
            fold_dir = os.path.join(training_results_dir, f"fold_{fold_idx+1}")
            os.makedirs(fold_dir, exist_ok=True)
            
            # Divide datos para validación interna
            X_train, X_val, y_train, y_val, train_mask, val_mask = trial_train_test_split(
                X_train_val, y_train_val, T_train_val, train_size=0.8, random_state=42, shuffle=True, return_mask=True
            )

            train_trials = [int(t) for t in np.unique(T_train_val[train_mask])]
            val_trials = [int(t) for t in np.unique(T_train_val[val_mask])]
            test_trials = [int(t) for t in np.unique(T[test_idx])]
            
            # Optimiza para este fold
            best_model, final_config, study, scaler = self._optimize_fold(
                fold_idx, fold_dir, X_train, y_train, X_val, y_val
            )
            
            # Registra la mejor configuración
            results["best_config_per_fold"].append(make_serializable(final_config))
            
            # Evalúa modelo
            fold_results = self._evaluate_model(
                best_model, scaler, X_test, y_test, X_val, y_val, study, train_trials, val_trials, test_trials
            )
            
            # Actualiza resultados
            self._update_results(results, fold_results)
            
            # Guarda resultados del fold
            self._save_fold_results(fold_dir, fold_results, study)
    
    def _optimize_fold(self, fold_idx, fold_dir, X_train, y_train, X_val, y_val):
        """Optimiza hiperparámetros para un solo fold."""
        # Inicializa configuración y obtiene escalador
        config, scaler = initialize_config(
            self.model_space, X_train, y_train, X_val, y_val, get_scaler=True
        )
        
        # Define función objetivo
        def objective_fn(trial):
            return self._objective_function(trial, config, fold_dir)
        
        # Crea y ejecuta estudio de Optuna
        study = self._create_study()
        study.optimize(objective_fn, n_trials=self.num_trials)
        
        # Imprime resultados
        self._print_study_results(study)
        
        # Carga el mejor modelo
        best_model_path = os.path.join(fold_dir, "best_model.pt")
        best_model = torch.load(best_model_path, weights_only=False)
        
        # Crea configuración final
        final_config = copy.deepcopy(config)
        for key, value in study.best_params.items():
            final_config[key] = value
        
        return best_model, final_config, study, scaler
    
    def _objective_function(self, trial, base_config, fold_dir):
        """Función objetivo para la optimización de Optuna."""
        # Crea una copia de la configuración base
        trial_config = copy.deepcopy(base_config)
        
        search_space = self.search_space or {
            "hidden_size": {"type": "int", "low": 128, "high": 512, "step": 64},
            "num_layers": {"type": "int", "low": 1, "high": 3},
            "dropout": {"type": "float", "low": 0.0, "high": 0.5},
            "lr": {"type": "float", "low": 1e-8, "high": 1e-3, "log": True},
        }
        for param, opts in search_space.items():
            if opts["type"] == "int":
                step = opts.get("step")
                if step is not None:
                    trial_config[param] = trial.suggest_int(param, opts["low"], opts["high"], step=step)
                else:
                    trial_config[param] = trial.suggest_int(param, opts["low"], opts["high"])
            elif opts["type"] == "float":
                trial_config[param] = trial.suggest_float(param, opts["low"], opts["high"], log=opts.get("log", False))
        
        # Maneja casos especiales para modelos RNN
        if hasattr(trial_config["model_class"], "__name__") and "RNN" in trial_config["model_class"].__name__:
            if trial_config["num_layers"] == 1:
                trial_config["dropout"] = 0.0  # Avoid PyTorch warning for single-layer RNNs
        
        # Crea directorio para esta prueba
        model_dir = os.path.join(fold_dir, f"trial_{trial.number}")
        os.makedirs(model_dir, exist_ok=True)
        trial_config["results_dir"] = model_dir
        
        # Entrena el modelo
        results_dict = train_model(trial_config)
        current_r2 = results_dict["r2_score"]
        
        # Guarda el modelo si es el mejor hasta ahora
        if trial.number == 0 or current_r2 > trial.study.best_value:
            print(f"New best model found! R² = {current_r2:.4f}")
            torch.save(results_dict["model"], os.path.join(fold_dir, "best_model.pt"))
        
        return current_r2
    
    def _create_study(self):
        """Crea un estudio de Optuna con el algoritmo de búsqueda especificado."""
        if self.search_alg == "bayes":
            return optuna.create_study(
                direction="maximize",
                sampler=optuna.samplers.TPESampler()
            )
        elif self.search_alg == "grid":
            return optuna.create_study(
                direction="maximize",
                sampler=optuna.samplers.GridSampler(self.search_space)
            )
    
    def _print_study_results(self, study):
        """Imprime los resultados de un estudio de Optuna."""
        print(f"Best trial number: {study.best_trial.number}")
        print(f"Best hyperparameters: {study.best_params}")
        print(f"Best Validation R² score: {study.best_value:.4f}")
    
    def _evaluate_model(self, model, scaler, X_test, y_test, X_val, y_val, study, train_trials, val_trials, test_trials):
        """Evalúa un modelo en datos de prueba y validación usando el evaluador."""
        # Escala datos de prueba
        X_test_scaled = scaler.feature_scaler.transform(X_test)
        X_val_scaled = scaler.feature_scaler.transform(X_val)
        
        # Inicializa diccionario de resultados
        fold_results = {}
        
        # Evalúa en datos de prueba
        model.eval()
        with torch.no_grad():
            # Configura tensores
            X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32).to(self.model_space["device"])
            
            # Obtiene predicciones de prueba y métricas
            y_test_scaled = self.evaluator.prepare_evaluation_data(scaler, y_test)
            predictions = self.evaluator.predict(model, X_test_tensor)
            r2_scores = self.evaluator.calculate_r2(y_test_scaled, predictions)
            predictions_rescaled = self.evaluator.rescale_predictions(scaler, predictions)
            fold_results["predictions"], fold_results["true_values"] = self.evaluator.prepare_results(predictions_rescaled, y_test)
            fold_results["r2_scores"] = r2_scores
            
            # Imprime resultados
            print(f"Test R² Score: {r2_scores[0]:.4f}")
            if len(r2_scores) > 1:
                print(f"Additional R² Scores: {r2_scores[1:]}")
            
            # Evalúa en datos de validación (comprobación de coherencia)
            X_val_tensor = torch.tensor(X_val_scaled, dtype=torch.float32).to(self.model_space["device"])
            y_val_scaled = self.evaluator.prepare_evaluation_data(scaler, y_val)
            val_predictions = self.evaluator.predict(model, X_val_tensor)
            val_r2_scores = self.evaluator.calculate_r2(y_val_scaled, val_predictions)
            
            # Añade métricas de validación a los resultados
            fold_results["val_r2"] = val_r2_scores[0]
            if len(val_r2_scores) > 1:
                fold_results["val_r2_scores"] = val_r2_scores[1:]
            
            # Añade información de los trials
            fold_results["train_trials"] = train_trials
            fold_results["val_trials"] = val_trials
            fold_results["test_trials"] = test_trials
            
            # Añade información del estudio
            fold_results["study"] = {
                "best_trial_number": study.best_trial.number,
                        "best_validation_r2_score": val_r2_scores[0]
            }
        
        return fold_results
    
    def _update_results(self, results, fold_results):
        """Actualiza los resultados globales usando el evaluador."""
        return self.evaluator.update_results(results, fold_results)
    
    def _save_fold_results(self, fold_dir, fold_results, study):
        """Guarda los resultados de un fold en un archivo JSON."""
        outer_fold_results = {
            "best_trial_number": study.best_trial.number,
            "best_validation_r2_score": fold_results["val_r2"],
            "test_r2_score": fold_results["r2_scores"][0],
            "train_trials": fold_results["train_trials"],
            "val_trials": fold_results["val_trials"],
            "test_trials": fold_results["test_trials"]
        }
        
        if len(fold_results["r2_scores"]) > 1:
            outer_fold_results["additional_test_r2_scores"] = fold_results["r2_scores"][1:]
            outer_fold_results["additional_val_r2_scores"] = fold_results.get("val_r2_scores", [])
        
        with open(f"{fold_dir}/results.json", "w") as f:
            json.dump(outer_fold_results, f, indent=4, default=str)
    
    def _save_final_results(self, results, experiment_dir):
        """Guarda los resultados finales en un archivo JSON."""
        # Asegurarse de que los resultados sean serializables
        json_results = {}
        
        # Copiar todos los campos existentes
        for key, value in results.items():
            json_results[key] = make_serializable(value)
        
        # Guardar en formato JSON
        with open(f"{experiment_dir}/final_results.json", "w") as f:
            json.dump(json_results, f, default=str)
    
    def _print_cv_summary(self, results):
        """Imprime un resumen de los resultados de validación cruzada usando el evaluador."""
        print("\n===== Nested CV Summary =====")
        self.evaluator.print_summary(results)       


def run_study(X, y, T, model_space, num_trials=5, outer_folds=5, inner_folds=5, save_path="results", search_alg="bayes", search_space=None, study_name=None):
    """
    Ejecuta una validación cruzada anidada para la optimización de hiperparámetros utilizando Optuna.

    Esta función realiza una búsqueda de hiperparámetros mediante Optuna, empleando validación cruzada anidada.
    Permite la evaluación de modelos con salida simple o múltiple, y almacena los resultados y configuraciones óptimas.

    Args:
        X (np.ndarray): Matriz de características de entrada.
        y (np.ndarray): Matriz o vector de etiquetas objetivo.
        T (np.ndarray): Vector de marcadores de ensayo para la división de folds.
        model_space (dict): Diccionario con la configuración base del modelo y parámetros fijos.
        num_trials (int, optional): Número de pruebas de Optuna por fold interno. Por defecto 5.
        outer_folds (int, optional): Número de folds para la validación cruzada externa. Por defecto 5.
        inner_folds (int, optional): Número de folds para la validación cruzada interna. Por defecto 5.
        save_path (str, optional): Ruta base donde guardar los resultados. Por defecto "results".
        search_alg (str, optional): Algoritmo de búsqueda de Optuna ("bayes" o "grid"). Por defecto "bayes".
        search_space (dict, optional): Espacio de búsqueda para GridSampler. Por defecto None.
        study_name (str, optional): Nombre del estudio para la carpeta de resultados. Por defecto None.

    Returns:
        bool: True si la optimización se ejecutó correctamente.

    Raises:
        ValueError: Si el diccionario model_space no contiene exactamente las claves requeridas.
    """
    # Separa fijos y optimizables
    fixed_space, auto_search_space = split_model_space(model_space)
    # Usa el search_space explícito si lo pasan, sino el generado
    search_space = search_space or auto_search_space

    # Solo chequea los mínimos requeridos
    required_keys = {
        "model_class",
        "output_size",
        "device",
        "num_epochs",
        "es_patience",
        "reg_type",
        "lambda_reg",
        "batch_size",
    }
    missing = required_keys - set(fixed_space.keys())
    if missing:
        raise ValueError(
            f"model_space must contain at least these keys: {sorted(required_keys)}. "
            f"Missing: {sorted(missing)}"
        )

    evaluator = create_evaluator(y.shape[1], fixed_space["device"])
    model_class = create_model_class(fixed_space["model_class"], y.shape[1])
    fixed_space["model_class"] = model_class

    tuner = Tuner(fixed_space, evaluator)
    tuner.set_study_params(
        save_path=save_path,
        study_name=study_name,
        num_trials=num_trials,
        outer_folds=outer_folds,
        inner_folds=inner_folds,
        search_alg=search_alg,
        search_space=search_space
    )
    tuner.run(X, y, T)
    return True