from ibioml.models import (
    DualOutputMLPModel, DualOutputRNNModel, DualOutputGRUModel, DualOutputLSTMModel
)
import json
from ibioml.utils.trainer_funcs import initialize_weights
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
import time
import pandas as pd
from sklearn.model_selection import KFold
from ibioml.utils.trainer_funcs import *
from ibioml.utils.data_scaler import scale_data

#------------------- Clases Entrenador -------------------#

# Clases Entrenador refactorizadas con patrón de método plantilla
class BaseTrainer:
    """Clase Entrenador base con funcionalidad compartida."""
    def __init__(self, config):
        self.config = config
        if "train_loader" in config and "val_loader" in config:
            self.initialize_model(config)
            self.train_loader = config["train_loader"]
            self.X_val, self.y_val = config["val_loader"].dataset.tensors[0].to(self.config["device"]), config["val_loader"].dataset.tensors[1].to(self.config["device"])
        self.losses = []
        self.metrics = {
            "train_loss": [],
            "val_loss": [],
            "r2_score": [],
            "epoch": [],
            "weight_mean": [],
            "weight_std": [],
            "grad_mean": [],
            "grad_std": [],
            "epoch_time_ms": [],
            "split": []
        }
        self.results = {}
        self.results_dir = config["results_dir"]
        self.model_name = config["model_class"].__name__
        self.run_id = f"{self.model_name}_{int(time.time())}"
        self.config["run_id"] = self.run_id
        self.config["model_name"] = self.model_name

    def initialize_model(self, config):
        """Inicializar el modelo, el criterio y el optimizador."""
        # Parámetros base del modelo
        model_kwargs = {
            "input_size": config["input_size"],
            "hidden_size": config["hidden_size"],
            "output_size": config["output_size"],
            "num_layers": config["num_layers"],
            "dropout": config["dropout"]
        }
        
        # Agregar parámetros opcionales de capas exclusivas si están presentes
        if "use_exclusive" in config:
            model_kwargs["use_exclusive"] = config["use_exclusive"]
        if "exclusive_hidden_size" in config:
            model_kwargs["exclusive_hidden_size"] = config["exclusive_hidden_size"]
        if "num_exclusive_layers" in config:
            model_kwargs["num_exclusive_layers"] = config["num_exclusive_layers"]
        
        self.model = config["model_class"](**model_kwargs).to(config["device"])
        initialize_weights(self.model) 
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=config["lr"])

    def apply_regularization(self, loss):
        """Aplicar regularización a la pérdida."""
        if self.config["reg_type"] is None:
            return loss

        for name, param in self.model.named_parameters():
            if 'weight' in name:
                # Aplicar regularización L1
                if self.config["reg_type"] == 'L1':
                    l1_norm = param.abs().sum()
                    loss += self.config["lambda_reg"] * l1_norm
                
                # Aplicar regularización L2
                elif self.config["reg_type"] == 'L2':
                    l2_norm = param.pow(2).sum()
                    loss += self.config["lambda_reg"] * l2_norm
        
        return loss

    def train_one_epoch(self):
        """Método plantilla para entrenar una época."""
        self.model.train()
        epoch_loss = 0
        epoch_loss_sin_norm = 0

        for batch_idx, (data, target) in enumerate(self.train_loader):
            data = data.to(self.config["device"])
            
            # Procesar objetivos de manera diferente según el tipo de modelo
            processed_target = self.process_target(target)
            
            self.optimizer.zero_grad()
            
            # Paso hacia adelante (manejado por subclases)
            outputs = self.forward_pass(data.float())
            
            # Calcular pérdida (manejado por subclases)
            loss = self.calculate_loss(outputs, processed_target)
            
            if self.config["reg_type"] is not None:
                epoch_loss_sin_norm += loss.item()
                loss = self.apply_regularization(loss)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            epoch_loss += loss.item()

        average_train_loss = epoch_loss / len(self.train_loader)
        if self.config["reg_type"] is not None:
            average_train_loss_sin_norm = epoch_loss_sin_norm / len(self.train_loader)
            return average_train_loss, average_train_loss_sin_norm
        else:
            return average_train_loss, None

    # Métodos abstractos a implementar por subclases
    def process_target(self, target):
        """Procesar datos objetivo según el tipo de modelo."""
        raise NotImplementedError("Las subclases deben implementar process_target")
    
    def forward_pass(self, data):
        """Realizar paso hacia adelante según el tipo de modelo."""
        raise NotImplementedError("Las subclases deben implementar forward_pass")
    
    def calculate_loss(self, outputs, targets):
        """Calcular pérdida según el tipo de modelo."""
        raise NotImplementedError("Las subclases deben implementar calculate_loss")
    
    def evaluate_model(self):
        """Evaluar el modelo."""
        raise NotImplementedError("Las subclases deben implementar evaluate_model")

    def calculate_statistics(self):
        """Calculate statistics for weights and gradients of the model."""
        weights = []
        gradients = []
        
        for param in self.model.parameters():
            if param.requires_grad:
                weights.append(param.data.cpu().numpy())
                # Verificar si el gradiente existe
                if param.grad is not None:
                    gradients.append(param.grad.cpu().numpy())
        
        weights = np.concatenate([w.flatten() for w in weights])
        
        # Verificar si hay gradientes disponibles
        if gradients:
            gradients = np.concatenate([g.flatten() for g in gradients])
            grad_mean = np.mean(gradients)
            grad_std = np.std(gradients)
        else:
            # Si no hay gradientes, usar valores predeterminados
            grad_mean = 0.0
            grad_std = 0.0
        
        weight_mean = np.mean(weights)
        weight_std = np.std(weights)
        
        return weight_mean, weight_std, grad_mean, grad_std

    def report_metrics(self, average_train_loss, val_loss, r2_score, epoch, average_train_loss_sin_norm=None, y_pred=None, epoch_time_ms=None):
        """Reportar métricas de entrenamiento."""
        weight_mean, weight_std, grad_mean, grad_std = self.calculate_statistics()
        
        # Guardar métricas en el diccionario
        self.metrics["train_loss"].append(average_train_loss)
        self.metrics["val_loss"].append(val_loss)
        self.metrics["r2_score"].append(r2_score)
        self.metrics["epoch"].append(epoch)
        self.metrics["weight_mean"].append(weight_mean)
        self.metrics["weight_std"].append(weight_std)
        self.metrics["grad_mean"].append(grad_mean)
        self.metrics["grad_std"].append(grad_std)
        self.metrics["epoch_time_ms"].append(epoch_time_ms)
        
        if "split_idx" in self.config:
            self.metrics["split"].append(self.config["split_idx"] + 1)
        else:
            self.metrics["split"].append(0)
        
        # Guardar métricas adicionales si están disponibles
        if average_train_loss_sin_norm is not None:
            if "train_loss_sin_norm" not in self.metrics:
                self.metrics["train_loss_sin_norm"] = []
            self.metrics["train_loss_sin_norm"].append(average_train_loss_sin_norm)
        
        # Imprimir el progreso
        if "split_idx" in self.config:               
            print(f'Epoch [{epoch}/{self.config["num_epochs"]}], Split: {self.config["split_idx"] + 1}, Train Loss: {average_train_loss:.4f}, Val Loss: {val_loss:.4f}, R2 Score: {r2_score:.4f}')
        else:
            print(f'Epoch [{epoch}/{self.config["num_epochs"]}], Train Loss: {average_train_loss:.4f}, Val Loss: {val_loss:.4f}, R2 Score: {r2_score:.4f}')
        print(f'Weight Mean: {weight_mean:.4f}, Weight Std: {weight_std:.4f}, Grad Mean: {grad_mean:.4f}, Grad Std: {grad_std:.4f}')
        if epoch_time_ms is not None:
            print(f'Epoch Time: {epoch_time_ms:.2f} ms')
        if average_train_loss_sin_norm is not None:
            print(f'Train Loss sin norm: {average_train_loss_sin_norm:.4f}')
        
        # Guardar resultados finales cuando termine el entrenamiento
        if self.done:
            best_val_loss, best_r2_score, best_y_pred = self.evaluate_model()
            self.results["r2_score"] = best_r2_score
            self.results["model"] = self.model
            self.results["config"] = {k: v for k, v in self.config.items() if not callable(v) and k not in ['dataset', 'device', 'train_loader', 'val_loader', 'y_scaler']}
            self._save_final_results(best_y_pred)
            
    def _save_final_results(self, y_pred):
        """Guardar resultados finales incluyendo métricas separadas."""
        # Asegurarse de que todas las métricas tengan la misma longitud
        # sin rellenar con NaNs
        min_length = min(len(v) for v in self.metrics.values() if isinstance(v, list))
        
        # Recortar arrays a la longitud mínima
        adjusted_metrics = {}
        for key, value in self.metrics.items():
            if isinstance(value, list):
                adjusted_metrics[key] = value[:min_length]  # Recortar al mínimo
            else:
                adjusted_metrics[key] = value
        
        # Crear el DataFrame con las métricas ajustadas
        metrics_df = pd.DataFrame(adjusted_metrics)
        metrics_path = os.path.join(self.results_dir, f"{self.run_id}_metrics.csv")
        metrics_df.to_csv(metrics_path, index=False)
        
        config_to_save = self.results["config"]
        config_to_save["r2_score"] = self.results["r2_score"]
        
        # Añadir R2 individuales
        if "r2_score_pos" in self.results:
            config_to_save["r2_score_pos"] = self.results["r2_score_pos"]
        if "r2_score_vel" in self.results:
            config_to_save["r2_score_vel"] = self.results["r2_score_vel"]
        
        config_path = os.path.join(self.results_dir, f"{self.run_id}_config.json")
        with open(config_path, 'w') as f:
            json.dump(config_to_save, f, indent=4, default=str)
        
        print(f"Resultados guardados en {self.results_dir}")

    def train(self):
        """Entrenar el modelo."""
        # La implementación sigue siendo la misma que en el Entrenador original
        epoch = 0
        self.done = False
        if self.config["es_patience"] is not None:
            es = EarlyStopping(patience=self.config["es_patience"])
    
        while not self.done and epoch < self.config["num_epochs"]:
            epoch_start_time = time.time()
            
            epoch += 1
            average_train_loss, average_train_loss_sin_norm = self.train_one_epoch()
            self.losses.append(average_train_loss)

            val_loss, r2_score, y_pred = self.evaluate_model()
            
            epoch_time_ms = (time.time() - epoch_start_time) * 1000
            
            if self.config["es_patience"] is not None:
                self.done = es(self.model, val_loss) 

            if epoch == self.config["num_epochs"]:
                self.done = True
            
            self.report_metrics(average_train_loss, val_loss, r2_score, epoch, average_train_loss_sin_norm, y_pred, epoch_time_ms)
    
    def train_with_cv(self):
        """Entrenar con validación cruzada."""
        # La implementación sigue siendo la misma que en el Entrenador original
        X_tensor, y_tensor = self.config["dataset"].tensors
        X = X_tensor.numpy()
        y = y_tensor.numpy()
        kf = KFold(n_splits=self.config["n_splits"])

        for split_idx, (train_idx, val_idx) in enumerate(kf.split(X)):
            print(f"\nSplit {split_idx + 1}/{self.config['n_splits']}")
            
            X_train = X[train_idx]
            y_train = y[train_idx]
            X_val = X[val_idx]
            y_val = y[val_idx]
            
            print(f"Train size: {X_train.shape}, Val size: {X_val.shape}")

            X_train_scaled, y_train_scaled, X_val_scaled, y_val_scaled, scaler = scale_data(X_train, y_train, X_val, y_val, return_scaler=True)
            
            self.train_loader, self.val_loader = create_dataloaders(X_train_scaled, y_train_scaled, X_val_scaled, y_val_scaled, self.config["batch_size"])
            
            self.X_val = X_val_scaled
            self.y_val = y_val_scaled
            
            self.config = configure_train_config(self.config, X_train, self.train_loader, self.val_loader, split_idx, scaler)

            self.initialize_model(self.config)
            self.train()

# Implementaciones concretas para modelos de salida única
class SingleOutputTrainer(BaseTrainer):
    def process_target(self, target):
        """Procesar objetivo para modelos de salida única."""
        return target.to(self.config["device"])
    
    def forward_pass(self, data):
        """Paso hacia adelante para modelos de salida única."""
        return self.model(data)
    
    def calculate_loss(self, output, target):
        """Calcular pérdida para modelos de salida única."""
        return self.criterion(output, target.float())
    
    def evaluate_model(self):
        """Evaluar modelos de salida única."""
        self.model.eval()
        with torch.no_grad():
            y_pred = self.model(self.X_val).cpu().numpy()
            y_pred_tensor = torch.tensor(y_pred, dtype=torch.float32).to(self.config["device"])
            y_val_tensor = self.y_val.to(self.config["device"])
            val_loss = self.criterion(y_pred_tensor, y_val_tensor)
            r2_score = 1 - val_loss.item() / torch.var(y_val_tensor).item()
        
        return val_loss.item(), r2_score, y_pred

# Implementaciones concretas para modelos de doble salida
class DualOutputTrainer(BaseTrainer):
    def __init__(self, config):
        super(DualOutputTrainer, self).__init__(config)
        # Añadir campos para métricas separadas de posición y velocidad
        self.metrics.update({
            "train_loss_pos": [],
            "train_loss_vel": [],
            "val_loss_pos": [],
            "val_loss_vel": [],
            "r2_score_pos": [],
            "r2_score_vel": []
        })
    
    def process_target(self, target):
        """Procesar objetivo para modelos de doble salida, dividiendo si es necesario."""
        if isinstance(target, tuple):
            # Caso donde target ya viene como tupla
            return (target[0].to(self.config["device"]), target[1].to(self.config["device"]))
        else:
            # Caso especial: matriz con exactamente dos columnas (posición y velocidad)
            if target.shape[1] == 2:  # Detectar cuando tenemos [posición, velocidad]
                pos = target[:, 0:1].to(self.config["device"])  # Primera columna (posición)
                vel = target[:, 1:2].to(self.config["device"])  # Segunda columna (velocidad)
                return (pos, vel)
            else:
                # Caso original (para compatibilidad con implementaciones anteriores)
                target_size = target.shape[1] // 2
                target1 = target[:, :target_size].to(self.config["device"])
                target2 = target[:, target_size:].to(self.config["device"])
                return (target1, target2)
    
    def forward_pass(self, data):
        """Paso hacia adelante para modelos de doble salida."""
        return self.model(data)
    
    def calculate_loss(self, outputs, targets):
        """Calcular pérdida combinada para modelos de doble salida."""
        output1, output2 = outputs
        target1, target2 = targets
        loss1 = self.criterion(output1, target1.float())  # Loss posición
        loss2 = self.criterion(output2, target2.float())  # Loss velocidad
        
        # Guardar los valores individuales para uso posterior
        self.current_pos_loss = loss1.item()
        self.current_vel_loss = loss2.item()
        
        return loss1 + loss2  # Devolver pérdida combinada
    
    def train_one_epoch(self):
        """Entrenar una época con seguimiento de métricas separadas."""
        self.model.train()
        epoch_loss = 0
        epoch_loss_sin_norm = 0
        epoch_pos_loss = 0  # Acumulador para loss de posición
        epoch_vel_loss = 0  # Acumulador para loss de velocidad

        for batch_idx, (data, target) in enumerate(self.train_loader):
            data = data.to(self.config["device"])
            
            processed_target = self.process_target(target)
            
            self.optimizer.zero_grad()
            
            outputs = self.forward_pass(data.float())
            
            loss = self.calculate_loss(outputs, processed_target)
            
            # Acumular las pérdidas individuales
            epoch_pos_loss += self.current_pos_loss
            epoch_vel_loss += self.current_vel_loss
            
            if self.config["reg_type"] is not None:
                epoch_loss_sin_norm += loss.item()
                loss = self.apply_regularization(loss)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            epoch_loss += loss.item()

        # Calcular promedios
        average_train_loss = epoch_loss / len(self.train_loader)
        average_pos_loss = epoch_pos_loss / len(self.train_loader)
        average_vel_loss = epoch_vel_loss / len(self.train_loader)
        
        # Guardar en métricas
        self.metrics["train_loss_pos"].append(average_pos_loss)
        self.metrics["train_loss_vel"].append(average_vel_loss)
        
        if self.config["reg_type"] is not None:
            average_train_loss_sin_norm = epoch_loss_sin_norm / len(self.train_loader)
            return average_train_loss, average_train_loss_sin_norm
        else:
            return average_train_loss, None
    
    def evaluate_model(self):
        """Evaluar modelos de doble salida con métricas separadas."""
        self.model.eval()
        with torch.no_grad():
            output1, output2 = self.model(self.X_val)
            
            # Dividir objetivos de validación
            if isinstance(self.y_val, tuple):
                y_val1, y_val2 = self.y_val
            else:
                # Verificar si tenemos exactamente 2 columnas (pos, vel)
                if self.y_val.shape[1] == 2:
                    y_val1 = self.y_val[:, 0:1]
                    y_val2 = self.y_val[:, 1:2]
                else:
                    target_size = self.y_val.shape[1] // 2
                    y_val1 = self.y_val[:, :target_size]
                    y_val2 = self.y_val[:, target_size:]
            
            # Calcular pérdida para cada salida
            loss1 = self.criterion(output1, y_val1)  # Loss posición
            loss2 = self.criterion(output2, y_val2)  # Loss velocidad
            
            # Pérdida combinada
            val_loss = (loss1 + loss2).item()
            
            # Guardar loss individuales en métricas
            val_loss_pos = loss1.item()
            val_loss_vel = loss2.item()
            self.metrics["val_loss_pos"].append(val_loss_pos)
            self.metrics["val_loss_vel"].append(val_loss_vel)
            
            # Calcular puntuación R2 para ambas salidas
            r2_pos = 1 - loss1.item() / torch.var(y_val1).item()  # R2 para posición
            r2_vel = 1 - loss2.item() / torch.var(y_val2).item()  # R2 para velocidad
            
            # Guardar R2 individuales en métricas
            self.metrics["r2_score_pos"].append(r2_pos)
            self.metrics["r2_score_vel"].append(r2_vel)
            
            # Puntuación R2 promedio
            r2_score = (r2_pos + r2_vel) / 2
            
            # Combinar predicciones para el reporte
            y_pred1 = output1.cpu().numpy()
            y_pred2 = output2.cpu().numpy()
            y_pred = [y_pred1, y_pred2]
        
        return val_loss, r2_score, y_pred
    
    def report_metrics(self, average_train_loss, val_loss, r2_score, epoch, average_train_loss_sin_norm=None, y_pred=None, epoch_time_ms=None):
        """Reportar métricas de entrenamiento incluyendo las separadas para posición y velocidad."""
        # Primero llamar al método base para reportar métricas estándar
        super().report_metrics(average_train_loss, val_loss, r2_score, epoch, average_train_loss_sin_norm, y_pred, epoch_time_ms)
        
        # Reportar métricas adicionales específicas de posición y velocidad
        pos_train_loss = self.metrics["train_loss_pos"][-1]
        vel_train_loss = self.metrics["train_loss_vel"][-1]
        pos_val_loss = self.metrics["val_loss_pos"][-1]
        vel_val_loss = self.metrics["val_loss_vel"][-1]
        pos_r2 = self.metrics["r2_score_pos"][-1]
        vel_r2 = self.metrics["r2_score_vel"][-1]
        
        print(f'Posición - Train Loss: {pos_train_loss:.4f}, Val Loss: {pos_val_loss:.4f}, R2: {pos_r2:.4f}')
        print(f'Velocidad - Train Loss: {vel_train_loss:.4f}, Val Loss: {vel_val_loss:.4f}, R2: {vel_r2:.4f}')
        
        # Si es el último epoch, guardar R2 individuales en los resultados finales
        if self.done:
            self.results["r2_score_pos"] = pos_r2
            self.results["r2_score_vel"] = vel_r2
    
    def _save_final_results(self, y_pred):
        """Guardar resultados finales incluyendo métricas separadas."""
        # Primero usar la implementación base
        super()._save_final_results(y_pred)
        
        # Añadir las métricas específicas al archivo de configuración
        config_path = os.path.join(self.results_dir, f"{self.run_id}_config.json")
        
        # Leer el archivo existente
        with open(config_path, 'r') as f:
            config_data = json.load(f)
        
        # Añadir R2 individuales
        if "r2_score_pos" in self.results:
            config_data["r2_score_pos"] = self.results["r2_score_pos"]
        if "r2_score_vel" in self.results:
            config_data["r2_score_vel"] = self.results["r2_score_vel"]
        
        # Guardar el archivo actualizado
        with open(config_path, 'w') as f:
            json.dump(config_data, f, indent=4, default=str)

# Clase fábrica para crear entrenadores apropiados
class TrainerFactory:
    @staticmethod
    def create_trainer(config):
        """Crear un entrenador basado en la clase de modelo en la configuración."""
        model_class = config["model_class"]
        
        if model_class in [DualOutputMLPModel, DualOutputRNNModel, DualOutputGRUModel, DualOutputLSTMModel]:
            return DualOutputTrainer(config)
        else:
            return SingleOutputTrainer(config)

# Función train_model actualizada para usar la fábrica
def train_model(config):
    """Entrenar un modelo usando el entrenador apropiado."""
    trainer = TrainerFactory.create_trainer(config)
    
    if "n_splits" in config:
        print("n_splits:", config["n_splits"])
        trainer.train_with_cv()
        return trainer.results
    else:   
        trainer.train()
        return trainer.results