#%%
import torch
import torch.nn as nn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # Usar GPU si está disponible, de lo contrario CPU

# Seed global para reproducibilidad
GLOBAL_SEED = 42

def set_global_seed(seed=None):
    """
    Establece la semilla global para reproducibilidad.
    
    Args:
        seed (int, optional): Semilla para los generadores aleatorios. Si es None, usa GLOBAL_SEED.
    """
    if seed is None:
        seed = GLOBAL_SEED
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

# Establecer seed global al importar el módulo
set_global_seed()


#%%
#------------------- Modelos de Aprendizaje Profundo -------------------#

class BaseMLP(nn.Module):
    """Clase base MLP con funcionalidad compartida."""
    def __init__(self, input_size, hidden_size, num_layers, dropout):
        """
        Args:
            input_size (int): El número de características de entrada de forma (batch_size, input_size).
            hidden_size (int): El número de neuronas en cada capa oculta.
            num_layers (int): El número de capas ocultas.
            dropout (float): La probabilidad de dropout para regularización.
        """
        super(BaseMLP, self).__init__()
        # Crear capas compartidas
        self.shared_layers = nn.Sequential()
        self.shared_layers.append(nn.Linear(input_size, hidden_size))
        self.shared_layers.append(nn.BatchNorm1d(hidden_size))
        self.shared_layers.append(nn.ReLU())
        self.shared_layers.append(nn.Dropout(dropout))
        
        for _ in range(num_layers - 1):
            self.shared_layers.append(nn.Linear(hidden_size, hidden_size))
            self.shared_layers.append(nn.BatchNorm1d(hidden_size))
            self.shared_layers.append(nn.ReLU())
            self.shared_layers.append(nn.Dropout(dropout))

class MLPModel(BaseMLP):
    """Modelo MLP con salida única."""
    def __init__(self, input_size, hidden_size, output_size, num_layers, dropout):
        """
        Args:
            input_size (int): El número de características de entrada de forma (batch_size, input_size).
            hidden_size (int): El número de neuronas en cada capa oculta.
            output_size (int): El número de características de salida.
            num_layers (int): El número de capas ocultas.
            dropout (float): La probabilidad de dropout para regularización.
        """
        super(MLPModel, self).__init__(input_size, hidden_size, num_layers, dropout)
        self.output_layer = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        """ Paso hacia adelante del modelo.
        Args:
            x (torch.Tensor): El tensor de entrada de forma (batch_size, input_size).
            
        Returns:
            torch.Tensor: El tensor de salida después de pasar por la red, de forma (batch_size, output_size).
        """
        x = x.to(device)
        features = self.shared_layers(x)
        return self.output_layer(features)

class DualOutputMLPModel(BaseMLP):
    """Modelo MLP con dos cabezas de salida separadas."""
    def __init__(self, input_size, hidden_size, output_size, num_layers, dropout):
        """
        Args:
            input_size (int): El número de características de entrada de forma (batch_size, input_size).
            hidden_size (int): El número de neuronas en cada capa oculta.
            output_size (int): El número de características de salida por cabeza.
            num_layers (int): El número de capas ocultas.
            dropout (float): La probabilidad de dropout para regularización.
        """
        super(DualOutputMLPModel, self).__init__(input_size, hidden_size, num_layers, dropout)
        # Crear dos cabezas de salida separadas
        self.output_head1 = nn.Linear(hidden_size, output_size)
        self.output_head2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        """ Paso hacia adelante del modelo.
        Args:
            x (torch.Tensor): El tensor de entrada de forma (batch_size, input_size).
            
        Returns:
            tuple(torch.Tensor, torch.Tensor): Dos tensores de salida, cada uno de forma (batch_size, output_size).
        """
        x = x.to(device)
        features = self.shared_layers(x)
        output1 = self.output_head1(features)
        output2 = self.output_head2(features)
        return output1, output2


class BaseRNN(nn.Module):
    """Clase RNN base con funcionalidad compartida."""
    def __init__(self, input_size, hidden_size, output_size, num_layers, dropout, rnn_type):
        """
        Args:
            input_size (int): Número de características de entrada.
            hidden_size (int): Número de características en el estado oculto.
            output_size (int): Número de características de salida.
            num_layers (int): Número de capas recurrentes.
            dropout (float): Probabilidad de dropout.
            rnn_type (str): Tipo de RNN ('RNN', 'GRU', 'LSTM').
        """
        super(BaseRNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # Seleccionar el tipo de RNN
        if rnn_type == "RNN":
            self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout)
        elif rnn_type == "GRU":
            self.rnn = nn.GRU(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout)
        elif rnn_type == "LSTM":
            self.rnn = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout)
        else:
            raise ValueError(f"Tipo de RNN no soportado: {rnn_type}")

        self.batch_norm = nn.BatchNorm1d(hidden_size)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        """
        Paso hacia adelante del modelo.
        Args:
            x (torch.Tensor): Tensor de entrada de forma (batch_size, sequence_length, input_size).
            
        Returns:
            torch.Tensor: Tensor de salida de forma (batch_size, output_size).
        """
        x = x.to(device)

        # Establecer estados ocultos y de celda iniciales
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        if isinstance(self.rnn, nn.LSTM):
            c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
            out, _ = self.rnn(x, (h0, c0))  # LSTM requiere (h0, c0)
        else:
            out, _ = self.rnn(x, h0)  # RNN y GRU solo requieren h0

        # Aplicar normalización por lotes
        out = self.batch_norm(out[:, -1, :])  # Usar el último paso de tiempo

        # Decodificar el estado oculto del último paso de tiempo
        out = self.fc(out)
        return out


class RNNModel(BaseRNN):
    """Modelo RNN."""
    def __init__(self, input_size, hidden_size, output_size, num_layers, dropout):
        super(RNNModel, self).__init__(input_size, hidden_size, output_size, num_layers, dropout, rnn_type="RNN")


class GRUModel(BaseRNN):
    """Modelo GRU."""
    def __init__(self, input_size, hidden_size, output_size, num_layers, dropout):
        super(GRUModel, self).__init__(input_size, hidden_size, output_size, num_layers, dropout, rnn_type="GRU")


class LSTMModel(BaseRNN):
    """Modelo LSTM."""
    def __init__(self, input_size, hidden_size, output_size, num_layers, dropout):
        super(LSTMModel, self).__init__(input_size, hidden_size, output_size, num_layers, dropout, rnn_type="LSTM")


class DualOutputRNN(BaseRNN):
    """Modelo RNN con dos cabezas de salida separadas."""
    def __init__(self, input_size, hidden_size, output_size, num_layers, dropout, rnn_type):
        """
        Args:
            input_size (int): Número de características de entrada.
            hidden_size (int): Número de características en el estado oculto.
            output_size (int): Número de características de salida por cabeza.
            num_layers (int): Número de capas recurrentes.
            dropout (float): Probabilidad de dropout.
            rnn_type (str): Tipo de RNN ('RNN', 'GRU', 'LSTM').
        """
        super(DualOutputRNN, self).__init__(input_size, hidden_size, output_size, num_layers, dropout, rnn_type)
        # Crear dos cabezas de salida separadas
        self.output_head1 = nn.Linear(hidden_size, output_size)
        self.output_head2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        """
        Paso hacia adelante del modelo.
        Args:
            x (torch.Tensor): Tensor de entrada de forma (batch_size, sequence_length, input_size).
            
        Returns:
            tuple(torch.Tensor, torch.Tensor): Dos tensores de salida, cada uno de forma (batch_size, output_size).
        """
        x = x.to(device)

        # Establecer estados ocultos y de celda iniciales
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        if isinstance(self.rnn, nn.LSTM):
            c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
            out, _ = self.rnn(x, (h0, c0))  # LSTM requiere (h0, c0)
        else:
            out, _ = self.rnn(x, h0)  # RNN y GRU solo requieren h0

        # Aplicar normalización por lotes
        out = self.batch_norm(out[:, -1, :])  # Usar el último paso de tiempo

        # Decodificar el estado oculto del último paso de tiempo en dos salidas
        output1 = self.output_head1(out)
        output2 = self.output_head2(out)
        return output1, output2
    

class DualOutputRNNModel(DualOutputRNN):
    """Modelo RNN con doble salida."""
    def __init__(self, input_size, hidden_size, output_size, num_layers, dropout):
        super(DualOutputRNNModel, self).__init__(input_size, hidden_size, output_size, num_layers, dropout, rnn_type="RNN")


class DualOutputGRUModel(DualOutputRNN):
    """Modelo GRU con doble salida."""
    def __init__(self, input_size, hidden_size, output_size, num_layers, dropout):
        super(DualOutputGRUModel, self).__init__(input_size, hidden_size, output_size, num_layers, dropout, rnn_type="GRU")


class DualOutputLSTMModel(DualOutputRNN):
    """Modelo LSTM con doble salida."""
    def __init__(self, input_size, hidden_size, output_size, num_layers, dropout):
        super(DualOutputLSTMModel, self).__init__(input_size, hidden_size, output_size, num_layers, dropout, rnn_type="LSTM")