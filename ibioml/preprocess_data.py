#%% IMPORTO LIBRERIAS
import numpy as np
import matplotlib.pyplot as plt
from scipy import io
from scipy import stats
import pickle
import time
import sys
from ibioml.utils.preprocessing_funcs import get_spikes_with_history, create_trial_markers

#%%
def load_data(file_path):
    """
    Cargamos los datos de un archivo .mat
    """
    mat_contents = io.loadmat(file_path)
    neural_data = mat_contents['neuronActivity'].copy()
    rewCtxt = mat_contents['rewCtxt'].copy()
    rewOdor = mat_contents['rewOdor'].copy()
    trialFinalBin = np.ravel(mat_contents['trialFinalBin'].copy())
    dPrime = np.ravel(mat_contents['dPrime'].copy())
    criterion = np.ravel(mat_contents['criterion'].copy())
    rewCtxt = rewCtxt.squeeze()
    rewOdor = rewOdor.squeeze()
    print("Shape del neural data:", neural_data.shape)
    print("Shape del neural data en Rewarded Context:", neural_data[rewCtxt==1,:].shape)

    # variables a decodificar
    pos_binned = mat_contents['position'].copy()
    vels_binned = mat_contents['velocity'].copy()

    return mat_contents, neural_data, rewCtxt, trialFinalBin, dPrime, criterion, rewOdor, pos_binned, vels_binned

def add_context_to_data(neural_data, rewCtxt):
    """
    Agregamos el contexto a los datos
    """
    rewCtxt_neg = np.logical_not(rewCtxt).astype("uint8")
    neural_data_with_ctxt = np.concatenate((neural_data, rewCtxt[:,np.newaxis], rewCtxt_neg[:,np.newaxis]), axis=1)
    return neural_data_with_ctxt

def process_history(neural_data, bins_before, bins_after, bins_current):
    """
    Procesamos la historia de los spikes
    """
    X = get_spikes_with_history(neural_data, bins_before, bins_after, bins_current)
    print("Shape del X:", X.shape)
    return X

def get_idx_by_high_trial_duration(trialDurationInBins, trialFinalBin):
    """
    Obtenemos los índices de los trials con duración muy larga
    """
    threshTrialDuration=np.mean(trialDurationInBins)+3*np.std(trialDurationInBins)
    trialsTooLong= np.ravel(np.where(trialDurationInBins>=threshTrialDuration))
    print('El trial ', trialsTooLong , 'es muy largo')

    indices_to_remove_trialDuration=[]
    for trial in trialsTooLong:
        if trial==0:
            startInd=0
        else:
            startInd=trialFinalBin[trial-1]+1
        endInd=trialFinalBin[trial]
        indices_to_remove_trialDuration.extend(range(startInd,endInd))

    return np.array(indices_to_remove_trialDuration)

def get_idx_by_low_performance(dPrime, trialFinalBin, threshold):
    """
    Obtenemos los índices de los trials con bajo rendimiento
    """
    low_performance_trials_indices = np.where((dPrime <= threshold) | (np.isnan(dPrime)))[0]

    # Mostrar los índices de los trials
    print("Trials con dPrime menor o igual a 2.8:", low_performance_trials_indices)
    print("Cantidad de trials:", len(low_performance_trials_indices))

    # Crear una lista para almacenar los índices de los time bins a eliminar
    indices_to_remove_low_performance = []

    for trial in low_performance_trials_indices:
        if trial==0:
            startInd=0
        else:
            startInd=trialFinalBin[trial-1]+1
        endInd=trialFinalBin[trial] 
        indices_to_remove_low_performance.extend(range(startInd,endInd+1))  

    return np.array(indices_to_remove_low_performance)

def clean_neurons_by_low_firing_rate(X, firingMinimo):
    """
    Eliminamos las neuronas con pocos spikes
    """
    nd_sum = np.nansum(X[:,0,:], axis=0)
    rmv_nrn_clean = np.where(nd_sum < firingMinimo)
    X = np.delete(X, rmv_nrn_clean, 2)
    return X

def save_data(X, y, trial_markers=None, file_path=None):
    """
    Save the data in a pickle file with optional trial markers
    """
    if trial_markers is not None:
        with open(file_path, 'wb') as f:
            pickle.dump((X, y, trial_markers), f)
        print("Datos con marcadores de trial guardados en", file_path)
    else:
        with open(file_path, 'wb') as f:
            pickle.dump((X, y), f)
        print("Datos guardados correctamente en", file_path)

def preprocess_data(file_path, file_name_to_save, bins_before, bins_after, bins_current, threshDPrime, firingMinimo):
    # Cargamos los datos
    mat_contents, neural_data, rewCtxt, trialFinalBin, dPrime, criterion, rewOdor, pos_binned, vels_binned = load_data(file_path)
    trialFinalBin[-1] = neural_data.shape[0]-1 # el último trial no tiene un trialFinalBin, lo agregamos manualmente
    
    # Agregamos el contexto a los datos
    neural_data_with_ctxt = add_context_to_data(neural_data, rewCtxt)
    
    # Datos a decodificar
    y = np.concatenate((pos_binned, vels_binned), axis=1)
    
    # Obtengo los spikes con historia
    X = get_spikes_with_history(neural_data_with_ctxt,bins_before,bins_after,bins_current)
    # esto lo realizo ahora para luego poder eliminar los spikes sin historia que quedan al principio y al final del tensor X, es decir, los nans que se generan en el proceso de obtener los spikes con historia en donde no hay historia
    print("Shape del X:", X.shape)
    
    # Crear marcadores de trial antes de cualquier eliminación
    trial_markers = create_trial_markers(trialFinalBin, neural_data.shape[0])
    
    # Obtenemos los índices de los trials con duración muy larga
    trialDurationInBins = np.ravel(mat_contents['trialDurationInBins'].copy())
    indices_to_remove_trialDuration = get_idx_by_high_trial_duration(trialDurationInBins, trialFinalBin)
    print("Índices de los time bins a eliminar por larga duración:", indices_to_remove_trialDuration)
    
    # CLEANING DE BOUNDARIES SIN HISTORY
    # removemos los primeros bins y los ultimos porque no tienen historia (son los creados por get_spikes_with_history), boundaries
    first_indexes = np.arange(bins_before)
    last_indexes = np.arange(X.shape[0]-bins_after,X.shape[0])
    
    indices_to_remove_temp = np.concatenate((first_indexes, indices_to_remove_trialDuration, last_indexes))
    print("Indices a remover por ahora, sin historia y por inactividad:", indices_to_remove_temp)
    
    # Obtenemos los índices de los trials con bajo rendimiento
    indices_to_remove_low_performance = get_idx_by_low_performance(dPrime, trialFinalBin, threshDPrime)
    print("Índices de los time bins a eliminar por bajo rendimiento:", indices_to_remove_low_performance)
    
    # Agregar los índices de bajo rendimiento a los índices a eliminar
    rmv_time=np.where(np.isnan(y[:,0])) # indices en los que la posicion es NaN
    indices_to_remove = np.union1d(rmv_time,np.union1d(indices_to_remove_temp, indices_to_remove_low_performance))

    print("Índices totales de los time bins a eliminar:", indices_to_remove)
    
    # Eliminamos los datos con bajo rendimiento y duración de trial
    X = np.delete(X, indices_to_remove, 0)
    y = np.delete(y, indices_to_remove, 0)
    
    # Actualizar los marcadores de trial eliminando los mismos índices
    trial_markers = np.delete(trial_markers, indices_to_remove, 0)
    
    # Eliminamos las neuronas con pocos spikes
    X = clean_neurons_by_low_firing_rate(X, firingMinimo)
    print("Shape de X final:", X.shape)
    
    # Flatten X: Esto lo necesito para entrenar los no recurrentes
    X_flat = X.reshape(X.shape[0], (X.shape[1] * X.shape[2]))
    
    # Guardamos los datos con marcadores de trial
    save_data(X, y[:, 0].reshape(-1, 1), trial_markers, 'data/'+file_name_to_save+'_withCtxt_onlyPosition.pickle')
    save_data(X_flat, y[:, 0].reshape(-1, 1), trial_markers, 'data/'+file_name_to_save+'_withCtxt_onlyPosition_flat.pickle')
    save_data(X, y[:, 1].reshape(-1, 1), trial_markers, 'data/'+file_name_to_save+'_withCtxt_onlyVelocity.pickle')
    save_data(X_flat, y[:, 1].reshape(-1, 1), trial_markers, 'data/'+file_name_to_save+'_withCtxt_onlyVelocity_flat.pickle')
    save_data(X, y, trial_markers, 'data/'+file_name_to_save+'_withCtxt.pickle')
    save_data(X_flat, y, trial_markers, 'data/'+file_name_to_save+'_withCtxt_flat.pickle')
    
    # Removemos las últimas dos columnas del tensor X para quedarnos solo con las neuronas como características
    X_no_context = X[:, :, :-2]
    print("Shape del X sin contexto:", X_no_context.shape)
    
    # Flatten X sin contexto
    X_no_context_flat = X_no_context.reshape(X_no_context.shape[0], (X_no_context.shape[1] * X_no_context.shape[2]))
    
    # Guardamos los datos sin contexto
    save_data(X_no_context, y[:, 0].reshape(-1, 1), trial_markers, 'data/'+file_name_to_save + '_onlyPosition.pickle')
    save_data(X_no_context_flat, y[:, 0].reshape(-1, 1), trial_markers, 'data/'+file_name_to_save + '_onlyPosition_flat.pickle')
    save_data(X_no_context, y[:, 1].reshape(-1, 1), trial_markers, 'data/'+file_name_to_save+'_onlyVelocity.pickle')
    save_data(X_no_context_flat, y[:, 1].reshape(-1, 1), trial_markers, 'data/'+file_name_to_save+'_onlyVelocity_flat.pickle')
    save_data(X_no_context, y, trial_markers, 'data/'+file_name_to_save + '.pickle')
    save_data(X_no_context_flat, y, trial_markers, 'data/'+file_name_to_save + '_flat.pickle')

    
#%%
# GRAFICAR DURACION DE LOS TRIALS
def plot_trial_duration(trialDurationInBins):
    """
    Plot the trial duration
    """
    plt.figure(figsize=(10, 6))
    plt.plot(trialDurationInBins, label='Duración de los trials')
    plt.axhline(np.mean(trialDurationInBins) + 3 * np.std(trialDurationInBins), color='r', linestyle='--', label='Umbral de duración')
    plt.xlabel('Índice del trial')
    plt.ylabel('Duración del trial (bins)')
    plt.show()

#%%
# GRAFICAR TRIALS CON LOW PERFORMANCE 
def plot_low_performance(dPrime):
    """
    Plot the low performance
    """
    plt.figure(figsize=(10, 6))
    plt.plot(dPrime, label='dPrime')
    plt.axhline(2.5, color='r', linestyle='--', label='Umbral de rendimiento')
    plt.xlabel('Índice del trial')
    plt.ylabel('dPrime')
    # x grid que marque 5 lugares equidistantes
    plt.xticks(np.arange(0, len(dPrime), len(dPrime)//5))
    plt.grid(True, axis='x')
    plt.legend()
    plt.show()
    
# %%
# Example usage
# preprocess_data(
#     file_path='datasets/DG_S19_bins200ms_completo.mat', 
#     file_name_to_save='S19/5_5_1/thresh2_5/bins200ms_preprocessed', 
#     bins_before=5, 
#     bins_after=5, 
#     bins_current=1, 
#     threshDPrime=2.5, 
#     firingMinimo=1000
# )


