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
    trialFinalBin = np.ravel(mat_contents['trialFinalBin'].copy()) - 1 # indices de matlab a python
    dPrime = np.ravel(mat_contents['dPrime'].copy())
    criterion = np.ravel(mat_contents['criterion'].copy())
    rewCtxt = rewCtxt.squeeze()
    rewOdor = rewOdor.squeeze()
    # print("Shape del neural data:", neural_data.shape)
    # print("Shape del neural data en Rewarded Context:", neural_data[rewCtxt==1,:].shape)

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
        indices_to_remove_trialDuration.extend(range(startInd,endInd+1))

    return np.array(indices_to_remove_trialDuration)

def get_idx_by_out_of_corridor_trials(pos_binned, trialFinalBin, thPosition=1500):
    """
    Índices de los time bins de trials cuya posición pico supera `thPosition`
    (en grados de encoder). Detecta artefactos de segmentación / inicio de
    sesión: el trial no termina y la posición rampa muy por fuera del corredor
    físico (p. ej. el primer trial de los datasets DG-naive, que llega a
    ~26500° cuando el corredor válido no pasa de ~1200°).

    Se elimina el trial COMPLETO. Es distinto —y complementario— al filtro de
    inmovilidad: durante estos trials el animal corre, así que sus bins en
    movimiento (que cargan las posiciones absurdas) sobreviven al filtro de
    inmovilidad y arruinan la estandarización de la posición.
    """
    position = np.ravel(pos_binned)
    trialFinalBin = np.ravel(trialFinalBin).astype(int)

    trialsOutOfCorridor = []
    indices_to_remove_out_of_corridor = []
    for trial in range(len(trialFinalBin)):
        startInd = 0 if trial == 0 else trialFinalBin[trial-1] + 1
        endInd = trialFinalBin[trial]
        trial_pos = position[startInd:endInd+1]
        if trial_pos.size == 0 or np.all(np.isnan(trial_pos)):
            continue
        if np.nanmax(trial_pos) > thPosition:
            trialsOutOfCorridor.append(trial)
            indices_to_remove_out_of_corridor.extend(range(startInd, endInd+1))

    if trialsOutOfCorridor:
        print('El trial ', trialsOutOfCorridor, 'sale del corredor (posición pico >', thPosition, 'grados)')

    return np.array(indices_to_remove_out_of_corridor, dtype=int)

def periods_immobility(vels_binned, thVel, binLength, thDur, trialFinalBin, bins_before, bins_after):
    
    candidates = np.array(np.ravel(vels_binned) <thVel, dtype=int)
    candidates[-1] = 0 
    jumps = np.append(candidates[0], np.diff(candidates))
    eventsOn = np.where(jumps==1)[0]
    eventsOff = np.where(jumps==-1)[0]
    durations=(eventsOff-eventsOn)*binLength/1000
    included = np.where(durations>thDur)[0]
    immobility=[]
    for i in included:
        immobility.extend(range(eventsOn[i]-bins_after, eventsOff[i]+bins_before+1))

    trial_transitions = []
    for trial_idx in range(len(trialFinalBin)):
        start_idx = trialFinalBin[trial_idx] + 1 - bins_before 
        if trial_idx == len(trialFinalBin)-1:
            end_idx = trialFinalBin[-1] 
        else:
            end_idx = trialFinalBin[trial_idx] + 1 + int(400/binLength) + bins_after # le sumo 400ms adicional para cubrir el principio del siguiente trial, que también suele ser inestable
        trial_transitions.extend(range(start_idx, end_idx+1))

    return immobility, trial_transitions

def get_idx_by_low_performance(dPrime, trialFinalBin, threshold):
    """
    Obtenemos los índices de los trials con bajo rendimiento
    """
    low_performance_trials_indices = np.where((dPrime <= threshold) | (np.isnan(dPrime)))[0]

    # Mostrar los índices de los trials
    print("Trials con dPrime que no cumplen el criterio:", low_performance_trials_indices)
    print("Cantidad de trials que no cumplen criterio dPrime:", len(low_performance_trials_indices))

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

def clean_neurons_by_low_firing_rate(neural_data, firingMinimo, binLength):
    """
    Eliminamos las neuronas con pocos spikes
    """
    # nd_sum = np.nansum(X[:,0,:], axis=0)
    # rmv_nrn_clean = np.where(nd_sum < firingMinimo)
    # X = np.delete(X, rmv_nrn_clean, 2)

    #nd_avg = np.nansum(neural_data, axis=0)/neural_data.shape[0] # promedio de spikes por neurona
    nd_avg = np.nanmean(neural_data, axis=0)/(binLength/1000)
    lowfiring_neurons = np.where(nd_avg < firingMinimo)
    
    #neural_data = np.delete(neural_data, lowfiring_neurons, 1)
    #print(lowfiring_neurons[0].shape[0], "neuronas con firing rate menor a", firingMinimo, "spikes/bin")
    
    #return neural_data, lowfiring_neurons
    return lowfiring_neurons

def clean_unstable_neurons(neural_data,numBlocks=2, threshDrift=0.5):

    # truncate so it's divisible by numBlocks
    upTo = neural_data.shape[0] - (neural_data.shape[0] % numBlocks)
    blockSize = upTo // numBlocks

    # reshape container
    reshape_neural_data = np.full((numBlocks, blockSize, neural_data.shape[1]), np.nan)

    # fill blocks
    for i in range(numBlocks):
        reshape_neural_data[i, :, :] = neural_data[i * blockSize:(i + 1) * blockSize, :]

    # mean across time within each block
    meanBlockActivity = np.mean(reshape_neural_data, axis=1)  # shape: (numBlocks, neurons)

    # drift index
    driftIndex = np.abs(meanBlockActivity[0, :] - meanBlockActivity[1, :]) / \
             (meanBlockActivity[0, :] + meanBlockActivity[1, :])

    # sort
    inds = np.argsort(driftIndex)  # ascending order
    vals = driftIndex[inds]

    return inds[vals > threshDrift]

def save_data(X, y, trial_markers=None, file_path=None):
    """
    Save the data in a pickle file with optional trial markers
    """
    if trial_markers is not None:
        with open(file_path, 'wb') as f:
            pickle.dump((X, y, trial_markers), f)
        #print("Datos con marcadores de trial guardados en", file_path)
    else:
        with open(file_path, 'wb') as f:
            pickle.dump((X, y), f)
        #print("Datos guardados correctamente en", file_path)

def preprocess_data(
    file_path,
    file_name_to_save,
    bins_before,
    bins_after,
    bins_current,
    threshDPrime,
    firingMinimo,
    binLength=200,
    thVel=1,
    thDur=4,
    thPosition=1500,
    data_dir="data",
):
    
    # Cargamos los datos
    mat_contents, neural_data, rewCtxt, trialFinalBin, dPrime, criterion, rewOdor, pos_binned, vels_binned = load_data(file_path)
    
    original_num_time_bins = neural_data.shape[0]

    
    # Datos a decodificar
    y = np.concatenate((pos_binned, vels_binned), axis=1)
           
    # Crear marcadores de trial antes de cualquier eliminación
    trial_markers = create_trial_markers(trialFinalBin, neural_data.shape[0])
    
    # # Obtenemos los índices de los trials con duración muy larga
    # trialDurationInBins = np.ravel(mat_contents['trialDurationInBins'].copy())
    # indices_to_remove_trialDuration = get_idx_by_high_trial_duration(trialDurationInBins, trialFinalBin)
    # print("Índices de los time bins a eliminar por larga duración:", indices_to_remove_trialDuration)
    
    # CLEANING DE BOUNDARIES SIN HISTORY
    first_indexes = np.arange(bins_before)
    last_indexes = np.arange(original_num_time_bins-bins_after,original_num_time_bins)
    
    # indices_to_remove_temp = np.concatenate((first_indexes, indices_to_remove_trialDuration, last_indexes))
    # print("Indices a remover por ahora, sin historia y por inactividad:", indices_to_remove_temp)
    
    # candidates = np.array(np.ravel(vels_binned) <thVel, dtype=int)
    # candidates[-1] = 0 
    # jumps = np.append(candidates[0], np.diff(candidates))
    # eventsOn = np.where(jumps==1)[0]
    # eventsOff = np.where(jumps==-1)[0]
    # durations=(eventsOff-eventsOn)*binLength/1000
    # included = np.where(durations>thDur)[0]
    # immobility=[]
    # for i in included:
    #     immobility.extend(range(eventsOn[i]-bins_after, eventsOff[i]+bins_before+1))

    # trial_transitions = []
    # for trial_idx in range(len(trialFinalBin)):
    #     start_idx = trialFinalBin[trial_idx] + 1 - bins_before 
    #     if trial_idx == len(trialFinalBin)-1:
    #         end_idx = trialFinalBin[-1] 
    #     else:
    #         end_idx = trialFinalBin[trial_idx] +1 + bins_after
    #     trial_transitions.extend(range(start_idx, end_idx+1))

    immobility,trial_transitions = periods_immobility(vels_binned, thVel, binLength, thDur, trialFinalBin, bins_before, bins_after)

    indices_to_remove_immob = np.concatenate((first_indexes, immobility, trial_transitions, last_indexes))
    print(f"Cantidad de time bins a remover por ahora por inmovilidad y transiciones de trials: {len(indices_to_remove_immob)}")

    # Obtenemos los índices de los trials con bajo rendimiento
    indices_to_remove_low_performance = get_idx_by_low_performance(dPrime, trialFinalBin, threshDPrime)
    indices_to_remove_low_performance = np.array(indices_to_remove_low_performance, dtype=int)
    print(f"Cantidad de time bins a eliminar por no cumplir criterio dPrime: {len(indices_to_remove_low_performance)}")

    # Obtenemos los índices de los trials que salen del corredor físico
    # (artefacto de segmentación / inicio de sesión: posición pico > thPosition)
    indices_to_remove_out_of_corridor = get_idx_by_out_of_corridor_trials(pos_binned, trialFinalBin, thPosition)
    indices_to_remove_out_of_corridor = np.array(indices_to_remove_out_of_corridor, dtype=int)
    print(f"Cantidad de time bins a eliminar por trials fuera del corredor: {len(indices_to_remove_out_of_corridor)}")

    # *** CORRECCIÓN DEL BUG: Manejar correctamente los tipos de datos ***
    rmv_time = np.where(np.isnan(y[:,0]))[0]  # Extraer solo la primera dimensión de la tupla

    # 🔧 CORRECCIÓN CRÍTICA: Convertir todos los índices a enteros
    rmv_time = np.array(rmv_time, dtype=int)
    indices_to_remove_immob = np.array(indices_to_remove_immob, dtype=int)
    indices_to_remove_low_performance = np.array(indices_to_remove_low_performance, dtype=int)

    # Combinar todos los índices y asegurar que el resultado sea entero
    indices_to_remove = np.union1d(
        rmv_time,
        np.union1d(
            indices_to_remove_immob,
            np.union1d(indices_to_remove_low_performance, indices_to_remove_out_of_corridor),
        ),
    ).astype(int)
   
    print(f"   Total de bins a eliminar: {len(indices_to_remove)}")
    #print(f"   Tipo de datos de indices_to_remove: {indices_to_remove.dtype}")
    
    # # Agregar los índices de bajo rendimiento a los índices a eliminar
    # rmv_time=np.where(np.isnan(y[:,0])) # indices en los que la posicion es NaN
    # indices_to_remove = np.union1d(rmv_time,np.union1d(indices_to_remove_temp, indices_to_remove_low_performance))

    #print("Índices totales de los time bins a eliminar:", indices_to_remove)
    
    
    # Agregamos el contexto a los datos
    neural_data_with_ctxt = add_context_to_data(neural_data, rewCtxt)

    # Obtengo los spikes con historia
    X = get_spikes_with_history(neural_data_with_ctxt,bins_before,bins_after,bins_current)
    #print("Shape del X:", X.shape)
    
    
    # Eliminamos los datos con bajo rendimiento y duración de trial
    X = np.delete(X, indices_to_remove, 0)
    y = np.delete(y, indices_to_remove, 0)
    
    # Actualizar los marcadores de trial eliminando los mismos índices
    trial_markers = np.delete(trial_markers, indices_to_remove, 0)
    
    neural_data = np.delete(neural_data, indices_to_remove, 0) # Esto lo necesito para el filtrado de neuronas inestables y de bajo firing rate, que se hace después del filtrado de time bins

    # Removemos neuronas inestables
    unstableNeurons = clean_unstable_neurons(neural_data, numBlocks=2, threshDrift=0.5)
    
    # Eliminamos las neuronas con pocos spikes
    lowfiring_neurons = clean_neurons_by_low_firing_rate(neural_data, firingMinimo, binLength)

        
    X=np.delete(X,np.union1d(lowfiring_neurons, unstableNeurons).astype(int),2)


    print("Shape de X final (con contexto):", X.shape)
    
    # Solo continuar si tenemos datos
    if X.shape[0] > 0 and X.shape[2] > 0:
        # Flatten X: Esto lo necesito para entrenar los no recurrentes
        X_flat = X.reshape(X.shape[0], (X.shape[1] * X.shape[2]))
        
        # Guardamos los datos con marcadores de trial
        save_data(X, y[:, 0].reshape(-1, 1), trial_markers, f'{data_dir}/{file_name_to_save}_withCtxt_onlyPosition.pickle')
        save_data(X_flat, y[:, 0].reshape(-1, 1), trial_markers, f'{data_dir}/{file_name_to_save}_withCtxt_onlyPosition_flat.pickle')
        save_data(X, y[:, 1].reshape(-1, 1), trial_markers, f'{data_dir}/{file_name_to_save}_withCtxt_onlyVelocity.pickle')
        save_data(X_flat, y[:, 1].reshape(-1, 1), trial_markers, f'{data_dir}/{file_name_to_save}_withCtxt_onlyVelocity_flat.pickle')
        save_data(X, y, trial_markers, f'{data_dir}/{file_name_to_save}_withCtxt.pickle')
        save_data(X_flat, y, trial_markers, f'{data_dir}/{file_name_to_save}_withCtxt_flat.pickle')
        
        # Removemos las últimas dos columnas del tensor X para quedarnos solo con las neuronas como características
        X_no_context = X[:, :, :-2]
       # print(f"   Shape del X sin contexto: {X_no_context.shape}")
        
        # Flatten X sin contexto
        X_no_context_flat = X_no_context.reshape(X_no_context.shape[0], (X_no_context.shape[1] * X_no_context.shape[2]))
        
        # Guardamos los datos sin contexto
        save_data(X_no_context, y[:, 0].reshape(-1, 1), trial_markers, f'{data_dir}/{file_name_to_save}_onlyPosition.pickle')
        save_data(X_no_context_flat, y[:, 0].reshape(-1, 1), trial_markers, f'{data_dir}/{file_name_to_save}_onlyPosition_flat.pickle')
        save_data(X_no_context, y[:, 1].reshape(-1, 1), trial_markers, f'{data_dir}/{file_name_to_save}_onlyVelocity.pickle')
        save_data(X_no_context_flat, y[:, 1].reshape(-1, 1), trial_markers, f'{data_dir}/{file_name_to_save}_onlyVelocity_flat.pickle')
        save_data(X_no_context, y, trial_markers, f'{data_dir}/{file_name_to_save}.pickle')
        save_data(X_no_context_flat, y, trial_markers, f'{data_dir}/{file_name_to_save}_flat.pickle')
        
        
        resultsPreprocess = 100*np.array([len(indices_to_remove_immob), len(indices_to_remove_low_performance), len(rmv_time), len(indices_to_remove)])/original_num_time_bins
        resultsPreprocess = np.round(resultsPreprocess, 2)
        resultsPreprocess = list(resultsPreprocess)
        #resultsPreprocess.extend([X.shape[0], lowfiring_neurons[0].shape[0], unstableNeurons.shape[0], X.shape[2]])
        resultsPreprocess.extend([X.shape[0], lowfiring_neurons[0].shape[0], np.setdiff1d(unstableNeurons,lowfiring_neurons[0]).shape[0], X.shape[2]-2])
        
        dictResultsPreprocess = {
            'success': True,
            'proportion_immobility': resultsPreprocess[0],
            'proportion_low_performance': resultsPreprocess[1],
            'proportion_nan_position': resultsPreprocess[2],
            'proportion_total_removed': resultsPreprocess[3],
            'time_bins_kept': resultsPreprocess[4],
            'low_firing_neurons': resultsPreprocess[5],
            'unstable_neurons': resultsPreprocess[6],
            'neurons_Kept': resultsPreprocess[7]
        }

        print(f"   ✅ Preprocesamiento completado exitosamente!")
 

        return dictResultsPreprocess
    else:
        resultsPreprocess =[np.nan, np.nan, np.nan, np.nan, np.nan , np.nan , np.nan , np.nan]
        dictResultsPreprocess = {
            'success': False,
            'proportion_immobility': resultsPreprocess[0],
            'proportion_low_performance': resultsPreprocess[1],
            'proportion_nan_position': resultsPreprocess[2],
            'proportion_total_removed': resultsPreprocess[3],
            'time_bins_kept': resultsPreprocess[4],
            'low_firing_neurons': resultsPreprocess[5],
            'unstable_neurons': resultsPreprocess[6],
            'neurons_Kept': resultsPreprocess[7]
        }

        print(f"   ⚠️  ADVERTENCIA: No quedan datos después del filtrado!")
        return dictResultsPreprocess
    
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


