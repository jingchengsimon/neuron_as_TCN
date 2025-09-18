import glob
import os
import pickle

def find_best_model(models_dir):
    """
    Find the best model (based on validation loss)
    
    Args:
        models_dir: Model directory
        
    Returns:
        best_model_path: Path to the best model .pt file 
        best_params_path: Corresponding .pickle file path
    """
    pickle_files = glob.glob(os.path.join(models_dir, '*.pickle'))
    
    if not pickle_files:
        raise ValueError(f"No model files found in {models_dir}")
    
    best_val_loss = float('inf')
    best_model_path = None
    best_params_path = None
    
    for pickle_path in pickle_files:
        try:
            with open(pickle_path, 'rb') as f:
                data = pickle.load(f)
            
            # Get minimum validation loss
            val_losses = data['training_history_dict']['val_spikes_loss']
            min_val_loss = min(val_losses)
            
            if min_val_loss < best_val_loss:
                best_val_loss = min_val_loss
                best_params_path = pickle_path
                best_model_path = pickle_path.replace('.pickle', '.pt')
        
        except Exception as e:
            print(f"Error reading {pickle_path}: {e}")
    
    if best_model_path is None:
        raise ValueError("No valid model files found")
    
    print(f"Found best model: {best_model_path}")
    
    return best_model_path, best_params_path
