import math

class DefaultConfigs():
    d_state = 16

    gpu_id = 1
  
    use_pca = False
    use_ratio = True
    use_hafm = False
    num_train = 10
    num_class = 1 
    num_band = 1 

    run_times = 1
    valid_freq = 20

    dataset = 'ip'.upper()
    test_ratio = 0.97
    epoches = 30
    patch_size = 11
    
    batch_size = 64
    learning_rate = 5e-4
    gamma = 0.9 
    weight_decay = 0 #

    patch_border_size = patch_size//2
    pca_components = 30
    embed_dim = 64
    depth = 1
    seed = 1
    patience = 200

    best_model_folder = './' + 'Wights' + '/'
    logs_folder =  './' + 'logs' + '/'
    records_folder = './' + 'records' + '/'

config = DefaultConfigs()