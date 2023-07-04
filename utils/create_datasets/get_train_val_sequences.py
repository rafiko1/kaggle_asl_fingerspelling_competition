import pandas as pd
import numpy as np
from sklearn.model_selection import GroupKFold

def get_train_val_sequences(df_path, n_splits, fold):
    df = pd.read_csv(df_path)
    
    df['participant_cumcount'] = df.groupby('participant_id').cumcount()
    df['fold'] = -1
    
    group_by_participant_id_array = np.array(df['participant_id'].values)
    gkf = GroupKFold(n_splits = n_splits).split(df, groups = group_by_participant_id_array)
    
    result = []
    for fold_ind, (train_idx, val_idx) in enumerate(gkf):
        df.loc[val_idx, "fold"] = fold_ind
    
    train_sequences = df[df['fold']!= fold].sequence_id
    val_sequences = df[df['fold']==fold].sequence_id
    
    return train_sequences, val_sequences