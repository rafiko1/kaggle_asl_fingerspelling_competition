import os 
import subprocess

GCS_PATH = {
            'ISLR':'gs://kds-a66bb5298b56b4f027480f95fae9c7e6779d66feba454d0ff664ccc8',
            'ASLFR': 'gs://kds-7f11cf5e1079d874f10f3ee95158636d10b17443cbbf15d93d5e53a5',
            'aslfr_train': 'gs://kds-6dacf7136101fe1e25dbc81631dbcd3c9fc4a43a4a1a6e58936701ac',
            'aslfr_supplemental': 'gs://kds-3276d0a7bacde791b0cde48ac4efddd397a44a4faceb39b950078c22'
            }

COMPETITION_PATH = GCS_PATH['ASLFR']

# def create_asl_fingerspelling_dir(dataset_path = 'datasets/asl_fingerspelling'):  
#     os.makedirs(dataset_path, exist_ok=True)
#     subprocess.run([f'!gsutil cp {COMPETITION_PATH}/train.csv {dataset_path}'])
#     subprocess.run([f'!gsutil cp {COMPETITION_PATH}/supplemental_metadata.csv {dataset_path}'])
#     subprocess.run([f'!gsutil cp {COMPETITION_PATH}/character_to_prediction_index.json {dataset_path}'])