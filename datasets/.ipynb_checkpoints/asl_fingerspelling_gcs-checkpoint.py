import os 

GCS_PATH = {
            'ISLR':'gs://kds-a66bb5298b56b4f027480f95fae9c7e6779d66feba454d0ff664ccc8',
            'ASLFR': 'gs://kds-1e4efec750c765afae26f4a9a1c1a2d995c7d320da64d01393f09fbf',
            'aslfr_train': 'gs://kds-e694bf64faf5d7bed51c516a5f6b2e2e1f3af2c6b28c3a317364d20d',
            'aslfr_supplemental': 'gs://kds-3276d0a7bacde791b0cde48ac4efddd397a44a4faceb39b950078c22'
            }

COMPETITION_PATH = GCS_PATH['ASLFR']

def create_asl_fingerspelling_dir(asl_dataset_path = 'datasets/asl_fingerspelling'):  
    os.makedirs(dataset_path)
    !gsutil cp {COMPETITION_PATH}/train.csv dataset_path
    !gsutil cp {COMPETITION_PATH}/supplemental_metadata.csv dataset_path
    !gsutil cp {COMPETITION_PATH}/character_to_prediction_index.json dataset_path