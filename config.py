ANNOT_PATH = '/home/cps_lab/seungeun/MRI/ptdata/BASELINE_LABEL_MERGE.csv'
GENE_PATH = '/home/cps_lab/seungeun/MRI/GWAS_pMCI_sMCI_128.csv'
PREPROCESSED_DATA_PATH = '/data1/seungeun/MRI/Preprocessing/Registration_BiasCorr/' 


STAITISTICS_FILE = "statistics_of_channels.json"

LABEL_LIST = [
    'sMCI', 'pMCI'
]

ALL_LABEL_LIST = [
    'CN', 'AD', 'sMCI', 'pMCI'
]

NUM_OF_GENES = 128

PATCH_SIZE = 16

PATCH_NUM_DEPTH = 4
PATCH_NUM_HEIGHT = 4
PATCH_NUM_WIDTH = 4

HEIGHT = 192
WIDTH = 192

VOLUME_SLICE_SIZE = 14
FEATURE_DIM = 128
VOLUME = 144