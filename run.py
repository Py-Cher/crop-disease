import subprocess
import os


subprocess.call([
    'python',
    'train.py',
    '--batch_size', '32',
    '--total_epoch', '10',
    '--lr', '1e-3',
    '--n_fold', '5',
    '--log_every', '100',
    '--strategy', 'epoch',
    #'--valid_every', args.module_name,
    #'--save_every', args.module_name,
    '--ckt_folder', '../checkpoint',
    '--img_folder', '/content/LG_data/train',
    '--train_csv', '/content/LG_data/train.csv',
    '--reload_step_from', '0',
    '--reload_folder_from', '0',
    '--model_name', 'test',
])
