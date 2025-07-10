"""
Author: Sai Mohan
Date: 2024-12-12
Description: script deletes the old folders for the given source folders
only keeping the latest ones based on the backup folder count

"""

import os
import shutil
import sys

def list_and_remove_old_folders(folder_path, folder_count) :
    listed_files =  os.listdir(folder_path)
    sorted_files = sorted(
            listed_files,
            key = lambda file_name: os.path.getmtime(os.path.join(folder_path, file_name)),
            )
    old_files = sorted_files[:-folder_count] if len(sorted_files) > folder_count else []  
    for file_name in old_files:
        full_path = os.path.join(folder_path, file_name)
        try:
            if os.path.isfile(full_path):
                os.remove(full_path)
            elif os.path.isdir(full_path):
                shutil.rmtree(full_path)
            print(f"deleted {file_name}")
        except Exception as e:
            print(f"Error deleting {full_path}: {e}")

# source constants
IMD_RADAR_NOWCAST = 'IMD_RADAR_NOWCAST'
RADAR_CTT_EUMETSAT_NOWCAST = 'RADAR_CTT_EUMETSAT_NOWCAST'

HOURS_IN_A_DAY = 24
BACKUP_COUNT_PER_HOUR = 6

SOURCE_FOLDER_BACKUP_DICT = {
        RADAR_CTT_EUMETSAT_NOWCAST : {
            '/data/NOWCAST_ML_MODEL/IMD_RADAR/RADAR_DATA_FOLDERS/': 2 * HOURS_IN_A_DAY * BACKUP_COUNT_PER_HOUR,
       	    '/data/NOWCAST_ML_MODEL/CTT_METEOLOGIX/CTT_INDIA_METEOLOGIX/': BACKUP_COUNT_PER_HOUR,
	    '/data/NOWCAST_ML_MODEL/MLP_INPUTS/': 2 * BACKUP_COUNT_PER_HOUR,
            '/data/NOWCAST_ML_MODEL/PYSTEPS_DATA/PYSTEPS_INPUTS/': BACKUP_COUNT_PER_HOUR,
	    '/data/NOWCAST_ML_MODEL/PYSTEPS_DATA/PYSTEPS_OUTPUT_DATA/': BACKUP_COUNT_PER_HOUR,
            '/data/NOWCAST_ML_MODEL/EUMETSAT/EUMETSAT_PPTRATE_DATA/HIGH_RESOLUTION_DATA/': BACKUP_COUNT_PER_HOUR,
            '/data/NOWCAST_ML_MODEL/EUMETSAT/EV_PAST_TIFFS/': BACKUP_COUNT_PER_HOUR
        }
        ,
        IMD_RADAR_NOWCAST : {
            '/data/NOWCAST_IMD_RADAR/RADAR_DATA_FOLDERS/': HOURS_IN_A_DAY * BACKUP_COUNT_PER_HOUR,
       	    '/data/NOWCAST_IMD_RADAR/PYSTEPS_DATA/PYSTEPS_INPUTS/': BACKUP_COUNT_PER_HOUR,
            '/data/NOWCAST_IMD_RADAR/PYSTEPS_DATA/PYSTEPS_OUTPUT_DATA/': BACKUP_COUNT_PER_HOUR,
            '/data/NOWCAST_IMD_RADAR/accum_radar_data/': BACKUP_COUNT_PER_HOUR
        }
}

if __name__ == "__main__":

    valid_sources = ['IMD_RADAR_NOWCAST', 'RADAR_CTT_EUMETSAT_NOWCAST']
    arguments = sys.argv 
    if len(arguments) < 2:
        print('Usage :: python clean_old_data.py <source>')
        sys.exit(1)
    source = arguments[1]
    if source not in valid_sources:
        print('Error :: invalid source argument found :: '+source)
        sys.exit()
    for folder_path, backup_folder_count in SOURCE_FOLDER_BACKUP_DICT[source].items():
        list_and_remove_old_folders(folder_path, backup_folder_count)
