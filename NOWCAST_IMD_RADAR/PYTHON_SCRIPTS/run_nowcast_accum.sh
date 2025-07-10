#!/bin/bash
cd /data/NOWCAST_IMD_RADAR/PYTHON_SCRIPTS/ 
source /data/conda/etc/profile.d/conda.sh

formatted_date=$(date +'%d%b%Y_%H%M')
conda activate imd_radar 
python accum_IMD_RADAR.py "$formatted_date" 
conda deactivate
conda activate met_work3 
python heatmaps_accum_IMD_RADAR.py "$formatted_date"
conda deactivate
