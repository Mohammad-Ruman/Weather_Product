#!/bin/bash
cd /data/NOWCAST_IMD_RADAR/PYTHON_SCRIPTS/ 
source /data/conda/etc/profile.d/conda.sh

#formatted_date=$(date +'%d%b%Y_%H%M')

# Use the first argument as the formatted date
formatted_date=$1

# Check if the argument is provided
if [ -z "$formatted_date" ]; then
  echo "Please provide timestamp as an argument."
  echo "Usage: $0 <formatted_date>"
  exit 1
else
 conda activate imd_radar 
 echo "INFO :: downloading imd_radar data ..."
 python imd_radar_process_realtime_NOWCAST.py "$formatted_date" 
 echo "INFO :: imd_radar download is complete"
 echo "INFO :: pysteps Started ..... "
 python pysteps_sixfiles_updated_NOWCAST.py "$formatted_date"
 echo "INFO :: pysteps is finished"
 conda deactivate

 echo "INFO :: heat_maps generation started"
 conda activate met_work3 
 python heatmaps_NOWCAST_updated_2.py "$formatted_date"
 conda deactivate
 python3 clean_old_data.py IMD_RADAR_NOWCAST
fi
