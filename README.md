# Inference ai2es Forecast Error -- xCITE
This will be a repository for the inference mechanisms of predicting the forecast error and bias for the nwp models: HRRR.
I utilize a Long-Short-Term-Memory ML model utilizing pytorch to learn and predict the 2-meter-temperature, total-wind, and precipitation error of NWP models. This readme will serve as a way to download, clean and utilize the data for similar applications. Either to recreate and test my own work, or to utilize with your own Mesonet. All I ask is that you reference our work when you do. 

## Downloading Data
For downloading the GFS, NAM and HRRR files, please read and follow the instructions of my co-athor. Lauriana Gaudet's github repo can be found: https://github.com/lgaudet/AI2ES-notebooks

Use these notebooks:
- s3_download.ipynb
- get_resampled_nysm_data.ipynb
- cleaning_bible-NYS-subset.ipynb
- all_models_comparison_to_NYSM.ipynb

Downloading NYSM data can be found here: https://www.nysmesonet.org/weather/requestdata

## Inference Pipeline
### src
| notebook | description |
|-----------|------------|
|pipeline.py| exectues data cleaning scripts and lstm inference mode and then saves output


## Cleaning Data 
### data_cleaning 
| notebook | description |
|-----------|------------|
|all_models_comparison_to_mesos_lstm.py| cleans hrrr data that was downloaded from lgaudet's github repo and only keeps variables and locations for nysm|
|forecase_hr_parquet_builder.py| reads in hrrr data by init time and compiles into temporally linear parquet by valid_time for an input forecast hour |
|get_resampled_nysm_data.py| reads in nysm data and compiles into temporally linear parquet by valid_time for collated for temporal resolution of  NWP model|

 ## LSTM
 ### model_architecture
| notebook | description |
|-----------|------------|
|encode_decode_lstm.py|encoder decoder architecture (recommended)|
|engine_lstm_training.py|train lstm on single gpu (recommended)|
|lstm_s2s_engine.py|Exectue inference|
|sequencer.py| persistence method sequencer to feed data to go into lstm|

 ### model_data
| notebook | description |
|-----------|------------|
|prepare_lstm_data.py| compiles data from hrrr + nysm into a dataframe, indexed by valid_time. Then normalizes data|
|hrrr_data.py| compiles cleaned data from hrrr into dataframe for lstm |
|nysm_data.py|compiles cleaned data from nysm into dataframe for lstm |
|encode.py|encodes time data into cyclic transform for lstm|
|get_closest_nysm_stations.py|function to triangulate target nysm|

