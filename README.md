# Arcade_DC
Data Challenge for Arcade Marketing Data Scientist

##### create_analyze_segments.ipynb: This notebook is designed to run the entire pipeline except feature profiling:
- It imports and calls functions to download all the data needed 
- It proceeds to create the master dataset carrying out all the necessary joins
- It calls the feature generator passing the above created dataset and teh date of segmentation as arguments 
- The custom segments are then created extracting users who qualify as non payers and whales(top 3%) spenders
- Then the features from feature grenerator are then passed to the function to run PCA and create principal components and perform k means clustering
- The labels for the clusters are received and stored
- The segments received and the custom segments are appended and analyzed
- The above pipeline is run for 3 differnet months and the labels for segments stored
- The segments across time are then analyzed to understand how users move from segments

##### main.ipynb: This is a copy of create_analyze_segments.ipynb in progress to create a full pipeline single call to effectively run the process across time

##### EDA_Data_Integrity_master_dtst_creation.ipynb: It has the EDA and data exploration carried out. It is the first notebook created

##### feature_generator_PCA.ipynb: This is created after the above notebook and is used to design the feature generator and put in place the PCA nd k means code carrying out analyses to identify optimal clusters and PCs to consider. This has the t-SNE code and plot.

##### all_functions.py : This has all the functions designed to modularize the process and create a pipeline

              
              
              
              
