# TnpB
=======

## To train the model yourself:

  -Use the requirement.txt file to set up the environment
  
  -To retrain the model please process the data  and place it in the current directory /data/processed

 #scripts
- run `data_preprocess.ipynb` to prepare the data
- run `models_trainvaltest.ipynb` or `models_trainvaltest.py`   to train and evaluate models
- run `models_inference.ipynb` or `models_inference.py` to run the trained models (i.e inference mode).
  - trained models ( `FFN`, `RNN`, `CNN`, `Transformer` based) are found under `output` directory 
- run `user_sample_inference.ipynb`  to test out on your sample data, the current example test file is: data/Endogenous_spacers_TnpB_list.csv.  Predicted output for the test samples are saved under `output` directory 

## To use the model for inference:
 -Use the requirement.txt file to set up the environment
 - place your Excel data file that includes the target sequences that you want to predict under the folder data, for instance     ./data/Endogenous_spacers_TnpB_list.csv
 - Download the trained model (the folder named output) from the link  https://www.dropbox.com/scl/fo/w2o66tafvt8upzcduwo52/ACNeSe1lCbruhiATdzeHKEI?rlkey=0l34lfkmguy88wvwjt18qxbco&st=also9pij&dl=0     and add it to the current directory  /output
 - run `user_sample_inference.ipynb`  to test out on your sample data, the current example test file is data/Endogenous_spacers_TnpB_list.csv.  The predicted output for the test samples is saved under the `output` directory 
    

=======
We also set up a web-based user-friendly interface https://www.tnpb.app/ where you can use our model model directly without installing or running any python code.
