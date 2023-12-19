# TnpB
=======

To train the model yourself:

  -Use requirement.txt file to set uo the environment
  
  -To retrain the model please process the data  and place it in the current directory /data/processed

# Jupyter Notebooks 📔:
- run `data_preprocess.ipynb` to prepare the data
- run `models_trainvaltest.ipynb` or `models_trainvaltest.py`   to train and evaluate models
- run `models_inference.ipynb` or `models_inference.py` to run the trained models (i.e inference mode).
  - trained models ( `FFN`, `RNN`, `CNN`, `Transformer` based) are found under `output` directory 
- run `user_sample_inference.ipynb`  to test out on your sample data, the current example test file is: data/Endogenous_spacers_TnpB_list.csv.  Predicted output for the test samples are saved under `output` directory 


=======
We also set up a web-based user friendly interface https://www.tnpb.app/ where you can use our model model directly without installing or running any python code.
