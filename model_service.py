from asyncore import read
import logging
import pickle
import mlflow
import wrapper
import os
import sqlite3
import sqlalchemy
import sys

#Steps:
# 1.save Model
# 2.log model - set conda env , mlflow save model , mlflow log model 
# 3.register model
# 4.load model

class MlflowModelService:

   def saveModel(self,model,variant,readable_model_id,preprocess_file_path=None):
    print('inside storeModel of model service.......')
    readable_model_id = readable_model_id.replace("/","__$__")
    model_name = "Original-Model"
    with mlflow.start_run() as active_run:
        active_run = mlflow.active_run()
        mlflow.sklearn.save_model(model,model_name)

        pyfunc_model_uri = self.logModel(readable_model_id,model_name,preprocess_file_path)
        self.registerModel(pyfunc_model_uri,readable_model_id)
        


   def loadModel(self,readable_model_id,version):
       print('inside load model of model service......')
       #readable_model_id = readable_model_id.replace("/","__$__")
       print('---------------------Shubham---------------'+readable_model_id)
       model_uri = 'models:/' + str(readable_model_id) + "/" + version
       print('model uri of loadmodel=='+ model_uri)
       self.model = mlflow.pyfunc.load_model(model_uri)
       print("Model loaded successfully ......")
       return self.model

  
   def logModel(self,readable_model_id,model_name,preprocess_file_path):
      print('Inside log model.....')
      artifacts = {
         "Original_Model":model_name,
         "Original-model":preprocess_file_path}
      
      model_data = mlflow.pyfunc.log_model(
             artifact_path=str(readable_model_id),
             python_model= wrapper.Model_Wrapper(),
             artifacts=artifacts,
             code_path= ["wrapper.py"]
        )
      print('model logged .......')
      return model_data.model_uri


   def registerModel(self,model_uri,readable_model_id):
        print('inside register model......')
        print('model uri=='+ model_uri)
        model_data = mlflow.register_model(model_uri,readable_model_id)
        print('Model registered......')




