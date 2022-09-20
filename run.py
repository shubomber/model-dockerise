import model_service
import pickle
import mlflow

class ModelFunction:

        def __init__(self):
          self.model_service_obj = model_service.MlflowModelService()

        def store_model(self,pickle_file):
          #mlflow.set_tracking_uri()
          print('inside store_model...')
          infile = open(pickle_file,'rb+')
          model = pickle.load(infile)
          print('model pickle file loaded....')
          infile.close()
          self.model_service_obj.saveModel(model,"sklearn","SentimentAnalysis","preprocess.py")


        def load_model(self):
           self.model = self.model_service_obj.loadModel("SentimentAnalysis","1")
           return self.model


obj = ModelFunction()
obj.store_model("sentiment_analysis_xgboost.pkl")
#model = obj.load_model()


#print(model.predict(["It is a bad day"]))

