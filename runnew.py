import pickle
import mlflow
import wrapper

class ModelFunction:

    def __init__(self):
        print('.....Model Function Initialized.....')
        mlflow.set_tracking_uri("sqlite:///mlruns.db")

    def store_model(self,pickle_file):
        infile = open(pickle_file,'rb+')
        model = pickle.load(infile)
        infile.close()
        self.saveModel(model,"sklearn","SentimentAnalysis","preprocess.py")
    
    def saveModel(self,model,variant,readable_model_id,preprocess_file_path=None):
        readable_model_id = readable_model_id.replace("/","__$__")
        model_name = "Mlflow-Model"
        with mlflow.start_run() as active_run:
         active_run=mlflow.active_run()
        mlflow.sklearn.save_model(model,model_name)

        pyfunc_model_uri = self.logModel(readable_model_id,model_name,preprocess_file_path)
        self.registerModel(pyfunc_model_uri,readable_model_id)
    
   
    def load_model(self):
        self.model = self.loadModel("SentimentAnalysis","1")
        return self.model

    def loadModel(self,readable_model_id,version):
        #readable_model_id = readable_model_id.replace("/","__$__")
        model_uri = 'models:/' + str(readable_model_id) + "/" + version
        print('model uri of loaded model=='+ model_uri)
        self.model = mlflow.pyfunc.load_model(model_uri)
        return self.model


    def logModel(self,readable_model_id,model_name,preprocess_file_path):
        artifacts = {
        "Original_Model":model_name,
        "Original-model":preprocess_file_path}
    
        model_data = mlflow.pyfunc.log_model(
            artifact_path=str(readable_model_id),
            python_model= wrapper.Model_Wrapper(),
            artifacts=artifacts,
            code_path= ["wrapper.py"]
    )
        return model_data.model_uri


    def registerModel(self,model_uri,readable_model_id):
        model_data = mlflow.register_model(model_uri,readable_model_id)
        print('.......Model registered......')


obj = ModelFunction()
obj.store_model("sentiment_analysis_xgboost.pkl")
#model = obj.load_model()


#print(model.predict(["It is a bad day"]))

