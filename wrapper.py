import mlflow
import os
import importlib.util
import preprocess

class Model_Wrapper(mlflow.pyfunc.PythonModel):

    def __init__(self):
        print("Directory ----> ",os.getcwd())
        

    def load_context(self,context):
        self.model=mlflow.pyfunc.load_model(context.artifacts["Original_Model"])

    def predict(self, context, model_input):
        print ("Invoking predict with \n", model_input)

        if "Preprocessor_file" in context.artifacts.keys():
            # passing the file name and path as argument
            spec = importlib.util.spec_from_file_location("preprocess", context.artifacts["Preprocessor_file"])

            # importing the module as preprocessor
            preprocessor = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(preprocessor)

            if "user_data" in context.artifacts.keys():
                os.chdir(context.artifacts["user_data"])

            model_input = preprocessor.preprocess(model_input)

        return self.model.predict(model_input)
