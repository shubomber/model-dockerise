FROM python:3
ENV MLFLOW_HOME /opt/mlflow
ENV MLFLOW_VERSION 1.24.0
ENV PORT 6063
RUN mkdir /mlmodel
WORKDIR /mlmodel
COPY requirements.txt /mlmodel
RUN pip install --upgrade pip
RUN pip install -r requirements.txt
COPY . /mlmodel
ENV MLFLOW_TRACKING_URI="sqlite:///mlruns.db"
EXPOSE 6063/tcp
RUN ["python3","runnew.py"]
CMD mlflow models serve -m models:/SentimentAnalysis/latest --host 0.0.0.0 --port ${PORT} --no-conda

