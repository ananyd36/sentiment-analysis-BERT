# sentiment-analysis-BERT

This is a personal project of performing sentiment analysis using BERT and deployment using flask.Further todo is docker containerization of deep learning model and push into production architecture.In this model bert base uncased was used for tokenization and training 

1.config.py: It contains the configuration of all parameters used in the model<br>
2.engine.py: All the main functions like training ,evaluation, loss functions are defined in this file.<br>
3.model.py: this file contains a model definition and the output layer definition according to the problem statemnet.<br>
4.train.py: This file is responsible for training of all dataset defined in dataset.py using the model defined in model.py using the train function.<br>
5.dataset.py: In this file the dataset of reviews is preprocessed according to the bert architecture and get ready for training phase.<br>
app.py: This is the flask file which is used as a service locally as a backend api to the model.<br>


For Dataset Refer...https://www.kaggle.com/abhishek/bert-base-uncased
