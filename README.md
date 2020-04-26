# csula_data_science


CS 4661 Data Science CSULA
12/01/2019

Project title: Spooky Author Identification. 

Team members: Ponaroth Eab, Julie Kasparian, Fernando Mejia, Daniel Preciado, and Margarita Nuridjanian.

Source: https://www.kaggle.com/c/spooky-author-identification/overview

Description: You're challenged to predict the author of excerpts from horror stories by Edgar Allan Poe, Mary Shelley, and HP Lovecraft.

Details: We are given checks of the writings of three horror authors and are tasked to identify each based on an excerpt of their work.

Goals: We will each use a different model in order to identify the author and will then compare the accuracy of the models and why think it might be the case.

Details about the Data that you are using:
The competition dataset contains text from works of fiction written by spooky authors of the public domain: Edgar Allan Poe, HP Lovecraft and Mary Shelley. The data was prepared by chunking larger texts into sentences using CoreNLP's MaxEnt sentence tokenizer, so you may notice the odd non-sentence here and there. Your objective is to accurately identify the author of the sentences in the test set. 

The responsibility of EACH TEAM MEMBER:
KNN - Daniel Preciado
Decision Tree - Julie Kasparian
Linear regression - Ponaroth Eab
Logistic regression - Margarita Nuridjanian
B.E.R.T -  Fernando Mejia

File Details:
train.csv - Includes the training data. We have data id of the data. The text is the quotes of each of the authors. The author column is who wrote each quote. The abbreviations for the three authors we're studying are: EAP: Edgar Allan Poe, HPL: HP Lovecraft; MWS: Mary Wollstonecraft Shelley. We will mainly be making the predictions using the test data since it has a column for the authors.
test.csv - The test data contains id and text column.
submission.csv - Is a sample submission of what the data should look like if exported for Logistic Regression. We can export the data for all the methods we worked on for the project.

Overview:
This project is developed with Python 3 and Jupyter Notebook web application.
We used the following libraries in our project:
Numpy
Pandas
Resample from sklearn.utils 
Preprocessing, metrics from sklearn
Train_test_split from sklearn.model_selection import 
Accuracy_score from sklearn.metrics 
matplotlib.pyplot 
KNeighborsClassifier from sklearn.neighbors
CountVectorizer from sklearn.feature_extraction.text 
LinearRegression from sklearn.linear_model
LogisticRegression from sklearn.linear_model
Cross_val_score from sklearn.model_selection

Methods:
B.E.R.T. - Bidirectional Encoder Representations from Transformer. Deep bidirectional representations from unlabeled text by jointly conditioning on both left and right context in all layers.
Decision Tree - Uses a tree-like model to visually and explicitly represent decisions and decision making.
Linear Regression - Uses a linear equation to predict continuous valued output. For this method to work, both features and label need to be enumerated.
Logistic Regression - Uses logistic function to model a binary dependent variable.
KNN (KNearest Neighbor) - Predicts the label for a data by checking its K-nearest neighbors. In this project, we convert words to numerical values, and then compares the words to other words with similar numerical values.

Algorithms:
Each text snippet is a vector, with each word corresponding to a dimension. The goal of the project is to get similar comments closer together in a high dimensional vector space.
First we initialized a count vectorizer(from sklearn) using the stop word ‘English’. We found initializing the vectorizer without the stop word resulted in a higher accuracy. Next we split the data set into training and testing data. After we used the vectorizer to transform the testing and training data. Data are now in sparse matrix format. We used predict_proba command to train the classifier to predict the class of the test data. We were then able to export the predict_proba data to a csv file. We used the predict command to predict the new labels for the testing set. We then were able to predict the accuracy score of the testing set using accuracy_score command where we used the y test set and the y predict data set.

Tools:
Pandas
Numpy
Sklearn Libraries:
CountVectorizer
train_test_split
Linear Regression
Logistic Regression
DecisionTreeClassifier
KNN
Cross_val_score
Fast Bert
Torch
 
Performance Comparison:
KNN accuracy score: 0.40
Decision tree accuracy score: 0.59
Logistic regression accuracy score: 0.79
B.E.R.T accuracy score: in progress
Linear regression RMSE: 3.625
Cross-validation RMSE: 1.501

Discussion About the Results:
The accuracy score for the decision tree model is 0.59. The low accuracy can be the result of a disadvantage that comes with using the decision tree model which is why the model generally give a low prediction accuracy compared to other machine learning algorithms. The decision tree model can also be problematic because the probability for the testing data seem to be excessively confident.
We notice that Logistic Regression has the highest level of accuracy compared to knn and Decision Tree. We also notice if we filter the dataset for only authors EAP and MWS, as well as take out the stop word from CountVectorize() command we get an increase in accuracy. For knn making this small change increased the accuracy from .40 to .64. This can be due to the fact that the filter reduces the amount of authors and the data size. As a result, there are fewer authors to compare the sentence fragments to. The stop word is responsible for reading the sentence up to a certain stop word. If the stop word is “English” it will read up to that point. Taking out the stop word results in all of the sentence being vectorized without being stopped. The more an author uses the stop word, the less of their text data will be read so it can reduce the accuracy of the data set overall.
Using linear regression model, we get an RMSE measurement of 3.625, which is not great for a data set that only have only three different labels. The cross-validation method scores a 1.501 on RMSE, which is more than two-time better in comparison. Linear regression needs more relevant numerical labels for it to predict better. 
Fernando tried to get the confidence of the likelihood using BERT(Pre-training of Deep Bidirectional Transformers for Language Understanding) in which it will give the probability on a text of the likelihood of it being one of the three authors. The main knobs you can turn on BERT are whether it is multi-label (in this case it was not). There are a few versions of BERT to use. I used BERT Base. Also, there is fine tuning where I would feed it more unlabeled data. In this case it would be to throw in all of the authors works.Lastly the last knob is the warm up steps. I was not able to get BERT to run to show my results because I have some problems with my GPU driver; it is needed in order to do the computation. I plan to either fix my driver or use cloud computing so I can show my score from kaggle before the presentation or to show a live demonstration where I test my model.
