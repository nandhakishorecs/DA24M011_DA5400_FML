# ------------------------------------------- Importing Libraries ----------------------------------------
import pandas as pd 
import numpy as np 
from sklearn.svm import SVC

from data_loader import split_csv_file, clean_dataframe, save_messages_to_files, read_emails, count_files_in_folder
from preprocessing import Preprocessing
from naive_bayes import NaiveBayesClassifier
from linear_model import LogisticRegression
from classification_metrics import Metrics

if __name__ == '__main__': 
    # ------------------------------------------- Personal Data ----------------------------------------
    print("\nDA5400 - Foundation of Machine Learning\n")
    print("\nAssignment 2\n")
    print("\nSubmitted by Nandhakishore C S - DA24M011\n")

    # ------------------------------------------- Getting Data ----------------------------------------
    dataset_path = '/Users/nandhakishorecs/Documents/IITM/Jul_2024/DA5400/Assignments/Assignment2/Solutions_DA24M011/enron_spam_data.csv'
    raw_df = pd.read_csv(dataset_path)
    raw_df = clean_dataframe(raw_df)

    # ------------------------------- Splitting Training and Testing Data -----------------------------
    split_csv_file(
        input_file = raw_df, 
        output_file1_name = 'enron_train_dataset', 
        output_file2_name = 'enron_test_dataset', 
        split_ratio = 0.999
    )
    
    # ----------------------------- Saving Emails as text files for testing ---------------------------
    test_data = pd.read_csv('enron_test_dataset.csv')
    save_messages_to_files(test_data)
    email_count = count_files_in_folder(folder_path = '/Users/nandhakishorecs/Documents/IITM/Jul_2024/DA5400/Assignments/Assignment2/Solutions_DA24M011/test')
    print(f'\nNumber of .txt files taken for testing:\t{email_count}\n')

    # --------------------------------- Cleaning training data ----------------------------------------
    train_data = pd.read_csv('enron_train_dataset.csv')
    train_data = train_data.drop(columns = ['Message ID', 'Subject', 'Date'])
    print('Number of columns in the training dataset:\n')
    print(train_data.columns)

    # ------------------------- Converting given text data into TF - IDF vectors --------------------
    train_X = np.array(train_data['Message'])
    vectoriser = Preprocessing.TFIDF_Vectorizer()
    tfidf_matrix = vectoriser.fit_transform(train_X)
    tfidf_matrix  

    # ---------------------------------------- Label Encoding ----------------------------------------
    encoder = Preprocessing.LabelEncoder() 
    y = np.array(train_data['Spam/Ham'])    
    train_y = encoder.fit_transform(y)

    # --------------------------------- MODEL 1 - Naive Bayes Classifier -----------------------------
    print('\nNaive Bayes and Logistic Regression are implemented from scratch!\n')
    model1 = NaiveBayesClassifier()
    model1.fit(tfidf_matrix, train_y)
    predictions = model1.predict(tfidf_matrix)
    
    print('\nNAIVE BAYES CLASSIFIER\n')
    metrics = Metrics() 
    accuracy = metrics.accuracy_score(train_y, predictions)
    print(f'\nAccuracy:\t{accuracy}\n')
    print('\nLabel: \'Spam\' is encoded as 1 and \'Ham\' is encoded as 0\n')
    print(metrics.classification_report(train_y, predictions))

    
    # --------------------------------- MODEL 2 - Logistic Regression ---------------------------------
    model2 = LogisticRegression(learning_rate=0.01, num_iterations=10_000)
    model2.fit(tfidf_matrix, train_y)
    predictions = model2.predict(tfidf_matrix)

    print('\nLOGISTIC REGRESSION\n')
    metrics = Metrics() 
    accuracy = metrics.accuracy_score(train_y, predictions)
    print(f'\nAccuracy:\t{accuracy}\n')
    print('\nLabel: \'Spam\' is encoded as 1 and \'Ham\' is encoded as 0\n')
    print(metrics.classification_report(train_y, predictions))

    # --------------------------------- MODEL 3 - Support Vector Machines ------------------------------
    model31 = SVC(kernel = 'linear')
    model31.fit(tfidf_matrix, train_y)
    predictions = model31.predict(tfidf_matrix)
    
    print('\nSUPPORT VECTOR MACHINES (USING LINEAR KERNEL)\n')
    metrics = Metrics() 
    accuracy = metrics.accuracy_score(train_y, predictions)
    print(f'\nAccuracy:\t{accuracy}\n')
    print('\nLabel: \'Spam\' is encoded as 1 and \'Ham\' is encoded as 0\n')
    print(metrics.classification_report(train_y, predictions))

    model32 = SVC(kernel = 'rbf')
    model32.fit(tfidf_matrix, train_y)
    predictions = model32.predict(tfidf_matrix)

    print('\nSUPPORT VECTOR MACHINES (USING RBF KERNEL)\n')
    metrics = Metrics() 
    accuracy = metrics.accuracy_score(train_y, predictions)
    print(f'\nAccuracy:\t{accuracy}\n')
    print('\nLabel: \'Spam\' is encoded as 1 and \'Ham\' is encoded as 0\n')
    print(metrics.classification_report(train_y, predictions))
    
    # --------------------------------- Testing --------------------------------------------------------
    print('\nTESTING USING EMAILS FROM TEXT FILES:\n')
    read_emails(
        directory_path='/Users/nandhakishorecs/Documents/IITM/Jul_2024/DA5400/Assignments/Assignment2/Solutions_DA24M011/test',
        vectoriser = vectoriser, 
        model = model1
    )