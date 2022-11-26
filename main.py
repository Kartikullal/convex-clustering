
from convex_clustering import KMeans, GMM
from metrics import external_evaluation, retrieve_number_labels
from parse_images import parse_images 
import numpy as np
from sklearn.metrics import accuracy_score
import warnings
import pandas as pd
from sklearn.model_selection import train_test_split
warnings.filterwarnings('ignore')

if __name__ == '__main__':
    print('Pulling Training Data')
    train_path = 'Data/train-images-idx3-ubyte'
    train_label_path = 'Data/train-labels-idx1-ubyte'
    
    train_parser = parse_images(train_path, train_label_path)
    X_train = train_parser.parse_images()
    y_train = train_parser.parse_labels()
    print('Train data Acquired')
    print()
    print('Pulling Test Data')
    test_path = 'Data/t10k-images-idx3-ubyte'
    test_label_path = 'Data/t10k-labels-idx1-ubyte'

    test_parser = parse_images(test_path, test_label_path)
    X_test= test_parser.parse_images()
    y_test = test_parser.parse_labels()
    print('Test Data Acquired')
    print()
    #Fit the data to the model

    print('Training KMEANS Model')
    kmeans = KMeans(k=10, metric='euclidian')

    kmeans_labels = kmeans.fit(X_train)

    number_labels = retrieve_number_labels(kmeans_labels, y_train)

    purity_10 = external_evaluation(y_train, number_labels, metric = 'purity')
    print("Purity for Kmeans using sklearn for 10 clusters: ",purity_10)

    gini_10 = external_evaluation(y_train, number_labels, metric = 'gini')
    print("Gini value for Kmeans using sklearn for 10 clusters: ",gini_10)
    print('Model Trained')
    print()

    print('GMM on Spambase')

    print('Getting the Data ready')

    spam_path = 'Data/spambase.data'
    columns = ['word_freq_make', 'word_freq_address', 'word_freq_all', 'word_freq_3d', 'word_freq_our', 'word_freq_over', 'word_freq_remove', 'word_freq_internet', 'word_freq_order', 'word_freq_mail', 'word_freq_receive', 'word_freq_will', 'word_freq_people', 'word_freq_report', 'word_freq_addresses', 'word_freq_free', 'word_freq_business', 'word_freq_email', 'word_freq_you', 'word_freq_credit', 'word_freq_your', 'word_freq_font', 'word_freq_000', 'word_freq_money', 'word_freq_hp', 'word_freq_hpl', 'word_freq_george', 'word_freq_650', 'word_freq_lab', 'word_freq_labs', 'word_freq_telnet', 'word_freq_857', 'word_freq_data', 'word_freq_415', 'word_freq_85', 'word_freq_technology', 'word_freq_1999', 'word_freq_parts', 'word_freq_pm', 'word_freq_direct', 'word_freq_cs', 'word_freq_meeting', 'word_freq_original', 'word_freq_project', 'word_freq_re', 'word_freq_edu', 'word_freq_table', 'word_freq_conference', 'char_freq_;', 'char_freq_(', 'char_freq_[', 'char_freq_!', 'char_freq_$', 'char_freq_#', 'capital_run_length_average', 'capital_run_length_longest', 'capital_run_length_total', 'class']
    df = pd.read_table(spam_path, sep =',', header=None, names = columns)
    print('Data Ready')

    np.random.seed(42)
    X = df.drop('class', axis = 1)
    y = df['class']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

    print('Training poistive and negatice GMM models')
    #Splitting negative class and training a GMM model on it
    X_train_neg = X_train[y_train == 0]

    GMM_neg = GMM(k=6)
    GMM_neg.fit(X_train_neg)

    #Splitting postive class and training a GMM model on it
    X_train_pos = X_train[y_train == 1]

    GMM_pos = GMM(k=2)
    GMM_pos.fit(X_train_pos)

    print('Traning Finished')

    print()
    print('Predicting probability of positive and negative class using GMM model')
    #Predicting for postive and negative GMM
    x_pos = GMM_pos.predict(X_test)
    x_neg = GMM_neg.predict(X_test)
    print('Done')

    print('Calculating maximum log liklihood and accuracy')
    #Calculating prior probability for each class
    p_pos = len(X_train_pos)/(len(X_train_pos) + len(X_train_neg))
    p_neg = len(X_train_neg)/(len(X_train_pos) + len(X_train_neg))

    #Calculating liklihood for positive and negative class
    p_pos_x = x_pos * p_pos
    p_neg_x = x_neg * p_neg

    #predicting class using maximum liklihood 
    predicted = [0] * len(p_pos_x)
    for i in range(len(p_pos_x)):
        if p_pos_x[i] >= p_neg_x[i]:
            predicted[i] = 1

    accuracy = accuracy_score(y_test, predicted)

    print('Accuracy for spambase using likelihood from Gaussian Mixtures:', accuracy)
    print('Done')