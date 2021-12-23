
#----------------------------------------------------------------------------------------------------------------------------------------------------------------------#

#Steps in Machine Learning Modeling:

#data collection
#data description
#data exploration (data visualization of i/p data)
#data cleaning
#data preparation
#training and evaluating the model
#interpreting the model (data visualization)
#making predictions with the model
#analysis : accuracy, confusion matrix, classification report

#----------------------------------------------------------------------------------------------------------------------------------------------------------------------#

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sys import exit

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVR
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn import model_selection
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import accuracy_score



#data collection
def csv_read_data():
    try:
        with open("pima-indians_diabetes.csv") as csv_file:
            diabetes_dataset = pd.read_csv(csv_file)
            csv_file.close()
        return diabetes_dataset

    except FileNotFoundError as error_msg:
        print(error_msg)



#data description
def describe_dataset(diabetes_dataset):
    print("\nDisplaying the records of the dataset\n",diabetes_dataset.head())

    starting_records = diabetes_dataset.head(10)
    print("\nDisplaying the starting records of the dataset\n",starting_records)

    ending_records = diabetes_dataset.tail(10)
    print("\nDisplaying the ending records of the dataset\n",ending_records)

    print("\nDisplaying the dataset info\n")
    diabetes_dataset.info()
    
    print("\nDescribing the data\n",diabetes_dataset.describe(include='all'))



#descriptive analysis
def descriptive_analysis(diabetes_dataset):

    #attribute 1
    print("\nDisplaying the minimum value of attribute Pregnant:",diabetes_dataset['Pregnant'].min())
    print("\nDisplaying the maximum value of attribute Pregnant:",diabetes_dataset['Pregnant'].max())
    print("\nDisplaying the mean value of attribute Pregnant:",diabetes_dataset['Pregnant'].mean())
    print("\nDisplaying the median value of attribute Pregnant:",diabetes_dataset['Pregnant'].median())
    print("\nDisplaying the count of attribute Pregnant:",diabetes_dataset['Pregnant'].count())
    print("\nDisplaying the standard deviation of attribute Pregnant:",diabetes_dataset['Pregnant'].std())
    #attribute 2
    print("\nDisplaying the minimum value of attribute Glucose:",diabetes_dataset['Glucose'].min())
    print("\nDisplaying the maximum value of attribute Glucose:",diabetes_dataset['Glucose'].max())
    print("\nDisplaying the mean value of attribute Glucose:",diabetes_dataset['Glucose'].mean())
    print("\nDisplaying the median value of attribute Glucose:",diabetes_dataset['Glucose'].median())
    print("\nDisplaying the count of attribute Glucose:",diabetes_dataset['Glucose'].count())
    print("\nDisplaying the standard deviation of attribute Glucose:",diabetes_dataset['Glucose'].std())
    #attribute 3
    print("\nDisplaying the minimum value of attribute Diastolic_BP:",diabetes_dataset['Diastolic_BP'].min())
    print("\nDisplaying the maximum value of attribute Diastolic_BP:",diabetes_dataset['Diastolic_BP'].max())
    print("\nDisplaying the mean value of attribute Diastolic_BP:",diabetes_dataset['Diastolic_BP'].mean())
    print("\nDisplaying the median value of attribute Diastolic_BP:",diabetes_dataset['Diastolic_BP'].median())
    print("\nDisplaying the count of attribute Diastolic_BP:",diabetes_dataset['Diastolic_BP'].count())
    print("\nDisplaying the standard deviation of attribute Diastolic_BP:",diabetes_dataset['Diastolic_BP'].std())



#data exploration

#graph 1 : Age vs Glucose & Diastolic_BP
def age_glucose_diastolicBP(diabetes_dataset):
    plt.bar(diabetes_dataset['Age'],diabetes_dataset['Glucose'],label='Glucose')   
    plt.bar(diabetes_dataset['Age'],diabetes_dataset['Diastolic_BP'],label='Diastolic_BP')
    plt.xlabel("Age") 
    plt.ylabel("Glucose & Diastolic_BP")
    plt.title("Diabetes Data Analysis")
    plt.legend()
    plt.show()


#graph 2 : Age vs BMI
def age_bmi(diabetes_dataset):
    plt.scatter(diabetes_dataset['Age'],diabetes_dataset['BMI'],color='orange')
    plt.xlabel("Age")
    plt.ylabel("BMI")
    plt.title("Diabetes Data Analysis")
    plt.show()


#graph 3 : BMI vs Class
def bmi_class(diabetes_dataset):
    plt.scatter(diabetes_dataset['BMI'],diabetes_dataset['Class'])   
    plt.xlabel("Age") 
    plt.ylabel("Class")
    plt.title("Diabetes Data Analysis")
    plt.show()

#graph 4 : Bar garph for Outcome of the dataset
def class_bargraph(diabetes_dataset):
    plt.bar(diabetes_dataset['Class'], diabetes_dataset['Class'].count(),label='Class',color=['blue','orange'])   
    plt.xlabel("Class") 
    plt.ylabel("Frequency")
    plt.xlim([0,1])
    plt.ylim([0,500])
    plt.xticks([0,1])
    plt.yticks([0,50,100,150,200,250,300,350,400,450,500])
    plt.title("Diabetes Data Analysis")
    plt.show()

#graph 5 : Age vs Class
def age_class(diabetes_dataset):
    plt.bar(diabetes_dataset['Class'], diabetes_dataset['Age'], label = 'Class')   
    plt.xlabel("Class") 
    plt.ylabel("Age")
    plt.xlim([0,1])
    plt.xticks([0,1])
    plt.title("Diabetes Data Analysis")
    plt.show()

#heatmap
def heatmap_graph(diabetes_dataset):
    corr = diabetes_dataset.corr()
    print("\nOutput table of Correlations\n",corr)
    print("\nHeatmap\n")
    sns.heatmap(corr, xticklabels=corr.columns, yticklabels=corr.columns)
    plt.show()


#data cleaning
def data_cleaning(diabetes_dataset):
    diabetes_dataset['Pregnant'].fillna(value = 0, inplace=True)
    diabetes_dataset['Glucose'].fillna(value = 0, inplace=True)
    diabetes_dataset['Diastolic_BP'].fillna(value = 0, inplace=True)
    diabetes_dataset['Skin_Fold'].fillna(value = 0, inplace=True)    
    diabetes_dataset['Serum_Insulin'].fillna(value = 0, inplace=True)    
    diabetes_dataset['BMI'].fillna(value = 0, inplace=True)    
    diabetes_dataset['Diabetes_Pedigree'].fillna(value = 0, inplace=True)    


#model building

#linear regression model
def linear_regression(diabetes_dataset):
    x = np.asarray(diabetes_dataset.drop('Class',1))
    y = diabetes_dataset[['Class']]

    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.20, random_state=0)
    regressor1 = LinearRegression()    
    regressor1.fit(X_train,y_train)
    y_pred = regressor1.predict(X_test)
    
    cutoff = 0.5                           
    y_pred_classes = np.zeros_like(y_pred)    
    y_pred_classes[y_pred > cutoff] = 1 
    
    y_test_classes = np.zeros_like(y_pred)
    y_test_classes[y_test > cutoff] = 1    

    print("\nConfusion Matrix is:\n")
    print(confusion_matrix(y_test_classes, y_pred_classes))
    print("\nClassification Report is:\n")
    print(classification_report(y_test_classes, y_pred_classes))
    acc1 = accuracy_score(y_true=y_test_classes, y_pred=y_pred_classes)*100
    print("\nAccuracy is:",acc1)


#logistic regression model
def logistic_regression(diabetes_dataset):
    x = np.asarray(diabetes_dataset.drop('Class',1))
    y = diabetes_dataset[['Class']]

    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.20, random_state=0)
    regressor2 = LogisticRegression()    
    regressor2.fit(X_train,y_train)
    y_pred = regressor2.predict(X_test)
    
    print("\nConfusion Matrix is:\n")
    print(confusion_matrix(y_test, y_pred))
    print("\nClassification Report is:\n")
    print(classification_report(y_test, y_pred))
    acc2 = accuracy_score(y_true=y_test, y_pred=y_pred)*100
    print("\nAccuracy is:",acc2)


#random forest classifier
def randomforestclassifier(diabetes_dataset):
    x = np.asarray(diabetes_dataset.drop('Class',1))
    y = diabetes_dataset[['Class']]

    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.20, random_state=0)
    regressor3 = RandomForestClassifier()    
    regressor3.fit(X_train,y_train)
    y_pred = regressor3.predict(X_test)
    
    print("\nConfusion Matrix is:\n")
    print(confusion_matrix(y_test, y_pred))
    print("\nClassification Report is:\n")
    print(classification_report(y_test, y_pred))
    acc3 = accuracy_score(y_true=y_test, y_pred=y_pred)*100
    print("\nAccuracy is:",acc3)



#svc_linear
def svc_linear(diabetes_dataset):
    x = np.asarray(diabetes_dataset.drop('Class',1))
    y = diabetes_dataset[['Class']]

    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.20, random_state=0)
    regressor4a = SVC(kernel = 'linear')    
    regressor4a.fit(X_train,y_train)
    y_pred = regressor4a.predict(X_test)
    
    print("\nConfusion Matrix is:\n")
    print(confusion_matrix(y_test, y_pred))
    print("\nClassification Report is:\n")
    print(classification_report(y_test, y_pred))
    acc4 = accuracy_score(y_true=y_test, y_pred=y_pred)*100
    print("\nAccuracy is:",acc4)



#svc_polynomial
def svc_polynomial(diabetes_dataset):
    x = np.asarray(diabetes_dataset.drop('Class',1))
    y = diabetes_dataset[['Class']]

    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.20, random_state=0)
    regressor4b = SVC(kernel = 'poly')    
    regressor4b.fit(X_train,y_train)
    y_pred = regressor4b.predict(X_test)
    
    print("\nConfusion Matrix is:\n")
    print(confusion_matrix(y_test, y_pred))
    print("\nClassification Report is:\n")
    print(classification_report(y_test, y_pred))
    acc5 = accuracy_score(y_true=y_test, y_pred=y_pred)*100
    print("\nAccuracy is:",acc5)



#svc_gaussian
def svc_gaussian(diabetes_dataset):
    x = np.asarray(diabetes_dataset.drop('Class',1))
    y = diabetes_dataset[['Class']]

    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.20, random_state=0)
    regressor4c = SVC(kernel = 'poly')    
    regressor4c.fit(X_train,y_train)
    y_pred = regressor4c.predict(X_test)
    
    print("\nConfusion Matrix is:\n")
    print(confusion_matrix(y_test, y_pred))
    print("\nClassification Report is:\n")
    print(classification_report(y_test, y_pred))
    acc6 = accuracy_score(y_true=y_test, y_pred=y_pred)*100
    print("\nAccuracy is:",acc6)



#svc_sigmoid
def svc_sigmoid(diabetes_dataset):
    x = np.asarray(diabetes_dataset.drop('Class',1))
    y = diabetes_dataset[['Class']]

    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.20, random_state=0)
    regressor4d = SVC(kernel = 'poly')    
    regressor4d.fit(X_train,y_train)
    y_pred = regressor4d.predict(X_test)
    
    print("\nConfusion Matrix is:\n")
    print(confusion_matrix(y_test, y_pred))
    print("\nClassification Report is:\n")
    print(classification_report(y_test, y_pred))
    acc7 = accuracy_score(y_true=y_test, y_pred=y_pred)*100
    print("\nAccuracy is:",acc7)



#naive bayes classifier
def naive_bayes(diabetes_dataset):
    x = np.asarray(diabetes_dataset.drop('Class',1))
    y = diabetes_dataset[['Class']]

    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.20, random_state=0)
    regressor5 = GaussianNB()    
    regressor5.fit(X_train,y_train)
    y_pred = regressor5.predict(X_test)
    
    print("\nConfusion Matrix is:\n")
    print(confusion_matrix(y_test, y_pred))
    print("\nClassification Report is:\n")
    print(classification_report(y_test, y_pred))
    acc8 = accuracy_score(y_true=y_test, y_pred=y_pred)*100
    print("\nAccuracy is:",acc8)
   


#knn classifier
def knn_classifier(diabetes_dataset):
    x = np.asarray(diabetes_dataset.drop('Class',1))
    y = diabetes_dataset[['Class']]

    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.20, random_state=0)
    regressor6 = KNeighborsClassifier(n_neighbors=3)  
    regressor6.fit(X_train,y_train)
    y_pred = regressor6.predict(X_test)
    
    print("\nConfusion Matrix is:\n")
    print(confusion_matrix(y_test, y_pred))
    print("\nClassification Report is:\n")
    print(classification_report(y_test, y_pred))
    acc9 = accuracy_score(y_true=y_test, y_pred=y_pred)*100
    print("\nAccuracy is:",acc9)



#accuracy for the machine learning models 
def accuracy():
    d = {
        'Algorithm' : ['Linear Regression', 'Logistic Regression','Random Forest Classifier','SVC Linear','SVC Polynomial','SVC Gaussian','SVC Sigmoid','Naive Bayes Classifier','KNN Classifier'],
        'Accuracy' : ['83.12','82.47','81.82','81.82','79.22','79.22','79.22','79.22','72.08']
    }
    data_table = pd.DataFrame(data=d)
    print("\nAccuracy Table:\n",data_table)



#main
if __name__=="__main__":
    
    #data collection
    csv_data = csv_read_data()

    while(True):
        user_input = input("\nSelect from the options below:\n 1. Data Description\n 2. Descriptive Analysis\n 3. Data Visualization\n 4. Predictive Analysis\n 5. Accuracy Table\n 6. Exit\n")

        if user_input == '1':
            #data description
            describe_dataset(csv_data)

        elif user_input == '2':
            #descriptive analysis
            descriptive_analysis(csv_data)

        elif user_input == '3':
            #data visualization
            while(True):
                graph_user_input = input("\nSelect from the options below for visualizations:\n 1. Age vs Glucose & Diastolic_BP\n 2. Age vs BMI\n 3. BMI vs Class\n 4. Class Graph\n 5. Age vs Class\n 6. Correlations & Heatmap\n 7. Exit\n")
            
                if graph_user_input == '1' :
                    age_glucose_diastolicBP(csv_data)

                elif graph_user_input == '2' :
                    age_bmi(csv_data)

                elif graph_user_input == '3':
                    bmi_class(csv_data)

                elif graph_user_input == '4':
                    class_bargraph(csv_data)

                elif graph_user_input == '5':
                    age_class(csv_data)
                    
                elif graph_user_input == '6':
                    heatmap_graph(csv_data)

                else:
                    break
        
        elif user_input == '4':
            #data cleaning
            data_cleaning(csv_data)

            #predictive analysis
            while(True):
                model_user_input = input("\nSelect from the options below for predictive model:\n 1. Linear Regression\n 2. Logistic Regression\n 3. Random Forest Classifier\n 4. Support Vector Machine\n 5. Naive Bayes Classifier\n 6. KNN Classifier\n 7. Exit\n")

                if model_user_input == '1':
                    linear_regression(csv_data)

                elif model_user_input == '2':
                    logistic_regression(csv_data)

                elif model_user_input == '3':
                    randomforestclassifier(csv_data)
            
                elif model_user_input == '4':
                    while(True):
                        svm_input = input("\nSelect from the below options for SVM Classifier:\n 1. SVM Linear\n 2. SVM Polynomial\n 3. SVM Gaussian\n 4. SVM Sigmoid\n 5.Exit\n")

                        if svm_input == '1':
                            svc_linear(csv_data)

                        elif svm_input == '2':
                            svc_polynomial(csv_data)

                        elif svm_input == '3':
                            svc_gaussian(csv_data)

                        elif svm_input == '4':
                            svc_sigmoid(csv_data)

                        else:
                            break

                elif model_user_input == '5':
                   naive_bayes(csv_data)

                elif model_user_input == '6':
                    knn_classifier(csv_data)

                else:
                    break

        elif user_input == '5':
            accuracy()

        else:
            exit()
