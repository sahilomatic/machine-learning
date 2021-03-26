import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.utils import resample

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline




class CheckAttritionDAO:
    
    
    def __init__(self):
        self.df = pd.read_csv("../documents/Employee_Attrition - Employee_Attrition.csv")
        
        print("Pre Processing of data is done in constructor")
        # Map salary into integers
        salary_map = {"low": 0, "medium": 1, "high": 2}
        self.df["salary"] = self.df["salary"].map(salary_map)
        
        # Create dummy variables for department feature
        self.df = pd.get_dummies(self.df, columns=["department"], drop_first=True)
        self.df.head()
        
        
        # Get number of positve and negative examples
        pos = self.df[self.df["left_company (1 means yes)"] == 1].shape[0]
        neg = self.df[self.df["left_company (1 means yes)"] == 0].shape[0]
        print("Number of employees left company in database = {}".format(pos))
        print("Number of employees stayed in company according to database= {}".format(neg))
        
        ratio = (pos*100/neg) 
        
        
        # Convert dataframe into numpy objects and split them into
        # train and test sets: 80/20
        from sklearn.model_selection import train_test_split
        X = self.df.loc[:, self.df.columns != "left_company (1 means yes)"].values
        y = self.df.loc[:, self.df.columns == "left_company (1 means yes)"].values.flatten()
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.2, stratify=y, random_state=1)
        
        
        if(ratio!=50):
            print("Ratio between positive and negative results : "+str(ratio)+ "%")
            print("Thus data is not balanced.")
            # Upsample minority class
            print("Upsampled data set is created")
            self.X_train_u, self.y_train_u = resample(self.X_train[self.y_train == 1],
                                            self.y_train[self.y_train == 1],
                                            replace=True,
                                            n_samples=self.X_train[self.y_train == 0].shape[0],
                                            random_state=1)
            self.X_train_u = np.concatenate((self.X_train[self.y_train == 0], self.X_train_u))
            self.y_train_u = np.concatenate((self.y_train[self.y_train == 0], self.y_train_u))
            
            # Downsample majority class
            print("Downsampled data set is created")
            self.X_train_d, self.y_train_d = resample(self.X_train[self.y_train == 0],
                                            self.y_train[self.y_train == 0],
                                            replace=True,
                                            n_samples=self.X_train[self.y_train == 1].shape[0],
                                            random_state=1)
            self.X_train_d = np.concatenate((self.X_train[self.y_train == 1], self.X_train_d))
            self.y_train_d = np.concatenate((self.y_train[self.y_train == 1], self.y_train_d))
   

    def logistic_regression(self,make_prediction):
        #Pipeline is created for features scaling  and Predictive Model
        self.model1 = make_pipeline(StandardScaler(),
                           LogisticRegression(class_weight="balanced"))
        
        methods_data = {"Original": (self.X_train, self.y_train),
                "Upsampled": (self.X_train_u, self.y_train_u),
                "Downsampled": (self.X_train_d, self.y_train_d)}
        
        
        output_score = []
        data_type = None
        max = 0
        for k,v in  methods_data.items():
            
            self.model1.fit(v[0],v[1])
            best_score = self.model1.score(self.X_test,self.y_test)
            if( best_score> max):
                max = best_score
                data_type = k
        
        if(data_type == "Original"):
            self.final_X = self.X_train
            self.final_Y = self.y_train
        elif(data_type == "Upsampled"):
            self.final_X = self.X_train_u
            self.final_Y = self.y_train_u   
        elif(data_type == "Downsampled"):
            self.final_X = self.X_train_d
            self.final_Y = self.y_train_d         
            
        print("Best accuracy score is with " + data_type+ " data set. Same data set will be used for all algorithms.")     
        self.model1.fit(self.final_X,self.final_Y)
        print("Accuracy score with Logistic Regression is : ",self.model1.score(self.X_test, self.y_test))
        
        return self.model1.predict(make_prediction)[0]
    
    
    def svm(self,make_prediction):
        self.model3 = make_pipeline(StandardScaler(),
                        SVC(C=0.01,
                            gamma=0.1,
                            kernel="poly",
                            degree=5,
                            coef0=10,
                            probability=True))
        
        
        self.model3.fit(self.final_X,self.final_Y)
        print("Accuracy score with SVM is : ",self.model3.score(self.X_test, self.y_test))
        return self.model3.predict(make_prediction)[0]
    
    def knn(self,make_prediction):
        self.model2 = make_pipeline(StandardScaler(), KNeighborsClassifier())
        self.model2.fit(self.final_X,self.final_Y)
        print("Accuracy score with K nearest neighbour is : ",self.model2.score(self.X_test, self.y_test))
        return self.model2.predict(make_prediction)[0]
        
        
if __name__ == "__main__":
    dao = CheckAttritionDAO()
    arr = np.array([[   0.66,    0.65 ,   5.0,    161.0 ,     3.0,      0.0,      0.0,     1.0,      0.0,
    0.0,      0.0,      0.0,      0.0,      0.0,      0.0,      1.0,      0.0  ]])
    print(dao.logistic_regression(arr))
            