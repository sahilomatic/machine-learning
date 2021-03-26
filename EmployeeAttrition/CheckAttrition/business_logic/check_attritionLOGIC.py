from database.check_attrition import CheckAttritionDAO
import numpy as np
import pickle
class CheckAttritionLogic:
    
    def major_voting(self,results):
        
            print("major_voting")
            count = max(set(results), key = results.count)
            
            print(count)
            if(count == 1):
                print("****** Final Answer*********")
                print("By major voting(Ensemble Learning) of various algorithms , employee will leave")
                return "By major voting(Ensemble Learning) of various algorithms , employee will leave"
            
            elif(count == 0):
                print("****** Final Answer*********")
                print("By major voting(Ensemble Learning) of various algorithms , employee will not leave")
                return "By major voting(Ensemble Learning) of various algorithms , employee will not leave"

            
            else:
                raise Exception 
        
    
    def attrition_result(self,arr):
        try:
            results= []
            dao = CheckAttritionDAO()
            
            
            print("lr")
            results.append(dao.logistic_regression(arr))
            print("svm")
            results.append(dao.svm(arr))
            
            print("knn")
            results.append(dao.knn(arr))
            
            
            return self.major_voting(results)
            
        except Exception:
            return "Some Error Occurred."    
        finally:
            if(dao is not None):
                print("********* models created are getting serialized for future use******")
                filename = "../documents/trained_models"
                outfile = open(filename,'wb')
                pickle.dump(dao,outfile)
                outfile.close()
    
    def get_result_on_earlier_trained_model(self,arr):
        
        filename = "../documents/trained_models"
        infile = open(filename,'rb')
        if(infile is not None):
            print("********* models created are getting serialized for future use******")

            obj = pickle.load(infile)
            infile.close()
            
            results= []
            
            
            
            results.append(obj.model1.predict(arr)[0])
            
            results.append(obj.model1.predict(arr)[0])
                
            print("Logistic Regression, K-nearest neighbour and Support vector Machine are applied.")
            results.append(obj.model1.predict(arr)[0])
            
            
            return self.major_voting(results)
        else:
            return "Pickled file is not present."
        
    
if __name__ == "__main__":
    dao = CheckAttritionLogic()
    arr = np.array([[   0.98,    0.66 ,   2.0 ,   255.0 ,     3.0,      0.0,      0.0,      0.0,      0.0,
    0.0 ,     0.0,      0.0,      0.0 ,     0.0 ,     0.0  ,    1.0 ,     0.0  ]])
    #print(arr)
    dao.attrition_result(arr)
    #dao.get_result_on_earlier_trained_model(arr)
    
    #dao.major_voting([0,1,1,1,0])