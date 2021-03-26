from flask import Flask,session,request,jsonify ,render_template,session
import json
import numpy as np
from business_logic.check_attritionLOGIC import CheckAttritionLogic

app = Flask(__name__) # '.' means the current directory

@app.route('/')
def index():
   return "<html><body><h1>'Only backend is created currently.'</h1></body></html>"



#use for training model and then getting result. For already trained model scroll down
@app.route('/check_attrition',methods = ['POST'])
def check_attrition():
    
    #all data has to be in float form , string value will give error
    ''' Input type is a list should be of following order,:
    
    ['satisfaction_level', 'last_evalation', 'nmber_project',
       'average_montly_hors', 'time_spend_company (years)',
       'Work_accident', 
       'promoted_in_last_5years (1 means yes)', 'salary',
       'department_RandD', 'department_acconting', 'department_hr',
       'department_management', 'department_marketing',
       'department_prodct_mng', 'department_sales', 'department_spport',
       'department_technical']
       
       
       Note:
       In departments put 1 at relevant position and 0 at rest
       
       for testing you can use following list:
       [   0.66,    0.65 ,   5.0,    161.0 ,     3.0,      0.0,      0.0,     1.0,      0.0,
        0.0,      0.0,      0.0,      0.0,      0.0,      0.0,      1.0,      0.0  ]
    
    '''
    json_string = request.data
    obj = json.loads(json_string)
    input_list = []
    input_list.append(obj['input'])
    
    arr = np.array(input_list)
    
    
    return jsonify(CheckAttritionLogic().attrition_result(arr))


#use for getting direct result
@app.route('/check_attrition_on_old_trained_model',methods = ['POST'])
def check_attrition_on_old_trained_model():
    json_string = request.data
    obj = json.loads(json_string)
    input_list = []
    input_list.append(obj['input'])
    
    arr = np.array(input_list)
    return jsonify(CheckAttritionLogic().get_result_on_earlier_trained_model(arr))

if __name__ == "__main__":
    app.run(host='0.0.0.0',port = 5000,debug = True)