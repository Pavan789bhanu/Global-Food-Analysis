from flask import Flask,request,render_template
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from src.pipelines.predict_pipeline import CustomData,PredictPipeline

application=Flask(__name__)

app=application

## Route for a home page

@app.route('/')
def index():
    return render_template('home.html') 

@app.route('/predictdata',methods=['GET','POST'])
def predict_datapoint():
    if request.method=='GET':
        return render_template('index.html')
    else:
        data=CustomData(
            adm0_id=int(request.form.get('adm0_id')),
            adm1_id=int(request.form.get('adm1_id')),
            cm_id=int(request.form.get('cm_id')),
            pt_id=int(request.form.get('pt_id')),
            mkt_id=int(request.form.get('mkt_id')),
            mp_year=int(request.form.get('mp_year'))

        )
        pred_df=data.get_data_as_data_frame()
        print(pred_df)
        print("Before Prediction")

        predict_pipeline=PredictPipeline()
        print("Mid Prediction")
        results=predict_pipeline.predict(pred_df)
        print("after Prediction")
        return render_template('index.html',results=results[0])
    

if __name__=="__main__":
    app.run(host="0.0.0.0")        