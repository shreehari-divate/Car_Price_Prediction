import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from flask import Flask, render_template, request,url_for
from src.pipeline.predict_pipeline import CustomData,PredictPipeline

app=Flask(__name__)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict",methods=["POST","GET"])
def predict():
    if request.method=="POST":
        car_name=request.form.get("name")
        data=CustomData(
            year=request.form.get("year"),
            kmdriven=request.form.get("kmdriven"),
            fuel=request.form.get("fuel"),
            engine=request.form.get('engine'),
            mileage=request.form.get("mileage"),
            power=request.form.get("power"),
            transmission=request.form.get("transmission"),
            seller=request.form.get("seller"),
            owner=request.form.get("owner")
        )
        pred_df=data.get_data_as_df()
        print("dataframe cols: ",pred_df.columns)
        print(pred_df)

        predict_pipeline=PredictPipeline()  
        results=predict_pipeline.predict(pred_df)[0]
        res_msg=f"You car {car_name}'s approximate price is Rs. {results:.2f}"
        return render_template('index.html',results=res_msg) 
    else:
        return render_template('index.html')
                                    
if __name__=="__main__":
    app.run(debug=True)