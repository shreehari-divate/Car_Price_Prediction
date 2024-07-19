import logging
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
        year = request.form.get("year")
        km_driven = request.form.get("km_driven")
        engine = request.form.get('engine')
        mileage = request.form.get("mileage")
        max_power = request.form.get("max_power")
        fuel = request.form.get("fuel")
        transmission = request.form.get("transmission")
        seller_type = request.form.get("seller_type")
        owner = request.form.get("owner")

        year = int(year)
        km_driven = int(km_driven)
        engine = int(engine)
        mileage = float(mileage)
        max_power = float(max_power)
        data = CustomData(
            year=year,
            kmdriven=km_driven,
            fuel=fuel,
            engine=engine,
            mileage=mileage,
            power=max_power,
            transmission=transmission,
            seller=seller_type,
            owner=owner)
        
        pred_df=data.get_data_as_df()
        print("dataframe cols: ",pred_df.columns)
        print(pred_df)

        predict_pipeline=PredictPipeline()  
        results=predict_pipeline.predict(pred_df)[0]
        try:
            res_msg=f"You car {car_name}'s approximate price is Rs. {results:.2f}"
        except Exception as e:
            res_msg="Error formatting the results"+e
        return render_template('index.html',results=res_msg) 

    else:
        return render_template('index.html')
                                    
if __name__=="__main__":
    app.run(debug=True, host='0.0.0.0', port=5000)