import logging
import sys
import os

from sklearn.preprocessing import OneHotEncoder
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np
from flask import Flask, render_template, request,url_for
from src.pipeline.predict_pipeline import CustomData,PredictPipeline

app=Flask(__name__)

@app.route("/")
def index():
    return render_template("index.html")


#original
@app.route("/predict",methods=["POST","GET"])
def predict():
    if request.method=="POST":
        try:
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

            if not isinstance(results,(int,float)):
                raise ValueError("Prediction is not a number")
            
            res_msg=f"You car {car_name}'s approximate price is Rs. {results:.2f}"
            return render_template('index.html',results=res_msg) 
        
        except Exception as e:
            res_msg=e
            render_template('index.html',error=res_msg)
    else:
        return render_template('index.html')
                                    

#second
'''@app.route("/predict", methods=["POST", "GET"])
def predict():
    if request.method == "POST":
        try:
            app.logger.debug("Request form data: %s", request.form)
            form_data = get_form_data(request)
            app.logger.debug("Processed form data: %s", form_data)
            data = process_form_data(form_data)
            app.logger.debug("Data for prediction: %s", data)
            results = make_prediction(data)
            app.logger.debug("Prediction results: %s", results)
            res_msg = format_result_message(form_data['car_name'], results)
            return render_template('index.html', results=res_msg)
        except Exception as e:
            app.logger.error("Error occurred during prediction: %s", str(e))
            return render_template('index.html', results="Error in prediction. Please check your inputs and try again.")
    else:
        return render_template('index.html')

def get_form_data(request):
    """Extract form data from the request."""
    try:
        form_data = {
            "car_name": request.form.get("name"),
            "year": int(request.form.get("year")),
            "km_driven": int(request.form.get("km_driven")),
            "engine": int(request.form.get("engine")),
            "mileage": float(request.form.get("mileage")),
            "max_power": float(request.form.get("max_power")),
            "fuel": request.form.get("fuel"),
            "transmission": request.form.get("transmission"),
            "seller_type": request.form.get("seller_type"),
            "owner": request.form.get("owner")
        }
        return form_data
    except Exception as e:
        app.logger.error("Error in extracting form data: %s", str(e))
        raise

def process_form_data(form_data):
    """Process form data into the format required for prediction."""
    try:
        data = CustomData(
            year=form_data["year"],
            kmdriven=form_data["km_driven"],
            fuel=form_data["fuel"],
            engine=form_data["engine"],
            mileage=form_data["mileage"],
            power=form_data["max_power"],
            transmission=form_data["transmission"],
            seller=form_data["seller_type"],
            owner=form_data["owner"]
        )
        pred_df = data.get_data_as_df()
        app.logger.debug("Dataframe columns: %s", pred_df.columns)
        app.logger.debug("Dataframe content: %s", pred_df)
        return pred_df
    except Exception as e:
        app.logger.error("Error in processing form data: %s", str(e))
        raise

def make_prediction(data):
    """Make prediction using the processed data."""
    try:
        predict_pipeline = PredictPipeline()
        results = predict_pipeline.predict(data)[0]
        if not isinstance(results, (int, float)):
            raise ValueError("Prediction result is not a number")
        return results
    except Exception as e:
        app.logger.error("Error in making prediction: %s", str(e))
        raise

def format_result_message(car_name, results):
    """Format the result message for display."""
    try:
        return f"Your car {car_name}'s approximate price is Rs. {results:.2f}"
    except Exception as e:
        app.logger.error("Error in formatting result message: %s", str(e))
        raise'''

if __name__=="__main__":
    app.run(debug=False, host='0.0.0.0')