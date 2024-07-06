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
        try:
            car_name=request.form.get("name")
            year = int(request.form.get("year"))
            kmdriven = int(request.form.get("kmdriven"))
            engine = int(request.form.get('engine'))
            mileage = float(request.form.get("mileage"))
            power = float(request.form.get("power"))
            fuel = request.form.get("fuel")
            transmission = request.form.get("transmission")
            seller = request.form.get("seller")
            owner = request.form.get("owner")

            data = CustomData(
                year=year,
                kmdriven=kmdriven,
                fuel=fuel,
                engine=engine,
                mileage=mileage,
                power=power,
                transmission=transmission,
                seller=seller,
                owner=owner)
            pred_df=data.get_data_as_df()
            print("dataframe cols: ",pred_df.columns)
            print(pred_df)

            predict_pipeline=PredictPipeline()  
            results=predict_pipeline.predict(pred_df)[0]
            res_msg=f"You car {car_name}'s approximate price is Rs. {results:.2f}"
            return render_template('index.html',results=res_msg) 
        except Exception as e:
            logging.error(f"error occured: {e} ",exc_info=True)
            return render_template('index.html',results="error occured")
    else:
        return render_template('index.html')
                                    
if __name__=="__main__":
    app.run(debug=True, host='0.0.0.0', port=5000)