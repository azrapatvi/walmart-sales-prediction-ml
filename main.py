from flask import Flask,render_template,request
import pickle
import pandas as pd

app=Flask(__name__)

@app.route("/",methods=['GET','POST'])
def home():
    if request.method=='GET':
        return render_template("index.html")
    else:
        store=int(request.form.get('store'))
        holiday_flag=request.form.get('holiday_week')
        temperature=float(request.form.get('temperature'))
        cpi=float(request.form.get('cpi'))
        unemployment=float(request.form.get('unemployment'))
        month=int(request.form.get('month'))
        year=int(request.form.get('year'))
        lag1=float(request.form.get('lag1'))

        with open("scaler.pkl","rb")as f:
            scaler=pickle.load(f)

        with open("RandomForestRegressor.pkl","rb") as f:
            model=pickle.load(f)

        df = pd.DataFrame([{
            "store": store,
            "holiday_flag": holiday_flag,
            "temperature": temperature,
            "cpi": cpi,
            "unemployment": unemployment,
            "month": month,
            "year": year,
            "lag1": lag1
        }])

        scaled_df=scaler.transform(df)

        prediction=model.predict(scaled_df)


        return render_template("index.html",prediction=prediction[0])
    
  

app.run(debug=True)