from flask import Flask, render_template, request
import pickle
import numpy as np
import pandas as pd

with open('model_pkl' , 'rb') as f:
   model = pickle.load(f)

cars_no = pd.read_csv('/mnt/c/Users/youse/Desktop/miskdoc/misk-DSI/saudi_used_cars_price_prediciton/cars_no_test.csv')

app = Flask(__name__)

@app.route('/')
def test():
    return render_template('home.html')

@app.route('/predict', methods=['POST'])
def home():
    car_make = request.form['make']
    car_make_no = cars_no['car_make_no'].where(cars_no['car_make'] == car_make).dropna().drop_duplicates().values[0]

    car_model = request.form['model']
    car_model_no = cars_no['car_model_no'].where(cars_no['car_model'] == car_model).dropna().drop_duplicates().values[0]

    model_year = request.form['year']

    km = np.sqrt(int(request.form['km']))


    transmission = request.form['transmission']
    transmission_no = cars_no['transmission_no'].where(cars_no['transmission'] == transmission).dropna().drop_duplicates().values[0]

    fuel = request.form['fuel']
    fuel_no = cars_no['fuel_no'].where(cars_no['fuel'] == fuel).dropna().drop_duplicates().values[0]

    color = request.form['color']
    color_no = cars_no['color_no'].where(cars_no['color'] == color).dropna().drop_duplicates().values[0]

    no_doors = request.form['no_doors']

    engine_capacity = np.sqrt(int(request.form['engine_capacity']))

    no_cylinders = request.form['no_cylinders']

    Accident_history = request.form['Accident_history']
    Accident_history_no = cars_no['Accident_history_no'].where(cars_no['Accident_history'] == Accident_history).dropna().drop_duplicates().values[0]

    arr = np.array([[car_make_no,car_model_no,model_year,km,transmission_no,fuel_no,color_no,no_doors,engine_capacity
,no_cylinders,Accident_history_no]])
    pred = model.predict(arr)
    return render_template('prediction.html', data=pred)

@app.route('/report')
def report():
    return render_template('report.html')

if __name__ == "__main__":
    app.run(debug=True)
    from waitress import serve
    serve(app, host="0.0.0.0", port=8080)
