from flask import Flask, request, jsonify
from flask_cors import CORS
import sqlite3
import os
import pandas as pd
import numpy as np
# import model.pkl
import pickle

clf = pickle.load(open('model.pkl', 'rb'))
clf2 = pickle.load(open('model2.pkl', 'rb'))

l1 = ['back_pain', 'constipation', 'abdominal_pain', 'diarrhoea', 'mild_fever', 'yellow_urine',
      'yellowing_of_eyes', 'acute_liver_failure', 'fluid_overload', 'swelling_of_stomach',
      'swelled_lymph_nodes', 'malaise', 'blurred_and_distorted_vision', 'phlegm', 'throat_irritation',
      'redness_of_eyes', 'sinus_pressure', 'runny_nose', 'congestion', 'chest_pain', 'weakness_in_limbs',
      'fast_heart_rate', 'pain_during_bowel_movements', 'pain_in_anal_region', 'bloody_stool',
      'irritation_in_anus', 'neck_pain', 'dizziness', 'cramps', 'bruising', 'obesity', 'swollen_legs',
      'swollen_blood_vessels', 'puffy_face_and_eyes', 'enlarged_thyroid', 'brittle_nails',
      'swollen_extremeties', 'excessive_hunger', 'extra_marital_contacts', 'drying_and_tingling_lips',
      'slurred_speech', 'knee_pain', 'hip_joint_pain', 'muscle_weakness', 'stiff_neck', 'swelling_joints',
      'movement_stiffness', 'spinning_movements', 'loss_of_balance', 'unsteadiness',
      'weakness_of_one_body_side', 'loss_of_smell', 'bladder_discomfort', 'foul_smell_of urine',
      'continuous_feel_of_urine', 'passage_of_gases', 'internal_itching', 'toxic_look_(typhos)',
      'depression', 'irritability', 'muscle_pain', 'altered_sensorium', 'red_spots_over_body', 'belly_pain',
      'abnormal_menstruation', 'dischromic _patches', 'watering_from_eyes', 'increased_appetite', 'polyuria', 'family_history', 'mucoid_sputum',
      'rusty_sputum', 'lack_of_concentration', 'visual_disturbances', 'receiving_blood_transfusion',
      'receiving_unsterile_injections', 'coma', 'stomach_bleeding', 'distention_of_abdomen',
      'history_of_alcohol_consumption', 'fluid_overload', 'blood_in_sputum', 'prominent_veins_on_calf',
      'palpitations', 'painful_walking', 'pus_filled_pimples', 'blackheads', 'scurring', 'skin_peeling',
      'silver_like_dusting', 'small_dents_in_nails', 'inflammatory_nails', 'blister', 'red_sore_around_nose',
      'yellow_crust_ooze']
d1 = {0: 'Fungal infection', 1: 'Allergy', 2: 'GERD', 3: 'Chronic cholestasis', 4: 'Drug Reaction', 5: 'Peptic ulcer diseae', 6: 'AIDS', 7: 'Diabetes ', 8: 'Gastroenteritis', 9: 'Bronchial Asthma', 10: 'Hypertension ', 11: 'Migraine', 12: 'Cervical spondylosis',
      13: 'Paralysis (brain hemorrhage)', 14: 'Jaundice', 15: 'Malaria', 16: 'Chicken pox', 17: 'Dengue', 18: 'Typhoid', 19: 'hepatitis A', 20: 'Hepatitis B', 21: 'Hepatitis C', 22: 'Hepatitis D', 23: 'Hepatitis E', 24: 'Alcoholic hepatitis', 25: 'Tuberculosis', 26: 'Common Cold', 27: 'Pneumonia', 28: 'Dimorphic hemmorhoids(piles)', 29: 'Heart attack', 30: 'Varicose veins', 31: 'Hypothyroidism', 32: 'Hyperthyroidism', 33: 'Hypoglycemia', 34: 'Osteoarthristis', 35: 'Arthritis', 36: '(vertigo) Paroymsal  Positional Vertigo', 37: 'Acne', 38: 'Urinary tract infection', 39: 'Psoriasis', 40: 'Impetigo'}


def predict(lis):
    yData = [0 for i in range(len(l1))]
    for i in range(len(l1)):
        if(l1[i] in lis):
            yData[i] = 1
    pred = clf.predict([yData])[0]
    return [d1[pred]]


def predict2(lis):
    yData = [0 for i in range(len(l1))]
    for i in range(len(l1)):
        if(l1[i] in lis):
            yData[i] = 1
    data = pd.DataFrame(data=[yData], columns=l1)
    pred = clf.predict_proba(data)[0]
    sorted = np.sort(pred)
    pred1 = d1[np.where(pred == sorted[-1])[0][0]] + \
        " " + str(int(sorted[-1]*100)) + "%"
    pred2 = ''
    if (sorted[-2] > 0.4):
        pred2 = d1[np.where(pred == sorted[-2])[0][0]] + \
            " " + str(int(sorted[-2]*100))+"%"
    return [pred1, pred2]


app = Flask(__name__)
CORS(app)

# Flask maps HTTP requests to Python functions.
# The process of mapping URLs to functions is called routing.


@app.route('/', methods=['GET'])
def home():
    return "<h1>Distant Reading Archive</h1><p>This is a prototype API</p>"

# A route to return all of available entries i our catalog.


@app.errorhandler(404)
def page_not_found(e):
    return "<h1>404</h1><p>The resource could not be found</p>", 404


@app.route('/api/predict', methods=['POST'])
def add_predict():
    # Receives the data in JSON format in a HTTP POST request
    if not request.is_json:
        return "<p>The content isn't of type JSON<\p>"
    else:
        data = request.get_json()
        lis = data['symptoms']
        return jsonify({"Prediction": predict(lis)})


@app.route('/api/predict2', methods=['POST'])
def add_predict2():
    # Receives the data in JSON format in a HTTP POST request
    if not request.is_json:
        return "<p>The content isn't of type JSON<\p>"
    else:
        data = request.get_json()
        lis = data['symptoms']
        return jsonify({"Prediction": predict2(lis)})


# A method that runs the application server.
if __name__ == "__main__":
    # Threaded option to enable multiple instances for multiple user access support
    app.run(debug=False, threaded=True)
