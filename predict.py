import joblib
import preprocessing as p


def predict(user_data):
    input_data = p.clear_input(user_data)
    text_vector = joblib.load('models/vectorizer.joblib')
    model = joblib.load('models/trained_model.joblib')

    v_text = text_vector.transform([input_data])
    prediction = model.predict(v_text)
    return prediction[0]


"""

Note : This is our prediction model which is based on Logistic Regression Model.

"""