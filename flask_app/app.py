from flask import Flask, request, jsonify
import joblib

app = Flask(__name__)
model = joblib.load('best_model.joblib')


@app.route('/predict', methods=['POST'])
def predict():
    print(request)
    data = request.get_json(force=True)
    print(data)
    prediction = model.predict([data['features']])
    return jsonify({'prediction': prediction.tolist()})


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5080)
