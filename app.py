from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)

# Load model và scaler
model = joblib.load('best_model_rf.pkl')  # hoặc best_model_knn.pkl
scaler = joblib.load('scaler.pkl')        # scaler đã lưu khi huấn luyện

@app.route('/', methods=['GET', 'POST'])
def home():
    prediction_text = ""
    if request.method == 'POST':
        try:
            age = float(request.form.get('age', 25))
            systolic = float(request.form.get('systolic', 110))
            diastolic = float(request.form.get('diastolic', 70))
            bs = float(request.form.get('bs', 5))
            temp = float(request.form.get('bodytemp', 98.6))
            hr = float(request.form.get('heartrate', 80))

            input_data = np.array([[age, systolic, diastolic, bs, temp, hr]])
            input_scaled = scaler.transform(input_data)

            pred = model.predict(input_scaled)[0]
            label_map = {0: 'High Risk', 1: 'Low Risk', 2: 'Mid Risk'}
            result = label_map.get(pred, 'Unknown')

            prediction_text = f"⚠️ <strong>Kết quả dự đoán:</strong> {result}"
        except Exception as e:
            prediction_text = f"❌ Lỗi: {str(e)}"

    return render_template('index.html', prediction_text=prediction_text)

if __name__ == '__main__':
    app.run(debug=True)
