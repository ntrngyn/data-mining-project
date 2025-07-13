from flask import Flask, render_template, request
import joblib
import pandas as pd
import numpy as np

app = Flask(__name__)

# Load Random Forest model, scaler, and label encoder
rf_model = joblib.load('rf_model.pkl')
scaler = joblib.load('scaler.pkl')
label_encoder = joblib.load('label_encoder.pkl')

@app.route('/', methods=['GET', 'POST'])
def home():
    prediction_text = ""
    age = systolic = diastolic = bs = bodytemp = heartrate = ""
    temp_unit = "C"

    if request.method == 'POST':
        try:
            # Lấy dữ liệu từ form
            age = float(request.form.get('age', 25))
            systolic = float(request.form.get('systolic', 110))
            diastolic = float(request.form.get('diastolic', 70))
            bs = float(request.form.get('bs', 6))
            bodytemp = float(request.form.get('bodytemp', 37))
            heartrate = float(request.form.get('heartrate', 80))
            temp_unit = request.form.get('temp_unit', 'C')

            # Kiểm tra và chuyển đổi nhiệt độ nếu là độ F
            if temp_unit == 'F':
                if not (86 <= bodytemp <= 113):
                    raise ValueError("Nhiệt độ °F phải nằm trong khoảng 86–113.")
                bodytemp = ((bodytemp - 32) * 5 / 9).__round__(1) 
                temp_unit = 'C'
            else:
                if not (30 <= bodytemp <= 45):
                    raise ValueError("Nhiệt độ °C phải nằm trong khoảng 30–45.")

            # Kiểm tra các giá trị bất thường
            # Kiểm tra các giá trị bất thường
            abnormal_conditions = []
            if age > 45:
                abnormal_conditions.append("Tuổi của bạn (>45) khá lớn, là yếu tố nguy cơ cao trong thai kỳ.")
            if age < 17:
                abnormal_conditions.append("Tuổi của bạn (<17) quá nhỏ, là yếu tố nguy cơ cao trong thai kỳ.")
            if systolic > 140:
                abnormal_conditions.append(f"Huyết áp tâm thu ({systolic} mmHg) cao hơn mức bình thường (90–140 mmHg).")
            elif systolic < 90:
                abnormal_conditions.append(f"Huyết áp tâm thu ({systolic} mmHg) thấp hơn mức bình thường (90–140 mmHg).")
            if diastolic > 90:
                abnormal_conditions.append(f"Huyết áp tâm trương ({diastolic} mmHg) cao hơn mức bình thường (60–90 mmHg).")
            elif diastolic < 60:
                abnormal_conditions.append(f"Huyết áp tâm trương ({diastolic} mmHg) thấp hơn mức bình thường (60–90 mmHg).")
            if bs > 7.0:
                abnormal_conditions.append(f"Chỉ số đường huyết ({bs} mmol/L) cao hơn mức bình thường (3.9–7.0 mmol/L).")
            elif bs < 3.9:
                abnormal_conditions.append(f"Chỉ số đường huyết ({bs} mmol/L) thấp hơn mức bình thường (3.9–7.0 mmol/L).")
            if bodytemp > 38:
                abnormal_conditions.append(f"Nhiệt độ cơ thể ({bodytemp:.1f} °C) cao hơn mức bình thường (36–38 °C).")
            elif bodytemp < 36:
                abnormal_conditions.append(f"Nhiệt độ cơ thể ({bodytemp:.1f} °C) thấp hơn mức bình thường (36–38 °C).")
            if heartrate > 100:
                abnormal_conditions.append(f"Nhịp tim ({heartrate} bpm) cao hơn mức bình thường (60–100 bpm).")
            elif heartrate < 60:
                abnormal_conditions.append(f"Nhịp tim ({heartrate} bpm) thấp hơn mức bình thường (60–100 bpm).")

            # Tạo DataFrame đầu vào
            input_data = pd.DataFrame([[age, systolic, diastolic, bs, bodytemp, heartrate]],
                                      columns=['Age', 'SystolicBP', 'DiastolicBP', 'BS', 'BodyTemp', 'HeartRate'])

            # Chuẩn hóa dữ liệu và dự đoán
            input_scaled = scaler.transform(input_data)
            prediction = rf_model.predict(input_scaled)[0]
            prediction_label = label_encoder.inverse_transform([prediction])[0]

            # Ánh xạ nhãn sang tiếng Việt
            label_map = {
                'high risk': 'Rủi ro cao',
                'mid risk': 'Rủi ro trung bình',
                'low risk': 'Rủi ro thấp'
            }
            prediction_vn = label_map.get(prediction_label.lower(), 'Không xác định')

            # Ánh xạ nhãn sang tiếng Việt
            if prediction_label.lower() == 'low risk':
                prediction_text = (
                    f'''
                    <div style="border-radius: 8px; border: 2px solid #4CAF50; background-color: #E8F5E9; padding: 20px; margin: 15px 0;">
                        <h6 style="color: #2E7D32; font-size: 20px; font-weight: bold; text-align: center;">{prediction_vn}</h6>
                        <p style="color: #1B5E20; font-weight: 500; font-size: 16px;">
                            ✅ <strong>Tốt:</strong> Kết quả cho thấy rủi ro thấp. Tiếp tục duy trì lối sống lành mạnh và kiểm tra sức khỏe định kỳ.
                        </p>
                    </div>
                    '''
                )
            elif prediction_label.lower() == 'mid risk':
                prediction_text = (
                    f'''
                    <div style="border-radius: 8px; border: 2px solid #FFA726; background-color: #FFF8E1; padding: 20px; margin: 15px 0;">
                        <h6 style="color: #F57C00; font-size: 20px; font-weight: bold; text-align: center;">{prediction_vn}</h6>
                        <p style="color: #E65100; font-weight: 500; font-size: 16px;">
                            ⚠️ <strong>Chú ý:</strong> Kết quả cho thấy rủi ro trung bình. Bạn nên tham khảo ý kiến bác sĩ và theo dõi sức khỏe thường xuyên.
                        </p>
                    </div>
                    '''
                )
            else:  # high risk
                prediction_text = (
                    f'''
                    <div style="border-radius: 8px; border: 2px solid #E53935; background-color: #FFEBEE; padding: 20px; margin: 15px 0;">
                        <h6 style="color: #D32F2F; font-size: 20px; font-weight: bold; text-align: center;">{prediction_vn}</h6>
                        <p style="color: #B71C1C; font-weight: 500; font-size: 16px;">
                            🚨 <strong>Cảnh báo:</strong> Kết quả cho thấy rủi ro cao. Vui lòng đến cơ sở y tế ngay lập tức để được kiểm tra và tư vấn.
                        </p>
                    </div>
                    '''
                )

            # Giao diện cho phần chỉ số bất thường (nếu có)
            if abnormal_conditions:
                prediction_text += (
                    f'''
                    <div style="border-radius: 8px; border: 1.5px dashed #9C27B0; background-color: #F3E5F5; padding: 15px; margin-top: 10px;">
                        <p style="color: #4A148C; font-weight: bold; font-size: 16px; margin-bottom: 10px;">
                            🎯 <strong>Lưu ý thêm:</strong> Một số chỉ số của bạn đang bất thường:
                        </p>
                        <ul style="margin-left: 20px; color: #4A148C; font-weight: 500;">
                            {"".join([f"<li>{condition}</li>" for condition in abnormal_conditions])}
                        </ul>
                        <p style="color: #4A148C; font-weight: bold; font-size: 15px;">
                            Vui lòng đến bác sĩ để được tư vấn và kiểm tra sức khỏe kịp thời.
                        </p>
                    </div>
                    '''
                )

        except Exception as e:
            prediction_text = f"❌ Lỗi: {str(e)}"

    return render_template('index.html',
                           prediction_text=prediction_text,
                           age=age,
                           systolic=systolic,
                           diastolic=diastolic,
                           bs=bs,
                           bodytemp=bodytemp,
                           heartrate=heartrate,
                           temp_unit=temp_unit)

if __name__ == '__main__':
    app.run(debug=True)