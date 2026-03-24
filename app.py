from flask import Flask, render_template, request
import joblib
import pandas as pd

app = Flask(__name__)

model    = joblib.load("knn_heart_model.pkl")
scaler   = joblib.load("heart_scaler.pkl")
expected = joblib.load("heart_columns.pkl")

@app.route("/")
def home():
    return render_template("index.html", result=None)

@app.route("/predict", methods=["POST"])
def predict():
    try:
        age             = int(request.form.get("age"))
        sex             = request.form.get("sex")
        chest_pain      = request.form.get("chest_pain")
        resting_bp      = int(request.form.get("resting_bp"))
        cholesterol     = int(request.form.get("cholesterol"))
        fasting_bs      = int(request.form.get("fasting_bs"))
        resting_ecg     = request.form.get("resting_ecg")
        max_hr          = int(request.form.get("max_hr"))
        exercise_angina = request.form.get("exercise_angina")
        oldpeak         = float(request.form.get("oldpeak"))
        st_slope        = request.form.get("st_slope")

        raw = {
            'Age': age, 'RestingBP': resting_bp,
            'Cholesterol': cholesterol, 'FastingBS': fasting_bs,
            'MaxHR': max_hr, 'Oldpeak': oldpeak,
            'Sex_' + sex: 1,
            'ChestPainType_' + chest_pain: 1,
            'RestingECG_' + resting_ecg: 1,
            'ExerciseAngina_' + exercise_angina: 1,
            'ST_Slope_' + st_slope: 1
        }

        df = pd.DataFrame([raw])
        for col in expected:
            if col not in df.columns:
                df[col] = 0
        df = df[expected]

        scaled     = scaler.transform(df)
        prediction = model.predict(scaled)[0]
        result     = "high" if prediction == 1 else "low"

        return render_template("index.html",
            result=result,
            age=age, sex=sex, chest_pain=chest_pain,
            resting_bp=resting_bp, cholesterol=cholesterol,
            fasting_bs=fasting_bs, resting_ecg=resting_ecg,
            max_hr=max_hr, exercise_angina=exercise_angina,
            oldpeak=oldpeak, st_slope=st_slope
        )

    except Exception as e:
        print(f"Error: {e}")
        return render_template("index.html", result=None, error=str(e))

if __name__ == "__main__":
    app.run(debug=False, port=5000)
