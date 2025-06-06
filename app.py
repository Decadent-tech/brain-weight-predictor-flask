import flask
from flask import Flask, request, render_template
import joblib
import pandas as pd
from datetime import datetime
from pathlib import Path
import matplotlib.pyplot as plt
from flask import send_file

# Initialize Flask app
app = Flask(__name__)

# Load the trained model
model = joblib.load("model.pkl")

# Function to log predictions
def log_prediction(head_size, age, gender, notes, prediction):
    log_df = pd.DataFrame([[datetime.now(), head_size, age, gender, notes, prediction]],
                          columns=["timestamp", "head_size", "age", "gender", "notes", "predicted_brain_weight"])
    if not Path("prediction_logs.csv").exists():
        log_df.to_csv("prediction_logs.csv", mode='a', header=True, index=False)
    else:
        log_df.to_csv("prediction_logs.csv", mode='a', header=False, index=False)

# Route for form and prediction
@app.route("/", methods=["GET", "POST"])
def predict():
    prediction = None
    error = None
    if request.method == "POST":
        try:
            head_size = float(request.form["head_size"])
            if head_size <= 0:
                raise ValueError("Head size must be positive.")

            age = request.form.get("age")
            gender = request.form.get("gender")
            notes = request.form.get("notes")

            prediction = model.predict([[head_size]])[0]
            log_prediction(head_size, age, gender, notes, prediction)
        except ValueError as ve:
            return render_template("form.html", prediction=None, error=str(ve))
    return render_template("form.html", prediction=prediction, error=error)
@app.route("/logs")
def view_logs():
    try:
        df = pd.read_csv("prediction_logs.csv")

        gender_filter = request.args.get("gender")
        date_filter = request.args.get("date")

        if gender_filter:
            df = df[df["gender"].str.lower() == gender_filter.lower()]
        if date_filter:
            df["timestamp"] = pd.to_datetime(df["timestamp"])
            df = df[df["timestamp"].dt.date == pd.to_datetime(date_filter).date()]

        avg = df["predicted_brain_weight"].mean() if not df.empty else 0
        table_html = df.to_html(classes='table table-striped', index=False)
        return render_template("logs.html", tables=[table_html], avg=avg, total=len(df), request=request)
    except Exception as e:
        return f"Error reading logs: {e}"
@app.route("/plot")
def plot_graph():
    try:
        df = pd.read_csv("prediction_logs.csv")
        df.columns = ["timestamp", "head_size", "age", "gender", "notes", "predicted_brain_weight"]

        # Only keep numeric values
        df = df.dropna(subset=["head_size", "predicted_brain_weight"])

        plt.figure(figsize=(8,5))
        plt.scatter(df["head_size"], df["predicted_brain_weight"], c="purple", edgecolors="k")
        plt.title("Head Size vs Predicted Brain Weight")
        plt.xlabel("Head Size (cm³)")
        plt.ylabel("Predicted Brain Weight (g)")
        plt.grid(True)

        # Save plot
        plot_path = "static/plot.png"
        plt.savefig(plot_path)
        plt.close()
        return render_template("plot.html", plot_url=plot_path)

    except Exception as e:
        return f"Error generating plot: {e}"


@app.route("/download_csv")
def download_csv():
    file_path = "prediction_logs.csv"
    return send_file(file_path, as_attachment=True)
@app.route("/about")
def about():
    return render_template("about.html")
if __name__ == "__main__":
    app.run(debug=True)
