from flask import Flask, render_template, request, redirect, url_for, jsonify, flash
import pandas as pd
import os
import logging

# Import the run_model function from model.py
from model import run_model


app = Flask(__name__)
app.secret_key = "somethingsecret"
app.config["UPLOAD_FOLDER"] = "uploads"
os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)

# Constants for CSV file paths
CSV_FILE = "Results/column_matching_results.csv"
UPDATED_CSV_FILE = "Results/updated_column_mapping_results.csv"
file_exists = os.path.exists(CSV_FILE)

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger()

@app.route("/", methods=["GET", "POST"])
def index():
    global CSV_FILE  # Move global declaration here
    show_results_table = os.path.exists(CSV_FILE)

    if request.method == "POST":
        # Handle file upload
        
        if "sourceFile" in request.files and "targetFile" in request.files:
            source_file = request.files["sourceFile"]
            target_file = request.files["targetFile"]

            if source_file.filename == "" or target_file.filename == "":
                flash("Error: Both files must be uploaded.", "error")
                return redirect(url_for("index"))

            source_file_path = os.path.join(app.config["UPLOAD_FOLDER"], source_file.filename)
            target_file_path = os.path.join(app.config["UPLOAD_FOLDER"], target_file.filename)

            source_file.save(source_file_path)
            target_file.save(target_file_path)

            flash("Files uploaded successfully.", "success")
            logger.info(f"Source file uploaded: {source_file_path}")
            logger.info(f"Target file uploaded: {target_file_path}")
            flash(f"Log: Source file uploaded: {source_file_path}", "info")
            flash(f"Log: Target file uploaded: {target_file_path}", "info")

            # Trigger the model
            flash("Model initiated.", "info")
            try:
                # Run the model with the uploaded files
                run_model(source_file_path, target_file_path)
                flash("Computing embeddings...", "info")
                flash("Results saved successfully.", "success")
            except Exception as e:
                flash(f"An error occurred while running the model: {e}", "error")
            show_results_table = os.path.exists(CSV_FILE)

            return redirect(url_for("index"))

        # Handle form data for column mapping updates
        try:
            updated_data = []
            for key, value in request.form.items():
                if key.startswith("action_"):
                    row_id = key.split("_")[1]
                    updated_data.append({
                        "Source Column": request.form[f"source_{row_id}"],
                        "Target Column": request.form[f"target_{row_id}"],
                        "Weighted Average Score": request.form[f"score_{row_id}"],
                        "Action": "checked" if value == "on" else "unchecked"
                    })
            updated_df = pd.DataFrame(updated_data)
            updated_df.to_csv(UPDATED_CSV_FILE, index=False)
            return redirect(url_for("index"))
        except Exception as e:
            return f"An error occurred while processing the form data: {e}"

    # Load mapping results from the CSV file
    try:
        results = []
        if os.path.exists(CSV_FILE):
            df = pd.read_csv(CSV_FILE)
            if not df.empty:
                one_to_one_mapping = (
                    df.sort_values(by="Weighted Average Score", ascending=False)
                    .drop_duplicates(subset=["Target Column"], keep="first")
                ).sort_values(by="Target Column")
                results = one_to_one_mapping.to_dict(orient="records")
            else:
                flash("The column matching results are empty.", "warning")
        else:
            flash("No results file found.", "warning")

        return render_template("index.html", show_results_table=show_results_table, results=results,file_exists = file_exists)
    except Exception as e:
        return f"An error occurred while reading or processing the CSV file: {e}"

@app.route("/get_probable_matches", methods=["GET"])
def get_probable_matches():
    try:
        source_column = request.args.get("source")
        if not source_column:
            return jsonify({"error": "Source column is required."}), 400

        if not os.path.exists(CSV_FILE):
            return jsonify({"error": f"File '{CSV_FILE}' does not exist."}), 500
        df = pd.read_csv(CSV_FILE)
        filtered_df = df[df["Source Column"] == source_column]
        top_matches = filtered_df.nlargest(5, "Weighted Average Score")
        matches = top_matches.to_dict(orient="records")
        return jsonify({"matches": matches}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/get_all_targets", methods=["GET"])
def get_all_targets():
    try:
        if not os.path.exists(CSV_FILE):
            return jsonify({"error": f"File '{CSV_FILE}' does not exist."}), 500

        df = pd.read_csv(CSV_FILE)
        targets = df["Target Column"].unique().tolist()
        return jsonify({"targets": targets}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/get_score_for_target", methods=["GET"])
def get_score_for_target():
    try:
        target = request.args.get("target")
        source = request.args.get("source")
        if not target or not source:
            return jsonify({"error": "Both source and target are required."}), 400

        if not os.path.exists(CSV_FILE):
            return jsonify({"error": f"File '{CSV_FILE}' does not exist."}), 500
        df = pd.read_csv(CSV_FILE)
        row = df[(df["Source Column"] == source) & (df["Target Column"] == target)]
        if row.empty:
            return jsonify({"error": "No score found for the given source and target."}), 404
        score = int(row["Weighted Average Score"].iloc[0])
        return jsonify({"score": score}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    try:
        if os.path.exists(UPDATED_CSV_FILE):
            os.remove(UPDATED_CSV_FILE)
        app.run(debug=True)
    except Exception as e:
        print(f"An error occurred while starting the Flask application: {e}")
