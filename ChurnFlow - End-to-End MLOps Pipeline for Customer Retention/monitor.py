from flask import Flask, request, jsonify, render_template_string
import pandas as pd
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset
import os
import retrain as rt

app = Flask(__name__)

# Local Development & Containerized Paths
BASE = "/app" if os.path.exists("/app") else os.getcwd()
DATA_DIR = os.path.join(BASE, "data", "processed")
LOGS_DIR = os.path.join(BASE, "logs")
MODELS_DIR = os.path.join(BASE, "models")
REPORT_HTML = os.path.join(LOGS_DIR, "monitor_report_latest.html")
REF_CSV = os.path.join(DATA_DIR, "ref_data.csv")
CUR_CSV = os.path.join(DATA_DIR, "current_batch.csv")  # rolling window store

os.makedirs(LOGS_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)

@app.route("/monitor", methods=["POST"])
def monitor():
    payload = request.get_json(silent=True) or {}
    records = payload.get("records", [])
    if not records:
        return jsonify({"ok": False, "error": "no records"}), 400

    cur = pd.DataFrame(records)
    if not os.path.exists(REF_CSV):
        return jsonify({"ok": False, "error": f"missing ref: {REF_CSV}"}), 500

    ref = pd.read_csv(REF_CSV)

    # align columns present in both
    common = [c for c in ref.columns if c in cur.columns]
    if not common:
        return jsonify({"ok": False, "error": "no common columns"}), 400
    ref = ref[common]
    cur = cur[common]

    # run evidently
    report = Report(metrics=[DataDriftPreset()])
    report.run(reference_data=ref, current_data=cur)
    report.save_html(REPORT_HTML)

    # simple decision rule
    summary = report.as_dict()
    dataset_drift = summary["metrics"][0]["result"]["dataset_drift"]

    # optional: append current to rolling CSV for visibility
    if os.path.exists(CUR_CSV):
        cur.to_csv(CUR_CSV, mode="a", header=False, index=False)
    else:
        cur.to_csv(CUR_CSV, index=False)

    if dataset_drift:
        # retrain requires new_data.csv with target column present
        try:
            rt.retrain_model()  # your function inside retrain.py
        except Exception as e:
            return jsonify({"ok": True, "drift": True, "retrain": False, "error": str(e)}), 200
        return jsonify({"ok": True, "drift": True, "retrain": True}), 200

    return jsonify({"ok": True, "drift": False}), 200

@app.get("/view_report")
def view_report():
    if not os.path.exists(REPORT_HTML):
        return "No report generated yet.", 404
    with open(REPORT_HTML, "r", encoding="utf-8") as f:
        return render_template_string(f.read())

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8001)
