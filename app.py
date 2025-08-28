import os
import pandas as pd
from flask import Flask, render_template, request, jsonify

# Create necessary folders if not exist
if not os.path.exists("templates"):
    os.makedirs("templates")

if not os.path.exists("bias_strategies.csv"):
    # Create CSV file with bias strategies
    csv_content = """Bias_Type,Detection_Method,Mitigation_Strategy
Gender Bias,Compare prediction rates across genders; fairness metrics like Equal Opportunity,Re-sampling underrepresented gender data; fairness constraints; adversarial debiasing
Class Imbalance,Confusion matrix per class; imbalance ratio; precision-recall curves,SMOTE (Synthetic Minority Oversampling Technique); class weighting; focal loss
Racial Bias,Demographic parity; disparate impact test; subgroup error analysis,Reweighting samples; adversarial training; fairness-aware preprocessing
Age Bias,Error analysis per age group; calibration curves by age,Representation learning; stratified sampling by age groups; fairness post-processing
Selection Bias,Analyze sampling pipeline; compare training vs. test distributions,Domain adaptation; stratified sampling; importance weighting
Measurement Bias,Check for noisy or skewed labels; annotation consistency tests,Improve data labeling; active learning; noise-robust algorithms
Label Bias,Compare human labels vs. ground truth; inter-rater reliability,Consensus labeling; crowdsourcing validation; calibration of annotators
Cultural Bias,Performance testing across cultural groups; subgroup error rates,Include culturally diverse data; fairness constraints; post-hoc corrections
Confirmation Bias,Audit feature engineering choices; cross-validation on alternative hypotheses,Blind evaluation; randomized controlled trials; model interpretation tools
Automation Bias,Analyze decision overrides vs. model output; error audits,Human-in-the-loop design; uncertainty estimation; override mechanisms
Historical Bias,Review dataset origin; test against current distribution,Update datasets regularly; reweight outdated samples; causal inference adjustments
Sampling Bias,Check sample distributions vs. population; chi-square goodness-of-fit test,Stratified random sampling; oversampling rare subgroups
Reporting Bias,Audit feature frequency; compare data coverage across sources,Encourage balanced reporting; data augmentation; source diversification
Overfitting Bias,Training vs. validation performance gap; cross-validation,Regularization (L1/L2); dropout; early stopping
Underfitting Bias,High bias error; poor training accuracy; residual error analysis,Increase model complexity; feature engineering; reduce constraints
Popularity Bias,Check distribution of recommendations; exposure fairness metrics,Re-ranking algorithms; diversity constraints in recommender systems
Temporal Bias,Performance drift analysis over time; rolling window evaluation,Time-series cross-validation; model retraining; domain adaptation
Geographical Bias,Error breakdown by location; subgroup fairness metrics,Collect region-specific data; transfer learning; geospatial weighting
Data Representation Bias,Feature distribution comparisons; fairness metrics per attribute,Feature rebalancing; fairness regularization; embedding debiasing
Evaluation Bias,Compare benchmark vs. real-world datasets; fairness audit,Build representative test sets; multiple benchmark evaluations"""
    with open("bias_strategies.csv", "w") as f:
        f.write(csv_content)

if not os.path.exists("templates/index.html"):
    # Create HTML file
    html_content = """<!DOCTYPE html>
<html>
<head>
    <title>Bias in ML Models</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; background: #f8f9fa; }
        h2 { text-align: center; color: #333; }
        form { text-align: center; margin-bottom: 20px; }
        input[type="text"] { padding: 8px; width: 70%; border: 1px solid #ccc; border-radius: 5px; }
        button { padding: 8px 15px; border: none; background: #007bff; color: white; border-radius: 5px; cursor: pointer; }
        button:hover { background: #0056b3; }
        table { width: 100%; border-collapse: collapse; margin-top: 20px; background: white; }
        th, td { padding: 10px; border: 1px solid #ddd; text-align: left; }
        th { background: #007bff; color: white; }
        tr:nth-child(even) { background: #f2f2f2; }
    </style>
</head>
<body>
    <h2>Bias Detection & Reduction in Machine Learning Models</h2>
    <form id="biasForm">
        <input type="text" name="bias_type" placeholder="Enter bias type (e.g., gender, class imbalance)" required>
        <button type="submit">Check</button>
    </form>
    <div id="results"></div>

    <script>
        document.getElementById("biasForm").onsubmit = async (e) => {
            e.preventDefault();
            const formData = new FormData(e.target);
            const res = await fetch("/check", { method: "POST", body: formData });
            const data = await res.json();

            let html = `
                <table>
                    <tr>
                        <th>Bias Type</th>
                        <th>Detection Method</th>
                        <th>Mitigation Strategy</th>
                    </tr>
            `;
            data.forEach(row => {
                html += `
                    <tr>
                        <td>${row.Bias_Type}</td>
                        <td>${row.Detection_Method}</td>
                        <td>${row.Mitigation_Strategy}</td>
                    </tr>
                `;
            });
            html += "</table>";
            document.getElementById("results").innerHTML = html;
        }
    </script>
</body>
</html>"""
    with open("templates/index.html", "w") as f:
        f.write(html_content)

# Flask app
app = Flask(__name__)
df = pd.read_csv("bias_strategies.csv")

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/check", methods=["POST"])
def check_bias():
    user_input = request.form["bias_type"].lower()
    matches = df[df["Bias_Type"].str.lower().str.contains(user_input, na=False)]

    if matches.empty:
        results = [{"Bias_Type": "Not found",
                    "Detection_Method": "N/A",
                    "Mitigation_Strategy": "Try another keyword"}]
    else:
        results = matches.to_dict(orient="records")

    return jsonify(results)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)