# Battery Manufacturing Digital Twin & Defect Optimisation

Models five cell-manufacturing stages and builds a defect-risk classifier, feature-importance report and constrained parameter-search experiment.

Run:
```bash
pip install -r requirements.txt
python src/run_pipeline.py
```
Outputs: synthetic data, metrics, stage summary, permutation importance and candidate parameter search.

**Limitations:** The dataset is synthetic and the defect rules are encoded in the simulator. Optimised settings are model-based candidates, not production recommendations. Replace the simulator with authorised process data and validate through controlled engineering trials before making operational claims.
