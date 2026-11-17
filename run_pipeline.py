from pathlib import Path
import json
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score
from sklearn.inspection import permutation_importance
from sklearn.model_selection import train_test_split
from simulator import generate_process_data

ROOT=Path(__file__).resolve().parents[1]; OUT=ROOT/"outputs"; OUT.mkdir(exist_ok=True)
df=generate_process_data(); df.to_csv(OUT/"synthetic_process_data.csv",index=False)
X,y=df.drop(columns="defect"),df.defect
# Domain-informed features expose process deviations and nonlinear risk mechanisms.
X["moisture_excess"]=(X["moisture_pct"]-.04).clip(lower=0)
X["viscosity_deviation"]=(X["slurry_viscosity_cp"]-3200).abs()
X["coating_temp_deviation"]=(X["coating_temp_c"]-105).abs()
X["thickness_deviation"]=(X["coating_thickness_um"]-110).abs()
X["pressure_deviation"]=(X["calender_pressure_mpa"]-85).abs()
X["formation_temp_deviation"]=(X["formation_temp_c"]-27).abs()
X["formation_current_deviation"]=(X["formation_current_c"]-1).abs()
cat=["stage"]; num=[c for c in X if c not in cat]
prep=ColumnTransformer([("num",Pipeline([("imp",SimpleImputer(strategy="median")),("scale",StandardScaler())]),num),
                        ("cat",OneHotEncoder(handle_unknown="ignore"),cat)])
model=Pipeline([("prep",prep),("rf",RandomForestClassifier(n_estimators=250,max_depth=12,
    min_samples_leaf=5,class_weight="balanced",random_state=42,n_jobs=-1))])
Xtr,Xte,ytr,yte=train_test_split(X,y,test_size=.25,random_state=42,stratify=y)
model.fit(Xtr,ytr); prob=model.predict_proba(Xte)[:,1]; pred=(prob>=.5).astype(int)
metrics={"roc_auc":float(roc_auc_score(yte,prob)),"precision":float(precision_score(yte,pred,zero_division=0)),
         "recall":float(recall_score(yte,pred,zero_division=0)),"f1":float(f1_score(yte,pred,zero_division=0)),
         "test_rows":len(yte),"synthetic_defect_rate":float(y.mean())}
(OUT/"model_metrics.json").write_text(json.dumps(metrics,indent=2))
pi=permutation_importance(model,Xte,yte,n_repeats=4,random_state=42,scoring="roc_auc",n_jobs=-1)
pd.DataFrame({"feature":X.columns,"importance_mean":pi.importances_mean}).sort_values(
    "importance_mean",ascending=False).to_csv(OUT/"permutation_importance.csv",index=False)
df.groupby("stage").agg(records=("defect","size"),defect_rate=("defect","mean"),
    energy_index=("energy_index","mean"),throughput_index=("throughput_index","mean")).to_csv(OUT/"stage_summary.csv")
# Candidate settings are compared on the same holdout sample; objective penalises risk and energy.
base=Xte.sample(min(400,len(Xte)),random_state=7).copy()
rows=[]
for temp in [95,100,105,110,115]:
 for thick in [102,106,110,114,118]:
  for pressure in [75,80,85,90,95]:
   z=base.copy(); z["coating_temp_c"]=temp; z["coating_thickness_um"]=thick; z["calender_pressure_mpa"]=pressure
   risk=model.predict_proba(z)[:,1].mean(); energy=.8+.004*temp+.003*pressure
   rows.append((temp,thick,pressure,risk,energy,risk+.025*energy+.002*abs(thick-110)))
opt=pd.DataFrame(rows,columns=["temp_c","thickness_um","pressure_mpa","predicted_risk","energy_index","objective"])
opt.sort_values("objective").to_csv(OUT/"parameter_search.csv",index=False)
print(json.dumps(metrics,indent=2)); print("Saved:",OUT)
