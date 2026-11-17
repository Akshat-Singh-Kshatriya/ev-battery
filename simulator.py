import numpy as np
import pandas as pd

STAGES = ["mixing", "coating", "calendering", "assembly", "formation"]

def generate_process_data(n=20000, seed=42):
    rng = np.random.default_rng(seed)
    stage = rng.choice(STAGES, n)
    moisture = np.clip(rng.normal(.035, .012, n), .005, .09)
    viscosity = rng.normal(3200, 420, n)
    coating_temp = rng.normal(105, 9, n)
    thickness = rng.normal(110, 7, n)
    pressure = rng.normal(85, 8, n)
    alignment = np.abs(rng.normal(0, .35, n))
    formation_temp = rng.normal(27, 4, n)
    formation_current = rng.normal(1, .18, n)
    energy = .8 + .004*coating_temp + .003*pressure + rng.normal(0,.08,n)
    throughput = np.clip(100-.12*np.abs(thickness-110)-1.5*alignment+rng.normal(0,2,n),70,105)

    # Explicit nonlinear defect mechanism; engineered deviation features are also
    # included in the pipeline so the model can learn these relationships.
    stage_risk = pd.Series(stage).map({"mixing":.25,"coating":.55,"calendering":.35,
                                       "assembly":.20,"formation":.45}).to_numpy()
    logit = (-4.8 + stage_risk
             + 65*np.maximum(moisture-.04,0)
             + 0.000003*(viscosity-3200)**2
             + .11*np.abs(coating_temp-105)
             + .14*np.abs(thickness-110)
             + .07*np.abs(pressure-85)
             + 2.0*alignment
             + .10*np.abs(formation_temp-27)
             + 1.8*np.abs(formation_current-1))
    prob = 1/(1+np.exp(-logit))
    defect = rng.binomial(1, np.clip(prob,.01,.9))
    return pd.DataFrame({"stage":stage,"moisture_pct":moisture,"slurry_viscosity_cp":viscosity,
        "coating_temp_c":coating_temp,"coating_thickness_um":thickness,
        "calender_pressure_mpa":pressure,"alignment_error_mm":alignment,
        "formation_temp_c":formation_temp,"formation_current_c":formation_current,
        "energy_index":energy,"throughput_index":throughput,"defect":defect})
