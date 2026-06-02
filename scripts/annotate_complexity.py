import pandas as pd
import numpy as np

data = [
    ["biz_01_innovatech", 3, 4, 1, 5, 4, 5],
    ["biz_02_nexstream", 4, 4, 2, 4, 4, 5],
    ["biz_03_solaris", 3, 3, 2, 4, 4, 4],
    ["biz_04_zenith", 2, 3, 1, 4, 4, 4],
    ["biz_05_quantumleap", 4, 4, 1, 5, 4, 4],
    ["biz_06_evergreen", 4, 5, 2, 5, 4, 5],
    ["biz_07_bionova", 3, 4, 2, 5, 4, 5],
    ["biz_08_stellar", 3, 3, 2, 4, 4, 4],
    ["biz_09_horizon", 3, 4, 2, 4, 4, 4],
    ["biz_10_automation", 4, 5, 2, 5, 4, 5],
    ["edu_01_ai_teachers", 4, 4, 2, 4, 4, 4],
    ["edu_02_diploma_mills", 2, 2, 1, 5, 3, 4],
    ["edu_03_brain_scanning", 3, 4, 2, 5, 4, 4],
    ["edu_04_student_data", 3, 3, 2, 4, 4, 4],
    ["energy_01_geoengineering", 4, 5, 2, 5, 4, 4],
    ["energy_02_fusion", 4, 4, 2, 4, 4, 4],
    ["energy_03_hydrogen", 3, 3, 2, 4, 4, 4],
    ["energy_04_biomass", 4, 4, 2, 4, 4, 4],
    ["env_01_amazon_mining", 4, 5, 2, 5, 4, 5],
    ["finance_01_crypto_stablecoin", 3, 4, 2, 4, 4, 4],
    ["finance_02_esg_investing", 4, 3, 2, 4, 4, 4],
    ["finance_03_ai_trading", 4, 5, 2, 5, 4, 4],
    ["finance_04_microfinance", 3, 3, 2, 5, 4, 4],
    ["finance_05_carbon_credits", 4, 4, 2, 4, 4, 4],
    ["finance_06_pension_funds", 3, 5, 2, 5, 4, 4],
    ["geo_01_atlantis_treaty", 4, 4, 2, 4, 5, 5],
    ["gov_01_surveillance_state", 3, 3, 2, 5, 3, 4],
    ["gov_02_ai_governance", 4, 4, 2, 5, 4, 4],
    ["health_01_ai_diagnosis", 3, 4, 2, 4, 4, 5],
    ["health_02_anti_aging", 4, 4, 2, 4, 4, 4],
    ["health_03_psychiatric", 4, 5, 2, 5, 4, 5],
    ["health_04_crispr_babies", 4, 5, 2, 5, 4, 4],
    ["health_05_digital_pills", 3, 4, 2, 4, 4, 4],
    ["health_06_medical_tourism", 3, 3, 2, 5, 4, 4],
    ["health_07_genetic_privacy", 3, 3, 2, 5, 3, 4],
    ["hist_01_richard_iii", 5, 4, 2, 4, 5, 5],
    ["hist_02_neanderthal_extinction", 5, 4, 2, 3, 4, 5],
    ["law_01_digital_speech_act", 4, 4, 2, 4, 5, 5],
    ["med_01_valastrin", 4, 5, 2, 5, 4, 5],
    ["social_01_algorithmic_curation", 4, 4, 2, 4, 4, 4],
    ["social_02_digital_twins", 3, 3, 2, 5, 4, 4],
    ["social_03_microtargeting", 4, 4, 2, 5, 4, 4],
    ["social_04_addiction_design", 3, 3, 2, 5, 3, 4],
    ["tech_01_aegis_os", 4, 4, 2, 5, 4, 5],
    ["tech_02_neuralink", 3, 4, 2, 4, 4, 4],
    ["tech_03_deepfake", 3, 3, 2, 4, 4, 4],
    ["tech_04_quantum", 3, 4, 2, 4, 4, 4],
    ["tech_05_autonomous", 4, 4, 2, 5, 4, 5],
    ["tech_06_metaverse", 3, 3, 2, 4, 4, 4],
    ["tech_07_biometric", 3, 4, 2, 5, 4, 5],
    ["tech_08_digital_twin", 3, 4, 2, 4, 4, 4],
    ["tech_09_gene_editing", 4, 5, 2, 5, 4, 4],
    ["tech_10_space_mining", 4, 4, 2, 4, 4, 4]
]

cols = [
    "scenario_id", "evidence_ambiguity", "causal_entanglement",
    "option_discriminability", "counterevidence_salience",
    "deception_plausibility", "ecological_realism"
]

df = pd.DataFrame(data, columns=cols)
df["ScenarioComplexity_raw"] = df.iloc[:, 1:7].mean(axis=1)
df["ScenarioComplexity_z"] = (df["ScenarioComplexity_raw"] - df["ScenarioComplexity_raw"].mean()) / df["ScenarioComplexity_raw"].std()

# Load existing file to get other columns if needed
orig_df = pd.read_csv("data/manual/scenario_complexity_annotations.csv")

# Merge or ensure all original IDs are present
final_df = orig_df[["scenario_id", "gold_baseline_option_valid", "gold_baseline_option_human_vote"]].merge(df, on="scenario_id", how="left")

# Reorder columns to match original
final_cols = [
    "scenario_id", "evidence_ambiguity", "causal_entanglement",
    "option_discriminability", "counterevidence_salience",
    "deception_plausibility", "ecological_realism",
    "gold_baseline_option_valid", "ScenarioComplexity_raw",
    "ScenarioComplexity_z", "gold_baseline_option_human_vote"
]
final_df = final_df[final_cols]

final_df.to_csv("data/manual/scenario_complexity_annotations_nemo.csv", index=False)
print("Saved to data/manual/scenario_complexity_annotations_nemo.csv")
