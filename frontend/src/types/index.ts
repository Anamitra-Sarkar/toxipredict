export interface AssayPrediction {
  assay: string;
  target_class: string;
  probability: number;
  predicted_class: "Toxic" | "Non-Toxic";
}

export interface PredictResponse {
  analysis_id: string;
  smiles: string;
  predictions: AssayPrediction[];
  uncertainty_weights: number[];
  molecule_num_atoms: number;
}

export interface ExplainResponse {
  target_task: string;
  shap_values: number[];
  attribution_map_base64: string;
  high_impact_atoms: number[];
}

export interface StructuralAlert {
  alert_name: string;
  details: string;
  matched_atoms: number[];
  high_attribution: boolean;
}

export interface AgentResponse {
  target_assay: string;
  risk_level: string;
  toxicophore_assessment: string;
  biochemical_mechanism: string;
  structural_alerts_found: StructuralAlert[];
  bioisosteric_replacements: string[];
  confidence: string;
}
