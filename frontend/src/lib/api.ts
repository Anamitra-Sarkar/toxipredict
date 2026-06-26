const API_BASE = process.env.NEXT_PUBLIC_API_URL || "";

async function getAuthToken(): Promise<string> {
  const { auth } = await import("./firebase");
  const user = auth.currentUser;
  if (!user) throw new Error("Not authenticated");
  return user.getIdToken();
}

async function request<T>(
  path: string,
  options: RequestInit = {}
): Promise<T> {
  const token = await getAuthToken();
  const res = await fetch(`${API_BASE}/api${path}`, {
    ...options,
    headers: {
      "Content-Type": "application/json",
      Authorization: `Bearer ${token}`,
      ...options.headers,
    },
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(err.detail || `Request failed: ${res.status}`);
  }
  return res.json();
}

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

export interface AgentResponse {
  target_assay: string;
  risk_level: string;
  toxicophore_assessment: string;
  biochemical_mechanism: string;
  structural_alerts_found: Array<{
    alert_name: string;
    details: string;
    matched_atoms: number[];
    high_attribution: boolean;
  }>;
  bioisosteric_replacements: string[];
  confidence: string;
}

export async function predict(smiles: string): Promise<PredictResponse> {
  return request<PredictResponse>("/predict", {
    method: "POST",
    body: JSON.stringify({ smiles }),
  });
}

export async function explain(
  smiles: string,
  targetTask: number = 0
): Promise<ExplainResponse> {
  return request<ExplainResponse>("/explain", {
    method: "POST",
    body: JSON.stringify({ smiles, target_task: targetTask }),
  });
}

export async function agentAnalysis(
  smiles: string,
  predictions: AssayPrediction[],
  shapAttributions: number[],
  highImpactAtoms: number[],
  targetAssay: string
): Promise<AgentResponse> {
  return request<AgentResponse>("/agent", {
    method: "POST",
    body: JSON.stringify({
      smiles,
      predictions,
      shap_attributions: shapAttributions,
      high_impact_atoms: highImpactAtoms,
      target_assay: targetAssay,
    }),
  });
}
