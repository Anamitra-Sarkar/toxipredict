import json
import os
from groq import Groq
from pydantic import BaseModel, Field, TypeAdapter
from typing import List, Optional
from config import GROQ_API_KEY, GROQ_MODEL


class ToxicologyReport(BaseModel):
    target_assay: str = Field(description="The target assay endpoint evaluated.")
    risk_level: str = Field(description="Overall risk level: Low, Moderate, High, or Critical.")
    toxicophore_assessment: str = Field(description="Analysis of which structural features contribute to predicted toxicity.")
    biochemical_mechanism: str = Field(description="Explanation of the biochemical mechanism linking structure to endpoint.")
    structural_alerts_found: List[dict] = Field(description="List of structural alerts matched and their relevance.")
    bioisosteric_replacements: List[str] = Field(description="Suggested safer bioisosteric replacements.")
    confidence: str = Field(description="Confidence in this assessment based on model prediction and alert matching.")


class ToxicologistAgent:
    def __init__(self):
        self.client = Groq(api_key=GROQ_API_KEY)
        self.model = GROQ_MODEL
        self.schema_adapter = TypeAdapter(ToxicologyReport)
        self.json_schema = self._build_strict_schema()

    def _build_strict_schema(self) -> dict:
        raw = self.schema_adapter.json_schema()
        raw["additionalProperties"] = False
        if "properties" in raw:
            for prop_val in raw["properties"].values():
                if isinstance(prop_val, dict) and "properties" in prop_val:
                    prop_val["additionalProperties"] = False
        if "$defs" in raw:
            for def_val in raw["$defs"].values():
                if isinstance(def_val, dict):
                    def_val["additionalProperties"] = False
        return raw

    def generate_report(self, smiles: str, target_assay: str, predictions: list,
                        shap_attributions: list, high_attr_indices: list,
                        alerts_found: list) -> ToxicologyReport:
        prompt = f"""You are an expert computational toxicologist. Analyze the following compound's toxicological profile.

COMPOUND SMILES: {smiles}
TARGET ASSAY: {target_assay} ({predictions[0].get('target_class', 'Unknown')})
PREDICTED PROBABILITY: {predictions[0].get('probability', 0.5):.4f}
PREDICTED CLASS: {predictions[0].get('predicted_class', 'Unknown')}

SHAP HIGH-ATTRIBUTION ATOM INDICES: {high_attr_indices}
FULL SHAP VECTOR: {shap_attributions}

STRUCTURAL ALERTS DETECTED: {json.dumps(alerts_found, indent=2)}

Based on this data:
1. Identify which specific structural fragments (toxicophores) are driving the {predictions[0].get('predicted_class', 'Unknown')} prediction for {target_assay}.
2. Explain the biochemical mechanism linking these fragments to the assay endpoint.
3. Suggest 2-3 specific bioisosteric replacements that could mitigate toxicity while preserving core scaffold.
4. Assess overall risk level.
5. Provide confidence in the assessment.

Output valid JSON with these fields:
- target_assay (string)
- risk_level (string: Low/Moderate/High/Critical)
- toxicophore_assessment (string)
- biochemical_mechanism (string)
- structural_alerts_found (array of objects with alert_name and details)
- bioisosteric_replacements (array of strings)
- confidence (string)

Output ONLY the JSON, no other text.
"""
        chat_completion = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "system",
                    "content": "You are a professional computational toxicologist. Output reports strictly adhering to the provided JSON schema."
                },
                {"role": "user", "content": prompt}
            ],
            response_format={"type": "json_object"},
            temperature=0.1,
        )

        raw_json = chat_completion.choices[0].message.content
        report = self.schema_adapter.validate_json(raw_json)
        return report
