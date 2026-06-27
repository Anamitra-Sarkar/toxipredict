"use client";

import { useState, useEffect, useCallback } from "react";
import { useRouter } from "next/navigation";
import { auth } from "@/lib/firebase";
import { predict, explain, agentAnalysis } from "@/lib/api";
import type { AssayPrediction, ExplainResponse, AgentResponse } from "@/types";
import { Header } from "@/components/layout/Header";
import { SmilesInput } from "@/components/prediction/SmilesInput";
import { PredictionGrid } from "@/components/prediction/PredictionGrid";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Loader2, Brain, AlertTriangle, FlaskConical } from "lucide-react";

export default function DashboardPage() {
  const router = useRouter();
  const [user, setUser] = useState<object | null>(null);
  const [loading, setLoading] = useState(false);
  const [predictions, setPredictions] = useState<AssayPrediction[]>([]);
  const [selectedAssay, setSelectedAssay] = useState<number | null>(null);
  const [smiles, setSmiles] = useState("");
  const [analysisId, setAnalysisId] = useState<string | null>(null);
  const [explainResult, setExplainResult] = useState<ExplainResponse | null>(null);
  const [agentResult, setAgentResult] = useState<AgentResponse | null>(null);
  const [explainLoading, setExplainLoading] = useState(false);
  const [agentLoading, setAgentLoading] = useState(false);

  useEffect(() => {
    const unsubscribe = auth.onAuthStateChanged((u) => {
      if (!u) {
        router.push("/login");
      } else {
        setUser(u);
      }
    });
    return unsubscribe;
  }, [router]);

  const handlePredict = useCallback(async (smilesStr: string) => {
    setLoading(true);
    setSmiles(smilesStr);
    setPredictions([]);
    setSelectedAssay(null);
    setExplainResult(null);
    setAgentResult(null);
    try {
      const result = await predict(smilesStr);
      setPredictions(result.predictions);
      setAnalysisId(result.analysis_id);
    } catch (err) {
      console.error("Prediction failed:", err);
    } finally {
      setLoading(false);
    }
  }, []);

  const handleExplain = useCallback(async () => {
    if (!smiles || selectedAssay === null) return;
    setExplainLoading(true);
    setExplainResult(null);
    try {
      const result = await explain(smiles, selectedAssay);
      setExplainResult(result);
    } catch (err) {
      console.error("Explain failed:", err);
    } finally {
      setExplainLoading(false);
    }
  }, [smiles, selectedAssay]);

  const handleAgent = useCallback(async () => {
    if (!smiles || !explainResult || selectedAssay === null) return;
    setAgentLoading(true);
    setAgentResult(null);
    try {
      const result = await agentAnalysis(
        smiles,
        predictions,
        explainResult.shap_values,
        explainResult.high_impact_atoms,
        predictions[selectedAssay]?.assay || "",
      );
      setAgentResult(result);
    } catch (err) {
      console.error("Agent analysis failed:", err);
    } finally {
      setAgentLoading(false);
    }
  }, [smiles, explainResult, predictions, selectedAssay]);

  if (!user) return null;

  const selectedPred = selectedAssay !== null ? predictions[selectedAssay] : null;

  return (
    <div className="min-h-screen bg-background">
      <Header />
      <main className="mx-auto max-w-7xl px-4 py-6 sm:px-6">
        <div className="mb-6">
          <h1 className="text-lg font-bold">Toxicological Profiling</h1>
          <p className="text-sm text-muted-foreground">
            Enter a SMILES string to predict toxicity across 10 biological endpoints
          </p>
        </div>

        <div className="grid gap-6 lg:grid-cols-3">
          <div className="space-y-6 lg:col-span-1">
            <SmilesInput onPredict={handlePredict} loading={loading} />

            {predictions.length > 0 && selectedAssay !== null && (
              <Card>
                <CardHeader>
                  <div className="flex items-center gap-2">
                    <FlaskConical className="h-4 w-4 text-primary" />
                    <h3 className="text-sm font-semibold">
                      {predictions[selectedAssay]?.assay}
                    </h3>
                  </div>
                </CardHeader>
                <CardContent className="space-y-3">
                  <Button
                    className="w-full"
                    size="sm"
                    onClick={handleExplain}
                    disabled={explainLoading}
                  >
                    {explainLoading ? (
                      <Loader2 className="mr-1.5 h-3.5 w-3.5 animate-spin" />
                    ) : null}
                    Generate SHAP Explanation
                  </Button>

                  {explainResult && (
                    <Button
                      className="w-full"
                      size="sm"
                      variant="secondary"
                      onClick={handleAgent}
                      disabled={agentLoading}
                    >
                      {agentLoading ? (
                        <Loader2 className="mr-1.5 h-3.5 w-3.5 animate-spin" />
                      ) : (
                        <Brain className="mr-1.5 h-3.5 w-3.5" />
                      )}
                      AI Toxicologist Analysis
                    </Button>
                  )}
                </CardContent>
              </Card>
            )}
          </div>

          <div className="lg:col-span-2">
            {loading && (
              <div className="flex items-center justify-center py-20">
                <Loader2 className="h-6 w-6 animate-spin text-primary" />
                <span className="ml-2 text-sm text-muted-foreground">
                  Running GNN inference...
                </span>
              </div>
            )}

            {predictions.length > 0 && (
              <PredictionGrid
                predictions={predictions}
                selectedIndex={selectedAssay}
                onSelectAssay={setSelectedAssay}
              />
            )}

            {!loading && predictions.length === 0 && (
              <div className="flex flex-col items-center justify-center rounded-xl border border-dashed border-border py-20">
                <FlaskConical className="mb-3 h-8 w-8 text-muted-foreground" />
                <p className="text-sm text-muted-foreground">
                  Enter a SMILES string and click Predict
                </p>
              </div>
            )}
          </div>
        </div>

        {explainResult && (
          <Card className="mt-6">
            <CardHeader>
              <h3 className="text-sm font-semibold">
                SHAP Attribution Map — {explainResult.target_task}
              </h3>
              <p className="text-xs text-muted-foreground">
                Atom-level game-theoretic contributions to the prediction. Red = promotes toxicity, Blue = reduces toxicity.
              </p>
            </CardHeader>
            <CardContent>
              {explainResult.attribution_map_base64 ? (
                <div className="flex justify-center">
                  <img
                    src={`data:image/png;base64,${explainResult.attribution_map_base64}`}
                    alt="SHAP Attribution Map"
                    className="max-h-80 rounded-lg"
                  />
                </div>
              ) : (
                <div className="flex items-center gap-2 text-sm text-muted-foreground">
                  <AlertTriangle className="h-4 w-4" />
                  Visualization unavailable
                </div>
              )}
              <div className="mt-4 flex flex-wrap gap-2">
                {explainResult.high_impact_atoms.map((idx) => (
                  <Badge key={idx} variant="warning">
                    Atom {idx}
                  </Badge>
                ))}
              </div>
            </CardContent>
          </Card>
        )}

        {agentResult && (
          <Card className="mt-6">
            <CardHeader>
              <div className="flex items-center gap-2">
                <Brain className="h-4 w-4 text-primary" />
                <h3 className="text-sm font-semibold">
                  AI Toxicologist Report — {agentResult.target_assay}
                </h3>
              </div>
            </CardHeader>
            <CardContent className="space-y-4">
              <div className="flex items-center gap-2">
                <span className="text-xs font-medium text-muted-foreground">
                  Risk Level:
                </span>
                <Badge
                  variant={
                    agentResult.risk_level === "Critical" || agentResult.risk_level === "High"
                      ? "toxic"
                      : agentResult.risk_level === "Moderate"
                      ? "warning"
                      : "safe"
                  }
                >
                  {agentResult.risk_level}
                </Badge>
                <span className="text-xs text-muted-foreground">
                  Confidence: {agentResult.confidence}
                </span>
              </div>

              <div>
                <h4 className="mb-1 text-xs font-semibold text-muted-foreground">
                  Toxicophore Assessment
                </h4>
                <p className="text-sm leading-relaxed">
                  {agentResult.toxicophore_assessment}
                </p>
              </div>

              <div>
                <h4 className="mb-1 text-xs font-semibold text-muted-foreground">
                  Biochemical Mechanism
                </h4>
                <p className="text-sm leading-relaxed">
                  {agentResult.biochemical_mechanism}
                </p>
              </div>

              {agentResult.structural_alerts_found.length > 0 && (
                <div>
                  <h4 className="mb-2 text-xs font-semibold text-muted-foreground">
                    Structural Alerts Found
                  </h4>
                  <div className="space-y-2">
                    {agentResult.structural_alerts_found.map((alert, i) => (
                      <div
                        key={i}
                        className={`rounded-lg border p-3 text-sm ${
                          alert.high_attribution
                            ? "border-destructive/30 bg-destructive/5"
                            : "border-border bg-secondary/30"
                        }`}
                      >
                        <div className="flex items-center gap-2">
                          {alert.high_attribution && (
                            <AlertTriangle className="h-3.5 w-3.5 text-destructive" />
                          )}
                          <span className="font-medium">{alert.alert_name}</span>
                        </div>
                        <p className="mt-1 text-xs text-muted-foreground">
                          {alert.details}
                        </p>
                      </div>
                    ))}
                  </div>
                </div>
              )}

              {agentResult.bioisosteric_replacements.length > 0 && (
                <div>
                  <h4 className="mb-1 text-xs font-semibold text-muted-foreground">
                    Suggested Bioisosteric Replacements
                  </h4>
                  <ul className="list-inside list-disc text-sm text-muted-foreground">
                    {agentResult.bioisosteric_replacements.map((rep, i) => (
                      <li key={i}>{rep}</li>
                    ))}
                  </ul>
                </div>
              )}
            </CardContent>
          </Card>
        )}
      </main>
    </div>
  );
}
