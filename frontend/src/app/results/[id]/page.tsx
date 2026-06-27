"use client";

import { useState, useEffect } from "react";
import { useRouter, useParams } from "next/navigation";
import { auth } from "@/lib/firebase";
import { Header } from "@/components/layout/Header";
import { Card, CardContent, CardHeader } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { PredictionGrid } from "@/components/prediction/PredictionGrid";
import type { AssayPrediction } from "@/types";
import { FlaskConical, Brain, AlertTriangle, ArrowLeft, Loader2 } from "lucide-react";
import Link from "next/link";

const API_BASE = "https://arko006-toxipredict-api.hf.space";

interface DetailData {
  analysis_id: string;
  smiles: string;
  created_at: string;
  predictions: AssayPrediction[];
  uncertainty_weights: number[];
  molecule_num_atoms: number;
}

export default function ResultDetailPage() {
  const router = useRouter();
  const params = useParams();
  const [user, setUser] = useState<object | null>(null);
  const [data, setData] = useState<DetailData | null>(null);
  const [loading, setLoading] = useState(true);
  const [selectedAssay, setSelectedAssay] = useState<number | null>(null);
  const [shapImage, setShapImage] = useState<string | null>(null);
  const [shapLoading, setShapLoading] = useState(false);

  useEffect(() => {
    const unsubscribe = auth.onAuthStateChanged((u) => {
      if (!u) {
        router.push("/login");
      } else {
        setUser(u);
        fetchDetail(u, params.id as string);
      }
    });
    return unsubscribe;
  }, [router, params.id]);

  const fetchDetail = async (u: { getIdToken: () => Promise<string> }, id: string) => {
    try {
      const token = await u.getIdToken();
      const res = await fetch(`${API_BASE}/api/history/${id}`, {
        headers: { Authorization: `Bearer ${token}` },
      });
      if (res.ok) {
        const detail = await res.json();
        setData(detail);
      }
    } catch (err) {
      console.error("Failed to fetch detail:", err);
    } finally {
      setLoading(false);
    }
  };

  const handleExplain = async () => {
    if (!data || selectedAssay === null) return;
    setShapLoading(true);
    try {
      const token = await auth.currentUser!.getIdToken();
      const res = await fetch(`${API_BASE}/api/explain`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          Authorization: `Bearer ${token}`,
        },
        body: JSON.stringify({
          smiles: data.smiles,
          target_task: selectedAssay,
        }),
      });
      if (res.ok) {
        const result = await res.json();
        if (result.attribution_map_base64) {
          setShapImage(`data:image/png;base64,${result.attribution_map_base64}`);
        }
      }
    } catch (err) {
      console.error("Explain failed:", err);
    } finally {
      setShapLoading(false);
    }
  };

  if (!user || loading) {
    return (
      <div className="flex min-h-screen items-center justify-center bg-background">
        <Loader2 className="h-6 w-6 animate-spin text-primary" />
      </div>
    );
  }

  if (!data) {
    return (
      <div className="min-h-screen bg-background">
        <Header />
        <main className="mx-auto max-w-4xl px-4 py-20 text-center">
          <p className="text-muted-foreground">Analysis not found</p>
          <Link href="/history" className="mt-4 inline-block text-sm text-primary hover:underline">
            Back to History
          </Link>
        </main>
      </div>
    );
  }

  const toxicCount = data.predictions.filter((p) => p.predicted_class === "Toxic").length;

  return (
    <div className="min-h-screen bg-background">
      <Header />
      <main className="mx-auto max-w-7xl px-4 py-6 sm:px-6">
        <div className="mb-6">
          <Link
            href="/history"
            className="mb-3 inline-flex items-center gap-1 text-xs text-muted-foreground hover:text-foreground"
          >
            <ArrowLeft className="h-3 w-3" />
            Back to History
          </Link>
          <div className="flex items-start justify-between">
            <div>
              <h1 className="text-lg font-bold">Analysis Detail</h1>
              <p className="font-mono mt-1 text-sm">{data.smiles}</p>
              <p className="mt-1 text-xs text-muted-foreground">
                {new Date(data.created_at).toLocaleString()} &middot;{" "}
                {data.molecule_num_atoms} atoms &middot; {data.analysis_id.slice(0, 8)}...
              </p>
            </div>
            <div className="flex items-center gap-2">
              <Badge variant={toxicCount > 0 ? "toxic" : "safe"}>
                {toxicCount} / {data.predictions.length} Toxic
              </Badge>
            </div>
          </div>
        </div>

        <div className="grid gap-6 lg:grid-cols-4">
          <div className="lg:col-span-3">
            <PredictionGrid
              predictions={data.predictions as AssayPrediction[]}
              selectedIndex={selectedAssay}
              onSelectAssay={setSelectedAssay}
            />
          </div>

          <div className="space-y-4 lg:col-span-1">
            <Card>
              <CardHeader>
                <h3 className="text-sm font-semibold">Actions</h3>
              </CardHeader>
              <CardContent className="space-y-3">
                <Button
                  className="w-full"
                  size="sm"
                  onClick={handleExplain}
                  disabled={selectedAssay === null || shapLoading}
                >
                  {shapLoading ? (
                    <Loader2 className="mr-1.5 h-3.5 w-3.5 animate-spin" />
                  ) : (
                    <Brain className="mr-1.5 h-3.5 w-3.5" />
                  )}
                  Show SHAP Map
                </Button>
              </CardContent>
            </Card>

            <Card>
              <CardHeader>
                <h3 className="text-sm font-semibold">
                  Uncertainty Weights
                </h3>
              </CardHeader>
              <CardContent className="space-y-2">
                {data.uncertainty_weights.map((w, i) => (
                  <div key={i} className="flex items-center justify-between text-xs">
                    <span className="text-muted-foreground">
                      {data.predictions[i]?.assay || `Task ${i}`}
                    </span>
                    <span className="font-mono">{w.toFixed(3)}</span>
                  </div>
                ))}
              </CardContent>
            </Card>
          </div>
        </div>

        {shapImage && selectedAssay !== null && (
          <Card className="mt-6">
            <CardHeader>
              <h3 className="text-sm font-semibold">
                SHAP Attribution — {data.predictions[selectedAssay]?.assay}
              </h3>
            </CardHeader>
            <CardContent>
              <div className="flex justify-center">
                <img
                  src={shapImage}
                  alt="SHAP Attribution Map"
                  className="max-h-96 rounded-lg"
                />
              </div>
            </CardContent>
          </Card>
        )}
      </main>
    </div>
  );
}
