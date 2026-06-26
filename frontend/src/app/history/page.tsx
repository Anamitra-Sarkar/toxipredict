"use client";

import { useState, useEffect } from "react";
import { useRouter } from "next/navigation";
import Link from "next/link";
import { auth } from "@/lib/firebase";
import { Header } from "@/components/layout/Header";
import { Card, CardContent } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { FlaskConical, History as HistoryIcon, ArrowRight } from "lucide-react";
import type { AssayPrediction } from "@/types";

const API_BASE = process.env.NEXT_PUBLIC_API_URL || "";

interface HistoryItem {
  analysis_id: string;
  smiles: string;
  created_at: string;
  predictions: AssayPrediction[];
  molecule_num_atoms?: number;
}

export default function HistoryPage() {
  const router = useRouter();
  const [user, setUser] = useState<object | null>(null);
  const [analyses, setAnalyses] = useState<HistoryItem[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const unsubscribe = auth.onAuthStateChanged((u) => {
      if (!u) {
        router.push("/login");
      } else {
        setUser(u);
        fetchHistory(u);
      }
    });
    return unsubscribe;
  }, [router]);

  const fetchHistory = async (u: { getIdToken: () => Promise<string> }) => {
    try {
      const token = await u.getIdToken();
      const res = await fetch(`${API_BASE}/api/history?limit=50`, {
        headers: { Authorization: `Bearer ${token}` },
      });
      if (res.ok) {
        const data = await res.json();
        setAnalyses(data.analyses || []);
      }
    } catch (err) {
      console.error("Failed to fetch history:", err);
    } finally {
      setLoading(false);
    }
  };

  if (!user) return null;

  return (
    <div className="min-h-screen bg-background">
      <Header />
      <main className="mx-auto max-w-4xl px-4 py-6 sm:px-6">
        <div className="mb-6 flex items-center gap-2">
          <HistoryIcon className="h-5 w-5 text-primary" />
          <h1 className="text-lg font-bold">Analysis History</h1>
        </div>

        {loading && (
          <div className="flex items-center justify-center py-20 text-sm text-muted-foreground">
            Loading...
          </div>
        )}

        {!loading && analyses.length === 0 && (
          <div className="flex flex-col items-center justify-center rounded-xl border border-dashed border-border py-20">
            <FlaskConical className="mb-3 h-8 w-8 text-muted-foreground" />
            <p className="text-sm text-muted-foreground">No analyses yet</p>
            <Link
              href="/dashboard"
              className="mt-3 text-sm font-medium text-primary hover:underline"
            >
              Run your first prediction
            </Link>
          </div>
        )}

        {!loading && analyses.length > 0 && (
          <div className="space-y-3">
            {analyses.map((item) => {
              const toxicCount = (item.predictions || []).filter(
                (p: AssayPrediction) => p.predicted_class === "Toxic"
              ).length;
              return (
                <Link key={item.analysis_id} href={`/results/${item.analysis_id}`}>
                  <Card hoverable>
                    <CardContent>
                      <div className="flex items-center justify-between">
                        <div className="min-w-0 flex-1">
                          <p className="font-mono text-sm">{item.smiles}</p>
                          <div className="mt-1 flex items-center gap-3 text-xs text-muted-foreground">
                            <span>
                              {new Date(item.created_at).toLocaleDateString()}
                            </span>
                            <span>{item.molecule_num_atoms || "?"} atoms</span>
                            <span>
                              {item.predictions?.length || 0} assays
                            </span>
                          </div>
                        </div>
                        <div className="flex items-center gap-3">
                          {toxicCount > 0 && (
                            <Badge variant="toxic">
                              {toxicCount} Toxic
                            </Badge>
                          )}
                          <ArrowRight className="h-4 w-4 text-muted-foreground" />
                        </div>
                      </div>
                    </CardContent>
                  </Card>
                </Link>
              );
            })}
          </div>
        )}
      </main>
    </div>
  );
}
