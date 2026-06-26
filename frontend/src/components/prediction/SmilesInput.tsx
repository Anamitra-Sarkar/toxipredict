"use client";

import { useState } from "react";
import { FlaskConical, Trash2 } from "lucide-react";
import { Button } from "@/components/ui/button";

const EXAMPLE_SMILES = [
  { name: "4-Chloroaniline", smiles: "Nc1ccc(Cl)cc1" },
  { name: "Aspirin", smiles: "CC(=O)Oc1ccccc1C(=O)O" },
  { name: "Caffeine", smiles: "CN1C=NC2=C1C(=O)N(C(=O)N2C)C" },
  { name: "Paracetamol", smiles: "CC(=O)Nc1ccc(O)cc1" },
  { name: "Benzene", smiles: "c1ccccc1" },
];

interface SmilesInputProps {
  onPredict: (smiles: string) => void;
  loading: boolean;
}

export function SmilesInput({ onPredict, loading }: SmilesInputProps) {
  const [smiles, setSmiles] = useState("");

  const handlePredict = () => {
    if (!smiles.trim()) return;
    onPredict(smiles.trim());
  };

  return (
    <div className="rounded-xl border border-border bg-card p-5">
      <div className="mb-3 flex items-center gap-2">
        <FlaskConical className="h-4 w-4 text-primary" />
        <h2 className="text-sm font-semibold">Input SMILES</h2>
      </div>

      <textarea
        value={smiles}
        onChange={(e) => setSmiles(e.target.value)}
        placeholder="Enter a SMILES string, e.g., Nc1ccc(Cl)cc1"
        className="font-mono w-full rounded-lg border border-border bg-background p-3 text-sm focus:border-primary focus:outline-none focus:ring-1 focus:ring-primary"
        rows={2}
      />

      <div className="mt-3 flex flex-wrap items-center gap-2">
        <Button onClick={handlePredict} disabled={loading || !smiles.trim()}>
          {loading ? "Predicting..." : "Predict"}
        </Button>
        <Button
          variant="ghost"
          size="sm"
          onClick={() => setSmiles("")}
          disabled={!smiles}
        >
          <Trash2 className="h-3.5 w-3.5" />
        </Button>
      </div>

      <div className="mt-3">
        <p className="mb-1.5 text-xs text-muted-foreground">Examples:</p>
        <div className="flex flex-wrap gap-1.5">
          {EXAMPLE_SMILES.map((ex) => (
            <button
              key={ex.name}
              onClick={() => setSmiles(ex.smiles)}
              className="rounded-md border border-border bg-secondary/30 px-2.5 py-1 text-xs text-muted-foreground transition-all hover:border-primary/30 hover:text-foreground"
            >
              {ex.name}
            </button>
          ))}
        </div>
      </div>
    </div>
  );
}
