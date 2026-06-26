"use client";

import { AssayCard } from "./AssayCard";
import type { AssayPrediction } from "@/types";

interface PredictionGridProps {
  predictions: AssayPrediction[];
  selectedIndex: number | null;
  onSelectAssay: (index: number) => void;
}

export function PredictionGrid({
  predictions,
  selectedIndex,
  onSelectAssay,
}: PredictionGridProps) {
  if (predictions.length === 0) return null;

  const toxicCount = predictions.filter((p) => p.predicted_class === "Toxic").length;
  const safeCount = predictions.length - toxicCount;

  return (
    <div>
      <div className="mb-4 flex items-center gap-3">
        <h2 className="text-sm font-semibold">Predictions</h2>
        <div className="flex gap-2 text-xs text-muted-foreground">
          <span className="text-destructive">{toxicCount} Toxic</span>
          <span>/</span>
          <span className="text-success">{safeCount} Non-Toxic</span>
        </div>
      </div>

      <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4">
        {predictions.map((pred, i) => (
          <AssayCard
            key={pred.assay}
            prediction={pred}
            index={i}
            onSelect={onSelectAssay}
            selected={selectedIndex === i}
          />
        ))}
      </div>
    </div>
  );
}
