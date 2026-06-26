"use client";

import { motion } from "framer-motion";
import { Badge } from "@/components/ui/badge";
import type { AssayPrediction } from "@/types";

interface AssayCardProps {
  prediction: AssayPrediction;
  index: number;
  onSelect?: (index: number) => void;
  selected?: boolean;
}

export function AssayCard({ prediction, index, onSelect, selected }: AssayCardProps) {
  const probPercent = (prediction.probability * 100).toFixed(1);
  const isToxic = prediction.predicted_class === "Toxic";

  const barColor = isToxic
    ? probPercent > "70"
      ? "bg-destructive"
      : "bg-warning"
    : "bg-success";

  const probColor = isToxic
    ? probPercent > "70"
      ? "text-destructive"
      : "text-warning"
    : "text-success";

  return (
    <motion.div
      initial={{ opacity: 0, y: 10 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.3, delay: index * 0.04 }}
      onClick={() => onSelect?.(index)}
      className={`cursor-pointer rounded-xl border p-4 transition-all duration-200 ${
        selected
          ? "border-primary bg-primary/5 shadow-sm"
          : "border-border bg-card hover:border-primary/30 hover:shadow-sm"
      }`}
    >
      <div className="mb-2 flex items-start justify-between">
        <div>
          <p className="text-sm font-semibold">{prediction.assay}</p>
          <p className="text-xs text-muted-foreground">{prediction.target_class}</p>
        </div>
        <Badge variant={isToxic ? "toxic" : "safe"}>
          {isToxic ? "Toxic" : "Safe"}
        </Badge>
      </div>

      <div className="mt-3">
        <div className="mb-1 flex items-center justify-between">
          <span className="text-xs text-muted-foreground">Probability</span>
          <span className={`text-xs font-semibold ${probColor}`}>
            {probPercent}%
          </span>
        </div>
        <div className="h-1.5 w-full overflow-hidden rounded-full bg-secondary">
          <motion.div
            initial={{ width: 0 }}
            animate={{ width: `${prediction.probability * 100}%` }}
            transition={{ duration: 0.6, delay: index * 0.05 }}
            className={`h-full rounded-full ${barColor}`}
          />
        </div>
      </div>
    </motion.div>
  );
}
