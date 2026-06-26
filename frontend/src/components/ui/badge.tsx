import { cn } from "@/lib/utils";

interface BadgeProps {
  variant?: "default" | "toxic" | "safe" | "warning" | "neutral";
  children: React.ReactNode;
  className?: string;
}

export function Badge({ variant = "default", children, className }: BadgeProps) {
  return (
    <span
      className={cn(
        "inline-flex items-center rounded-full px-2.5 py-0.5 text-xs font-medium",
        {
          "bg-primary/10 text-primary": variant === "default",
          "bg-red-50 text-red-700 dark:bg-red-900/30 dark:text-red-400": variant === "toxic",
          "bg-green-50 text-green-700 dark:bg-green-900/30 dark:text-green-400": variant === "safe",
          "bg-amber-50 text-amber-700 dark:bg-amber-900/30 dark:text-amber-400": variant === "warning",
          "bg-secondary text-secondary-foreground": variant === "neutral",
        },
        className
      )}
    >
      {children}
    </span>
  );
}
