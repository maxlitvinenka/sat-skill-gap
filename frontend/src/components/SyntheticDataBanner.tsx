import { AlertTriangle } from "lucide-react";
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";

export function SyntheticDataBanner({ liveMode }: { liveMode?: boolean }) {
  return (
    <Alert variant="warning" className="mb-6 flex gap-3 items-start">
      <AlertTriangle className="h-5 w-5 shrink-0 text-brand-gold mt-0.5" />
      <div>
        <AlertTitle>
          Synthetic demo data only
          {liveMode && (
            <span className="ml-2 inline-flex items-center rounded-full bg-emerald-600/15 px-2 py-0.5 text-xs font-medium text-emerald-700">
              Live mode
            </span>
          )}
        </AlertTitle>
        <AlertDescription>
          This dashboard uses simulated student scores. No real student information is included.
          {liveMode && " Predictions run on-demand via the Python API."}
        </AlertDescription>
      </div>
    </Alert>
  );
}
