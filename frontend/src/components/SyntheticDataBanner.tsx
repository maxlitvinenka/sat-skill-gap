import { AlertTriangle } from "lucide-react";
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";

export function SyntheticDataBanner() {
  return (
    <Alert variant="warning" className="mb-6 flex gap-3 items-start">
      <AlertTriangle className="h-5 w-5 shrink-0 text-brand-gold mt-0.5" />
      <div>
        <AlertTitle>Synthetic demo data only</AlertTitle>
        <AlertDescription>
          This dashboard uses simulated student scores. No real student information is included.
        </AlertDescription>
      </div>
    </Alert>
  );
}
