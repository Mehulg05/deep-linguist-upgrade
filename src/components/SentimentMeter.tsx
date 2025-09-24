import { Progress } from "@/components/ui/progress";

interface SentimentMeterProps {
  sentiment: {
    label: string;
    score: number;
  };
}

export const SentimentMeter = ({ sentiment }: SentimentMeterProps) => {
  const getColor = () => {
    if (sentiment.label === 'POSITIVE') return 'bg-sentiment-positive';
    if (sentiment.label === 'NEGATIVE') return 'bg-sentiment-negative';
    return 'bg-sentiment-neutral';
  };

  const getGradient = () => {
    if (sentiment.label === 'POSITIVE') return 'from-sentiment-positive to-success';
    if (sentiment.label === 'NEGATIVE') return 'from-sentiment-negative to-destructive';
    return 'from-sentiment-neutral to-muted-foreground';
  };

  return (
    <div className="space-y-2">
      <div className="flex justify-between text-xs text-muted-foreground">
        <span>Negative</span>
        <span>Neutral</span>
        <span>Positive</span>
      </div>
      <div className="relative">
        <Progress 
          value={sentiment.score * 100} 
          className="h-3 bg-secondary"
        />
        <div 
          className={`absolute top-0 h-3 rounded-full bg-gradient-to-r ${getGradient()} transition-all duration-500`}
          style={{ width: `${sentiment.score * 100}%` }}
        />
      </div>
      <div className="text-center text-sm font-medium">
        {sentiment.score < 0.4 ? '😔 Negative' : sentiment.score > 0.6 ? '😊 Positive' : '😐 Neutral'}
      </div>
    </div>
  );
};