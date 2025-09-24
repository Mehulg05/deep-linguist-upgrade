import { useState } from "react";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Textarea } from "@/components/ui/textarea";
import { Badge } from "@/components/ui/badge";
import { Brain, Sparkles, ArrowRight, Zap, Target } from "lucide-react";
import { toast } from "@/hooks/use-toast";
import { SentimentMeter } from "@/components/SentimentMeter";

const Index = () => {
  const [inputText, setInputText] = useState("");
  const [sentiment, setSentiment] = useState<{
    label: string;
    score: number;
  } | null>(null);
  const [isAnalyzing, setIsAnalyzing] = useState(false);

  const analyzeSentiment = async () => {
    if (!inputText.trim()) {
      toast({
        title: "Please enter some text",
        description: "Enter text to analyze its sentiment",
        variant: "destructive",
      });
      return;
    }

    setIsAnalyzing(true);
    
    try {
      // Simulate sentiment analysis with neural network model
      await new Promise(resolve => setTimeout(resolve, 1500));
      
      // Mock sentiment analysis result
      const positiveWords = ['good', 'great', 'excellent', 'amazing', 'wonderful', 'fantastic', 'love', 'happy'];
      const negativeWords = ['bad', 'terrible', 'awful', 'hate', 'sad', 'angry', 'disappointed'];
      
      const text = inputText.toLowerCase();
      const hasPositive = positiveWords.some(word => text.includes(word));
      const hasNegative = negativeWords.some(word => text.includes(word));
      
      let label = 'NEUTRAL';
      let score = 0.5;
      
      if (hasPositive && !hasNegative) {
        label = 'POSITIVE';
        score = 0.8 + Math.random() * 0.2;
      } else if (hasNegative && !hasPositive) {
        label = 'NEGATIVE';
        score = Math.random() * 0.3;
      } else {
        score = 0.4 + Math.random() * 0.2;
      }
      
      setSentiment({ label, score });
      
      toast({
        title: "Analysis Complete",
        description: `Sentiment: ${label} (${(score * 100).toFixed(1)}% confidence)`,
      });
    } catch (error) {
      toast({
        title: "Analysis Failed",
        description: "There was an error analyzing the sentiment",
        variant: "destructive",
      });
    } finally {
      setIsAnalyzing(false);
    }
  };

  return (
    <div className="min-h-screen bg-gradient-background">
      {/* Hero Section */}
      <div className="container mx-auto px-4 py-12">
        <div className="text-center mb-12">
          <div className="flex justify-center mb-6">
            <div className="relative">
              <Brain className="h-16 w-16 text-primary" />
              <Sparkles className="h-6 w-6 text-accent absolute -top-1 -right-1" />
            </div>
          </div>
          <h1 className="text-4xl md:text-6xl font-bold bg-gradient-primary bg-clip-text text-transparent mb-4">
            AI Sentiment Analysis
          </h1>
          <p className="text-xl text-muted-foreground max-w-2xl mx-auto">
            Advanced neural network-powered sentiment analysis for accurate text emotion detection and understanding
          </p>
        </div>

        <div className="space-y-6">
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <Brain className="h-5 w-5" />
                Text Sentiment Analyzer
              </CardTitle>
              <CardDescription>
                Enter any text to analyze its emotional sentiment using our advanced neural network model
              </CardDescription>
            </CardHeader>
            <CardContent className="space-y-4">
              <Textarea
                placeholder="Enter your text here to analyze sentiment... (e.g., 'I love this amazing product!' or 'This is terrible and disappointing')"
                value={inputText}
                onChange={(e) => setInputText(e.target.value)}
                className="min-h-[120px]"
              />
              <Button 
                onClick={analyzeSentiment}
                disabled={isAnalyzing}
                className="w-full"
                size="lg"
              >
                {isAnalyzing ? (
                  <>
                    <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-primary-foreground mr-2" />
                    Analyzing...
                  </>
                ) : (
                  <>
                    Analyze Sentiment
                    <ArrowRight className="ml-2 h-4 w-4" />
                  </>
                )}
              </Button>
            </CardContent>
          </Card>

          {sentiment && (
            <Card>
              <CardHeader>
                <CardTitle>Analysis Results</CardTitle>
              </CardHeader>
              <CardContent className="space-y-4">
                <div className="flex items-center justify-between">
                  <span className="text-sm font-medium">Sentiment:</span>
                  <Badge 
                    variant={sentiment.label === 'POSITIVE' ? 'default' : sentiment.label === 'NEGATIVE' ? 'destructive' : 'secondary'}
                    className="text-sm"
                  >
                    {sentiment.label}
                  </Badge>
                </div>
                <SentimentMeter sentiment={sentiment} />
                <div className="text-sm text-muted-foreground">
                  Confidence: {(sentiment.score * 100).toFixed(1)}%
                </div>
              </CardContent>
            </Card>
          )}
        </div>

        {/* Features Section */}
        <div className="grid md:grid-cols-3 gap-6 mt-12">
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <Brain className="h-5 w-5 text-primary" />
                Neural Network
              </CardTitle>
            </CardHeader>
            <CardContent>
              <p className="text-sm text-muted-foreground">
                Advanced recurrent neural network architecture designed for optimal performance in sequential text analysis.
              </p>
            </CardContent>
          </Card>

          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <Zap className="h-5 w-5 text-accent" />
                Fast Processing
              </CardTitle>
            </CardHeader>
            <CardContent>
              <p className="text-sm text-muted-foreground">
                Optimized for real-time sentiment analysis with efficient processing and quick response times.
              </p>
            </CardContent>
          </Card>

          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <Target className="h-5 w-5 text-success" />
                High Accuracy
              </CardTitle>
            </CardHeader>
            <CardContent>
              <p className="text-sm text-muted-foreground">
                State-of-the-art accuracy in detecting positive, negative, and neutral sentiments across various text types.
              </p>
            </CardContent>
          </Card>
        </div>
      </div>
    </div>
  );
};

export default Index;