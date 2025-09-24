import { useState } from "react";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Textarea } from "@/components/ui/textarea";
import { Badge } from "@/components/ui/badge";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Brain, Code, Sparkles, ArrowRight } from "lucide-react";
import { toast } from "@/hooks/use-toast";
import { CodeComparison } from "@/components/CodeComparison";
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
      // Simulate sentiment analysis (in real implementation, this would use the GRU model)
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
            GRU Sentiment Analysis
          </h1>
          <p className="text-xl text-muted-foreground max-w-2xl mx-auto">
            Advanced sentiment analysis using Gated Recurrent Units (GRU) for more efficient and accurate text understanding
          </p>
        </div>

        <Tabs defaultValue="analyzer" className="w-full">
          <TabsList className="grid w-full grid-cols-2">
            <TabsTrigger value="analyzer">Live Analyzer</TabsTrigger>
            <TabsTrigger value="code">LSTM → GRU Conversion</TabsTrigger>
          </TabsList>

          <TabsContent value="analyzer" className="space-y-6">
            <Card>
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <Brain className="h-5 w-5" />
                  Text Sentiment Analyzer
                </CardTitle>
                <CardDescription>
                  Enter any text to analyze its emotional sentiment using our GRU-based model
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
          </TabsContent>

          <TabsContent value="code">
            <CodeComparison />
          </TabsContent>
        </Tabs>

        {/* Features Section */}
        <div className="grid md:grid-cols-3 gap-6 mt-12">
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <Brain className="h-5 w-5 text-primary" />
                GRU Architecture
              </CardTitle>
            </CardHeader>
            <CardContent>
              <p className="text-sm text-muted-foreground">
                GRUs offer simpler architecture compared to LSTMs while maintaining comparable performance with fewer parameters.
              </p>
            </CardContent>
          </Card>

          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <Sparkles className="h-5 w-5 text-accent" />
                Better Performance
              </CardTitle>
            </CardHeader>
            <CardContent>
              <p className="text-sm text-muted-foreground">
                Faster training and inference times due to simplified gating mechanism and reduced computational complexity.
              </p>
            </CardContent>
          </Card>

          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <Code className="h-5 w-5 text-success" />
                Easy Migration
              </CardTitle>
            </CardHeader>
            <CardContent>
              <p className="text-sm text-muted-foreground">
                Simple drop-in replacement for LSTM layers with minimal code changes required for existing projects.
              </p>
            </CardContent>
          </Card>
        </div>
      </div>
    </div>
  );
};

export default Index;