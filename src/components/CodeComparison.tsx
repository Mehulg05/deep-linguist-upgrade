import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Code, ArrowRight, CheckCircle, XCircle } from "lucide-react";

export const CodeComparison = () => {
  const lstmCode = `import tensorflow as tf
from tensorflow.keras import layers, models

# LSTM-based Sentiment Analysis Model
def create_lstm_model(vocab_size, embedding_dim, max_length):
    model = models.Sequential([
        layers.Embedding(vocab_size, embedding_dim, input_length=max_length),
        layers.LSTM(128, dropout=0.5, recurrent_dropout=0.5),
        layers.Dense(64, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(1, activation='sigmoid')
    ])
    
    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    return model

# Training
model = create_lstm_model(vocab_size=10000, embedding_dim=100, max_length=200)
history = model.fit(X_train, y_train, epochs=10, batch_size=32)`;

  const gruCode = `import tensorflow as tf
from tensorflow.keras import layers, models

# GRU-based Sentiment Analysis Model (Improved)
def create_gru_model(vocab_size, embedding_dim, max_length):
    model = models.Sequential([
        layers.Embedding(vocab_size, embedding_dim, input_length=max_length),
        layers.GRU(128, dropout=0.5, recurrent_dropout=0.5),
        layers.Dense(64, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(1, activation='sigmoid')
    ])
    
    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    return model

# Training (Faster convergence expected)
model = create_gru_model(vocab_size=10000, embedding_dim=100, max_length=200)
history = model.fit(X_train, y_train, epochs=10, batch_size=32)`;

  const differences = [
    {
      aspect: "Parameters",
      lstm: "More parameters (3 gates)",
      gru: "Fewer parameters (2 gates)",
      better: "gru"
    },
    {
      aspect: "Training Speed",
      lstm: "Slower training",
      gru: "Faster training",
      better: "gru"
    },
    {
      aspect: "Memory Usage",
      lstm: "Higher memory usage",
      gru: "Lower memory usage",
      better: "gru"
    },
    {
      aspect: "Complexity",
      lstm: "More complex gating",
      gru: "Simpler gating mechanism",
      better: "gru"
    },
    {
      aspect: "Performance",
      lstm: "Slightly better on some tasks",
      gru: "Comparable performance",
      better: "equal"
    }
  ];

  return (
    <div className="space-y-6">
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Code className="h-5 w-5" />
            LSTM to GRU Migration Guide
          </CardTitle>
          <CardDescription>
            Converting your sentiment analysis model from LSTM to GRU for better efficiency
          </CardDescription>
        </CardHeader>
        <CardContent>
          <Tabs defaultValue="comparison" className="w-full">
            <TabsList className="grid w-full grid-cols-2">
              <TabsTrigger value="comparison">Code Comparison</TabsTrigger>
              <TabsTrigger value="differences">Key Differences</TabsTrigger>
            </TabsList>

            <TabsContent value="comparison" className="space-y-4">
              <div className="grid md:grid-cols-2 gap-4">
                <Card>
                  <CardHeader>
                    <CardTitle className="text-lg flex items-center gap-2">
                      <XCircle className="h-4 w-4 text-destructive" />
                      LSTM Implementation
                    </CardTitle>
                    <CardDescription>Original code using LSTM layers</CardDescription>
                  </CardHeader>
                  <CardContent>
                    <pre className="bg-muted p-4 rounded-lg text-xs overflow-x-auto">
                      <code>{lstmCode}</code>
                    </pre>
                  </CardContent>
                </Card>

                <Card>
                  <CardHeader>
                    <CardTitle className="text-lg flex items-center gap-2">
                      <CheckCircle className="h-4 w-4 text-success" />
                      GRU Implementation
                    </CardTitle>
                    <CardDescription>Updated code using GRU layers</CardDescription>
                  </CardHeader>
                  <CardContent>
                    <pre className="bg-muted p-4 rounded-lg text-xs overflow-x-auto">
                      <code>{gruCode}</code>
                    </pre>
                  </CardContent>
                </Card>
              </div>

              <Card className="bg-gradient-neural text-primary-foreground">
                <CardContent className="pt-6">
                  <div className="flex items-center gap-2 mb-2">
                    <ArrowRight className="h-5 w-5" />
                    <span className="font-semibold">Migration Summary</span>
                  </div>
                  <p className="text-sm opacity-90">
                    Simply replace <code className="bg-black/20 px-1 rounded">layers.LSTM</code> with <code className="bg-black/20 px-1 rounded">layers.GRU</code> 
                    in your model definition. The rest of the code remains identical!
                  </p>
                </CardContent>
              </Card>
            </TabsContent>

            <TabsContent value="differences" className="space-y-4">
              <div className="space-y-3">
                {differences.map((diff, index) => (
                  <Card key={index}>
                    <CardContent className="pt-4">
                      <div className="flex items-center justify-between">
                        <div className="flex-1">
                          <h4 className="font-semibold text-sm mb-1">{diff.aspect}</h4>
                          <div className="grid grid-cols-2 gap-4 text-xs">
                            <div className="flex items-center gap-2">
                              <Badge variant="outline" className="text-xs">LSTM</Badge>
                              <span className="text-muted-foreground">{diff.lstm}</span>
                            </div>
                            <div className="flex items-center gap-2">
                              <Badge variant="outline" className="text-xs">GRU</Badge>
                              <span className="text-muted-foreground">{diff.gru}</span>
                            </div>
                          </div>
                        </div>
                        <div className="ml-4">
                          {diff.better === "gru" && <CheckCircle className="h-5 w-5 text-success" />}
                          {diff.better === "lstm" && <XCircle className="h-5 w-5 text-destructive" />}
                          {diff.better === "equal" && <div className="h-5 w-5 rounded-full bg-muted" />}
                        </div>
                      </div>
                    </CardContent>
                  </Card>
                ))}
              </div>

              <Card className="bg-success text-success-foreground">
                <CardContent className="pt-6">
                  <div className="flex items-center gap-2 mb-2">
                    <CheckCircle className="h-5 w-5" />
                    <span className="font-semibold">Why Choose GRU?</span>
                  </div>
                  <ul className="text-sm space-y-1 opacity-90">
                    <li>• Faster training and inference</li>
                    <li>• Fewer parameters to tune</li>
                    <li>• Better generalization on smaller datasets</li>
                    <li>• Easier to optimize and debug</li>
                  </ul>
                </CardContent>
              </Card>
            </TabsContent>
          </Tabs>
        </CardContent>
      </Card>
    </div>
  );
};