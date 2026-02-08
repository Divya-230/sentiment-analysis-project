from flask import Flask, render_template, request, jsonify, send_file
import pandas as pd
import os
import io
from textblob import TextBlob
import matplotlib.pyplot as plt
import base64
from io import BytesIO
from wordcloud import WordCloud
import numpy as np

app = Flask(__name__)
app.secret_key = 'fixed-sentiment-analysis-2024'

# Global variable to store the current analysis
CURRENT_ANALYSIS = None

class SimpleSentimentAnalyzer:
    def analyze_sentiment(self, text):
        analysis = TextBlob(str(text))
        if analysis.sentiment.polarity > 0.1:
            return 1, 'Positive'
        elif analysis.sentiment.polarity < -0.1:
            return -1, 'Negative'
        else:
            return 0, 'Neutral'
    
    def analyze_dataframe(self, df, text_column):
        df_analyzed = df.copy()
        sentiments = []
        labels = []
        
        for text in df_analyzed[text_column]:
            sentiment, label = self.analyze_sentiment(text)
            sentiments.append(sentiment)
            labels.append(label)
        
        df_analyzed['sentiment'] = sentiments
        df_analyzed['sentiment_label'] = labels
        return df_analyzed

class SimpleVisualizer:
    def create_pie_chart(self, df):
        sentiment_counts = df['sentiment_label'].value_counts()
        
        fig, ax = plt.subplots(figsize=(8, 6))
        colors = ['#28a745', '#ffc107', '#dc3545']  # green, yellow, red
        
        wedges, texts, autotexts = ax.pie(
            sentiment_counts.values, 
            labels=sentiment_counts.index,
            colors=colors[:len(sentiment_counts)],
            autopct='%1.1f%%',
            startangle=90
        )
        
        ax.set_title('Sentiment Distribution')
        
        # Convert to base64
        img_buffer = BytesIO()
        plt.savefig(img_buffer, format='png', bbox_inches='tight', dpi=100)
        plt.close()
        
        img_buffer.seek(0)
        img_str = base64.b64encode(img_buffer.read()).decode()
        return f"data:image/png;base64,{img_str}"
    
    def create_bar_chart(self, df):
        sentiment_counts = df['sentiment_label'].value_counts()
        
        fig, ax = plt.subplots(figsize=(8, 6))
        colors = ['#28a745', '#ffc107', '#dc3545']
        
        bars = ax.bar(sentiment_counts.index, sentiment_counts.values, 
                     color=colors[:len(sentiment_counts)])
        ax.set_title('Sentiment Counts')
        ax.set_xlabel('Sentiment')
        ax.set_ylabel('Count')
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{int(height)}', ha='center', va='bottom')
        
        # Convert to base64
        img_buffer = BytesIO()
        plt.savefig(img_buffer, format='png', bbox_inches='tight', dpi=100)
        plt.close()
        
        img_buffer.seek(0)
        img_str = base64.b64encode(img_buffer.read()).decode()
        return f"data:image/png;base64,{img_str}"
    
    def create_wordcloud(self, df, sentiment=None):
        if sentiment:
            texts = df[df['sentiment_label'] == sentiment]['feedback']
        else:
            texts = df['feedback']
        
        all_text = ' '.join(texts.astype(str))
        
        if len(all_text.strip()) == 0:
            all_text = "No data available"
        
        wordcloud = WordCloud(
            width=400, 
            height=300, 
            background_color='white',
            max_words=50
        ).generate(all_text)
        
        # Convert to base64 for HTML display
        img_buffer = BytesIO()
        plt.figure(figsize=(8, 6))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        plt.tight_layout(pad=0)
        plt.savefig(img_buffer, format='png', bbox_inches='tight', dpi=100)
        plt.close()
        
        img_buffer.seek(0)
        img_str = base64.b64encode(img_buffer.read()).decode()
        return f"data:image/png;base64,{img_str}"

# Initialize components
analyzer = SimpleSentimentAnalyzer()
visualizer = SimpleVisualizer()

@app.route('/')
def index():
    return '''
    <!DOCTYPE html>
    <html>
    <head>
        <title>Fixed Sentiment Analysis</title>
        <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/css/bootstrap.min.css" rel="stylesheet">
    </head>
    <body>
        <div class="container mt-5">
            <div class="row">
                <div class="col-md-8 mx-auto">
                    <div class="card shadow">
                        <div class="card-header bg-primary text-white">
                            <h2 class="text-center">✅ FIXED: Policy Feedback Sentiment Analysis</h2>
                        </div>
                        <div class="card-body">
                            <p class="text-success"><strong>This is the FIXED version that will show your actual data!</strong></p>
                            
                            <form action="/analyze-file" method="post" enctype="multipart/form-data">
                                <div class="mb-3">
                                    <label class="form-label">Upload Feedback File (CSV):</label>
                                    <input class="form-control" type="file" name="file" accept=".csv" required>
                                    <div class="form-text">File must have a column named 'feedback' with text data</div>
                                </div>
                                <button type="submit" class="btn btn-success">Analyze File</button>
                            </form>
                            
                            <hr>
                            
                            <div class="mt-3">
                                <h5>Test with Sample Data:</h5>
                                <a href="/create-test-file" class="btn btn-outline-primary">Create Test File</a>
                                <a href="/test-analyze" class="btn btn-outline-success">Auto Test</a>
                                <a href="/dashboard" class="btn btn-info">Go to Dashboard</a>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    </body>
    </html>
    '''

@app.route('/analyze-file', methods=['POST'])
def analyze_file():
    global CURRENT_ANALYSIS
    
    try:
        if 'file' not in request.files:
            return "No file uploaded", 400
        
        file = request.files['file']
        if file.filename == '':
            return "No file selected", 400
        
        print(f"🔄 Processing: {file.filename}")
        
        # Read CSV file
        df = pd.read_csv(file)
        print(f"✅ Loaded {len(df)} rows. Columns: {list(df.columns)}")
        
        # Check if feedback column exists
        if 'feedback' not in df.columns:
            return f"Error: 'feedback' column not found. Available columns: {list(df.columns)}", 400
        
        # Analyze sentiment
        analyzed_df = analyzer.analyze_dataframe(df, 'feedback')
        
        # Calculate summary
        total = len(analyzed_df)
        positive = len(analyzed_df[analyzed_df['sentiment_label'] == 'Positive'])
        negative = len(analyzed_df[analyzed_df['sentiment_label'] == 'Negative'])
        neutral = len(analyzed_df[analyzed_df['sentiment_label'] == 'Neutral'])
        
        summary = {
            'total_feedback': total,
            'positive_count': positive,
            'negative_count': negative,
            'neutral_count': neutral,
            'positive_percentage': round((positive/total)*100, 2),
            'negative_percentage': round((negative/total)*100, 2),
            'neutral_percentage': round((neutral/total)*100, 2)
        }
        
        print(f"📊 Analysis Results: {summary}")
        
        # Store analysis globally
        CURRENT_ANALYSIS = {
            'analyzed_df': analyzed_df,
            'summary': summary,
            'filename': file.filename
        }
        
        # Generate visualizations
        pie_chart = visualizer.create_pie_chart(analyzed_df)
        bar_chart = visualizer.create_bar_chart(analyzed_df)
        wordcloud_positive = visualizer.create_wordcloud(analyzed_df, 'Positive')
        wordcloud_negative = visualizer.create_wordcloud(analyzed_df, 'Negative')
        wordcloud_neutral = visualizer.create_wordcloud(analyzed_df, 'Neutral')
        
        return f'''
        <!DOCTYPE html>
        <html>
        <head>
            <title>Analysis Complete</title>
            <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/css/bootstrap.min.css" rel="stylesheet">
        </head>
        <body>
            <div class="container mt-4">
                <div class="alert alert-success">
                    <h4>✅ Analysis Complete!</h4>
                    <p>File: {file.filename} | Total: {total} feedback entries</p>
                </div>
                
                <div class="row mb-4">
                    <div class="col-md-4">
                        <div class="card text-white bg-success">
                            <div class="card-body text-center">
                                <h3>{summary['positive_percentage']}%</h3>
                                <p>Positive ({positive})</p>
                            </div>
                        </div>
                    </div>
                    <div class="col-md-4">
                        <div class="card text-white bg-warning">
                            <div class="card-body text-center">
                                <h3>{summary['neutral_percentage']}%</h3>
                                <p>Neutral ({neutral})</p>
                            </div>
                        </div>
                    </div>
                    <div class="col-md-4">
                        <div class="card text-white bg-danger">
                            <div class="card-body text-center">
                                <h3>{summary['negative_percentage']}%</h3>
                                <p>Negative ({negative})</p>
                            </div>
                        </div>
                    </div>
                </div>
                
                <div class="text-center">
                    <a href="/dashboard" class="btn btn-primary btn-lg">📊 View Full Dashboard</a>
                    <a href="/" class="btn btn-secondary">Upload New File</a>
                </div>
                
                <div class="mt-4">
                    <h5>Sample of Analyzed Data:</h5>
                    <div class="table-responsive">
                        <table class="table table-striped">
                            <thead>
                                <tr><th>Feedback</th><th>Sentiment</th></tr>
                            </thead>
                            <tbody>
                                {"".join([f'<tr><td>{row["feedback"][:80]}...</td><td><span class="badge bg-{"success" if row["sentiment_label"] == "Positive" else "danger" if row["sentiment_label"] == "Negative" else "warning"}">{row["sentiment_label"]}</span></td></tr>' for row in analyzed_df.head(5).to_dict('records')])}
                            </tbody>
                        </table>
                    </div>
                </div>
            </div>
        </body>
        </html>
        '''
        
    except Exception as e:
        return f"Error: {str(e)}", 500

@app.route('/dashboard')
def dashboard():
    global CURRENT_ANALYSIS
    
    if CURRENT_ANALYSIS is None:
        return '''
        <!DOCTYPE html>
        <html>
        <head>
            <title>Dashboard - No Data</title>
            <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/css/bootstrap.min.css" rel="stylesheet">
        </head>
        <body>
            <div class="container mt-5">
                <div class="alert alert-warning text-center">
                    <h3>📊 Analytics Dashboard</h3>
                    <p>No data available yet. Please upload a file first.</p>
                    <a href="/" class="btn btn-primary">Upload File</a>
                    <a href="/test-analyze" class="btn btn-success">Load Test Data</a>
                </div>
            </div>
        </body>
        </html>
        '''
    
    # We have data - show real analytics
    analyzed_df = CURRENT_ANALYSIS['analyzed_df']
    summary = CURRENT_ANALYSIS['summary']
    filename = CURRENT_ANALYSIS['filename']
    
    # Generate visualizations
    pie_chart = visualizer.create_pie_chart(analyzed_df)
    bar_chart = visualizer.create_bar_chart(analyzed_df)
    wordcloud_positive = visualizer.create_wordcloud(analyzed_df, 'Positive')
    wordcloud_negative = visualizer.create_wordcloud(analyzed_df, 'Negative')
    wordcloud_neutral = visualizer.create_wordcloud(analyzed_df, 'Neutral')
    
    return f'''
    <!DOCTYPE html>
    <html>
    <head>
        <title>Dashboard - {filename}</title>
        <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/css/bootstrap.min.css" rel="stylesheet">
        <style>
            .card {{ margin-bottom: 20px; }}
            .stat-card {{ text-align: center; padding: 20px; }}
            .stat-number {{ font-size: 2.5rem; font-weight: bold; }}
        </style>
    </head>
    <body>
        <nav class="navbar navbar-dark bg-primary">
            <div class="container">
                <span class="navbar-brand mb-0 h1">📊 LIVE Analytics Dashboard</span>
                <div>
                    <a href="/" class="btn btn-light">Upload New File</a>
                </div>
            </div>
        </nav>
        
        <div class="container mt-4">
            <div class="alert alert-info">
                <h4>📈 Real Data Analytics</h4>
                <p><strong>File:</strong> {filename} | <strong>Total entries:</strong> {summary['total_feedback']}</p>
                <p><strong>This dashboard shows ACTUAL data from your uploaded file!</strong></p>
            </div>
            
            <!-- Summary Stats -->
            <div class="row mb-4">
                <div class="col-md-3">
                    <div class="card text-white bg-success">
                        <div class="card-body stat-card">
                            <div class="stat-number">{summary['positive_percentage']}%</div>
                            <p>Positive ({summary['positive_count']})</p>
                        </div>
                    </div>
                </div>
                <div class="col-md-3">
                    <div class="card text-white bg-warning">
                        <div class="card-body stat-card">
                            <div class="stat-number">{summary['neutral_percentage']}%</div>
                            <p>Neutral ({summary['neutral_count']})</p>
                        </div>
                    </div>
                </div>
                <div class="col-md-3">
                    <div class="card text-white bg-danger">
                        <div class="card-body stat-card">
                            <div class="stat-number">{summary['negative_percentage']}%</div>
                            <p>Negative ({summary['negative_count']})</p>
                        </div>
                    </div>
                </div>
                <div class="col-md-3">
                    <div class="card text-white bg-info">
                        <div class="card-body stat-card">
                            <div class="stat-number">{summary['total_feedback']}</div>
                            <p>Total Feedback</p>
                        </div>
                    </div>
                </div>
            </div>
            
            <!-- Charts -->
            <div class="row mb-4">
                <div class="col-md-6">
                    <div class="card">
                        <div class="card-header bg-primary text-white">
                            <h5>Sentiment Distribution</h5>
                        </div>
                        <div class="card-body text-center">
                            <img src="{pie_chart}" alt="Pie Chart" class="img-fluid">
                        </div>
                    </div>
                </div>
                <div class="col-md-6">
                    <div class="card">
                        <div class="card-header bg-primary text-white">
                            <h5>Sentiment Counts</h5>
                        </div>
                        <div class="card-body text-center">
                            <img src="{bar_chart}" alt="Bar Chart" class="img-fluid">
                        </div>
                    </div>
                </div>
            </div>
            
            <!-- Word Clouds -->
            <div class="row mb-4">
                <div class="col-md-4">
                    <div class="card">
                        <div class="card-header bg-success text-white">
                            <h5>Positive Feedback</h5>
                        </div>
                        <div class="card-body text-center">
                            <img src="{wordcloud_positive}" alt="Positive Word Cloud" class="img-fluid">
                        </div>
                    </div>
                </div>
                <div class="col-md-4">
                    <div class="card">
                        <div class="card-header bg-warning text-white">
                            <h5>Neutral Feedback</h5>
                        </div>
                        <div class="card-body text-center">
                            <img src="{wordcloud_neutral}" alt="Neutral Word Cloud" class="img-fluid">
                        </div>
                    </div>
                </div>
                <div class="col-md-4">
                    <div class="card">
                        <div class="card-header bg-danger text-white">
                            <h5>Negative Feedback</h5>
                        </div>
                        <div class="card-body text-center">
                            <img src="{wordcloud_negative}" alt="Negative Word Cloud" class="img-fluid">
                        </div>
                    </div>
                </div>
            </div>
            
            <!-- Data Table -->
            <div class="card">
                <div class="card-header">
                    <h5>Analyzed Feedback Data</h5>
                </div>
                <div class="card-body">
                    <div class="table-responsive">
                        <table class="table table-striped">
                            <thead>
                                <tr>
                                    <th>Feedback Text</th>
                                    <th>Sentiment</th>
                                </tr>
                            </thead>
                            <tbody>
                                {"".join([f'<tr><td>{row["feedback"]}</td><td><span class="badge bg-{"success" if row["sentiment_label"] == "Positive" else "danger" if row["sentiment_label"] == "Negative" else "warning"}">{row["sentiment_label"]}</span></td></tr>' for row in analyzed_df.head(10).to_dict('records')])}
                            </tbody>
                        </table>
                    </div>
                </div>
            </div>
        </div>
    </body>
    </html>
    '''

@app.route('/create-test-file')
def create_test_file():
    # Create a test CSV file
    test_data = {
        'feedback': [
            "EXCELLENT policy! This will greatly benefit our community and improve quality of life for everyone.",
            "TERRIBLE decision that will destroy small businesses and hurt our local economy significantly.",
            "Reasonable approach with good intentions, but needs more detailed implementation planning.",
            "Outstanding work by our representatives! This policy addresses all the key issues effectively.",
            "Complete disaster - ignores expert recommendations and will have negative consequences.",
            "Good balanced policy that considers different perspectives and stakeholder interests.",
            "Very disappointing approach that fails to address the root causes of the problem.",
            "Fantastic initiative that demonstrates forward-thinking leadership and vision.",
            "Poorly conceived policy that creates more problems than it actually solves.",
            "Moderate support for this proposal, though some aspects need further refinement."
        ]
    }
    
    df = pd.DataFrame(test_data)
    df.to_csv('test_feedback.csv', index=False)
    
    return '''
    <div class="alert alert-success">
        <h4>✅ Test file created: test_feedback.csv</h4>
        <p>Now upload this file through the main page!</p>
        <a href="/" class="btn btn-primary">Go Back</a>
    </div>
    '''

@app.route('/test-analyze')
def test_analyze():
    global CURRENT_ANALYSIS
    
    # Create test data directly
    test_data = {
        'feedback': [
            "EXCELLENT policy! This will greatly benefit our community.",
            "TERRIBLE decision that will destroy small businesses.",
            "Reasonable approach with good intentions.",
            "Outstanding work by our representatives!",
            "Complete disaster - ignores expert recommendations.",
            "Good balanced policy that considers different perspectives.",
            "Very disappointing approach.",
            "Fantastic initiative with great vision.",
            "Poorly conceived policy.",
            "Moderate support for this proposal."
        ]
    }
    
    df = pd.DataFrame(test_data)
    analyzed_df = analyzer.analyze_dataframe(df, 'feedback')
    
    total = len(analyzed_df)
    positive = len(analyzed_df[analyzed_df['sentiment_label'] == 'Positive'])
    negative = len(analyzed_df[analyzed_df['sentiment_label'] == 'Negative'])
    neutral = len(analyzed_df[analyzed_df['sentiment_label'] == 'Neutral'])
    
    CURRENT_ANALYSIS = {
        'analyzed_df': analyzed_df,
        'summary': {
            'total_feedback': total,
            'positive_count': positive,
            'negative_count': negative,
            'neutral_count': neutral,
            'positive_percentage': round((positive/total)*100, 2),
            'negative_percentage': round((negative/total)*100, 2),
            'neutral_percentage': round((neutral/total)*100, 2)
        },
        'filename': 'test_data.csv'
    }
    
    return '''
    <div class="alert alert-success">
        <h4>✅ Test data loaded successfully!</h4>
        <p>Test data has been analyzed and stored. Now visit the dashboard to see REAL analytics.</p>
        <a href="/dashboard" class="btn btn-success">View Dashboard</a>
        <a href="/" class="btn btn-primary">Upload Real File</a>
    </div>
    '''

if __name__ == '__main__':
    print("🚀 Starting FIXED Sentiment Analysis App...")
    print("📍 Open: http://localhost:5001")
    print("📊 This version WILL show your actual data!")
    app.run(debug=True, host='0.0.0.0', port=5001)