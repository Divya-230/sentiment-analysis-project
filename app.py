from flask import Flask, render_template, request, jsonify, send_file
import pandas as pd
import os
import io
from textblob import TextBlob
import matplotlib.pyplot as plt
import base64
from io import BytesIO
import numpy as np
from collections import Counter
import re
from datetime import datetime

app = Flask(__name__)
app.secret_key = 'professional-sentiment-analysis-2024'

# Global storage
current_data = None

class ProfessionalAnalyzer:
    def analyze_sentiment(self, text):
        analysis = TextBlob(str(text))
        polarity = analysis.sentiment.polarity
        subjectivity = analysis.sentiment.subjectivity
        
        if polarity > 0.2:
            return 1, 'Positive', polarity, subjectivity
        elif polarity < -0.2:
            return -1, 'Negative', polarity, subjectivity
        else:
            return 0, 'Neutral', polarity, subjectivity
    
    def analyze_dataframe(self, df, text_column='feedback'):
        df_analyzed = df.copy()
        sentiments = []
        labels = []
        polarities = []
        subjectivities = []
        
        for text in df_analyzed[text_column]:
            sentiment, label, polarity, subjectivity = self.analyze_sentiment(text)
            sentiments.append(sentiment)
            labels.append(label)
            polarities.append(polarity)
            subjectivities.append(subjectivity)
        
        df_analyzed['sentiment'] = sentiments
        df_analyzed['sentiment_label'] = labels
        df_analyzed['polarity'] = polarities
        df_analyzed['subjectivity'] = subjectivities
        return df_analyzed

class ProfessionalViz:
    def __init__(self):
        # Use a basic style that works with all matplotlib versions
        try:
            plt.style.use('ggplot')
        except:
            # If ggplot is not available, use default style
            pass
        
        self.colors = {
            'Positive': '#00D26A',
            'Neutral': '#FFB800', 
            'Negative': '#FF4757',
            'primary': '#2E86AB',
            'secondary': '#A23B72',
            'dark': '#1A1A2E'
        }
    
    def create_pie_chart(self, df):
        sentiment_counts = df['sentiment_label'].value_counts()
        
        fig, ax = plt.subplots(figsize=(8, 8))
        colors = [self.colors[sentiment] for sentiment in sentiment_counts.index]
        
        wedges, texts, autotexts = ax.pie(
            sentiment_counts.values,
            labels=sentiment_counts.index,
            colors=colors,
            autopct='%1.1f%%',
            startangle=90,
            explode=[0.05] * len(sentiment_counts),
            shadow=True,
            textprops={'fontsize': 12, 'weight': 'bold'}
        )
        
        ax.set_title('Sentiment Distribution', fontsize=16, fontweight='bold', pad=20)
        
        img_buffer = BytesIO()
        plt.savefig(img_buffer, format='png', bbox_inches='tight', dpi=120, facecolor='white')
        plt.close()
        
        img_buffer.seek(0)
        img_str = base64.b64encode(img_buffer.read()).decode()
        return f"data:image/png;base64,{img_str}"
    
    def create_bar_chart(self, df):
        sentiment_counts = df['sentiment_label'].value_counts()
        
        fig, ax = plt.subplots(figsize=(10, 6))
        colors = [self.colors[sentiment] for sentiment in sentiment_counts.index]
        
        bars = ax.bar(sentiment_counts.index, sentiment_counts.values, 
                     color=colors, alpha=0.8, edgecolor='white', linewidth=2)
        ax.set_title('Sentiment Analysis Results', fontsize=16, fontweight='bold', pad=20)
        ax.set_xlabel('Sentiment Category', fontweight='bold')
        ax.set_ylabel('Number of Responses', fontweight='bold')
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                   f'{int(height)}', ha='center', va='bottom', fontweight='bold', fontsize=12)
        
        ax.grid(axis='y', alpha=0.3)
        plt.xticks(fontweight='bold')
        
        img_buffer = BytesIO()
        plt.savefig(img_buffer, format='png', bbox_inches='tight', dpi=120, facecolor='white')
        plt.close()
        
        img_buffer.seek(0)
        img_str = base64.b64encode(img_buffer.read()).decode()
        return f"data:image/png;base64,{img_str}"
    
    def create_polarity_histogram(self, df):
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Create histogram of polarity scores
        n, bins, patches = ax.hist(df['polarity'], bins=20, alpha=0.7, 
                                  color=self.colors['primary'], edgecolor='white')
        
        ax.set_title('Distribution of Sentiment Polarity Scores', fontsize=16, fontweight='bold', pad=20)
        ax.set_xlabel('Polarity Score (-1 to 1)', fontweight='bold')
        ax.set_ylabel('Frequency', fontweight='bold')
        ax.grid(alpha=0.3)
        
        # Add vertical lines for sentiment thresholds
        ax.axvline(x=0.2, color='red', linestyle='--', alpha=0.7, label='Positive Threshold')
        ax.axvline(x=-0.2, color='red', linestyle='--', alpha=0.7, label='Negative Threshold')
        ax.legend()
        
        img_buffer = BytesIO()
        plt.savefig(img_buffer, format='png', bbox_inches='tight', dpi=120, facecolor='white')
        plt.close()
        
        img_buffer.seek(0)
        img_str = base64.b64encode(img_buffer.read()).decode()
        return f"data:image/png;base64,{img_str}"
    
    def create_timeseries_chart(self, df):
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Simulate time series data for demo
        dates = pd.date_range(start='2024-01-01', periods=len(df), freq='D')
        cumulative_positive = (df['sentiment_label'] == 'Positive').cumsum()
        cumulative_negative = (df['sentiment_label'] == 'Negative').cumsum()
        
        ax.plot(dates, cumulative_positive, label='Cumulative Positive', 
                color=self.colors['Positive'], linewidth=3)
        ax.plot(dates, cumulative_negative, label='Cumulative Negative', 
                color=self.colors['Negative'], linewidth=3)
        
        ax.set_title('Sentiment Trends Over Time', fontsize=16, fontweight='bold', pad=20)
        ax.set_xlabel('Date', fontweight='bold')
        ax.set_ylabel('Cumulative Count', fontweight='bold')
        ax.legend()
        ax.grid(alpha=0.3)
        plt.xticks(rotation=45)
        
        img_buffer = BytesIO()
        plt.savefig(img_buffer, format='png', bbox_inches='tight', dpi=120, facecolor='white')
        plt.close()
        
        img_buffer.seek(0)
        img_str = base64.b64encode(img_buffer.read()).decode()
        return f"data:image/png;base64,{img_str}"
    
    def create_top_words_chart(self, df, sentiment=None):
        if sentiment:
            texts = df[df['sentiment_label'] == sentiment]['feedback']
        else:
            texts = df['feedback']
        
        all_text = ' '.join(texts.astype(str)).lower()
        words = re.findall(r'\b[a-z]{3,15}\b', all_text)
        
        # Enhanced stopwords list
        stopwords = {'the', 'and', 'for', 'that', 'with', 'this', 'have', 'from', 'they', 'will', 
                    'their', 'has', 'been', 'are', 'what', 'were', 'your', 'there', 'about', 'which',
                    'when', 'would', 'could', 'should', 'been', 'were', 'them', 'then', 'than'}
        words = [word for word in words if word not in stopwords]
        
        word_freq = Counter(words).most_common(8)
        
        if not word_freq:
            return self._create_empty_chart("No significant words found")
        
        words, counts = zip(*word_freq)
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        color = self.colors.get(sentiment, self.colors['primary'])
        bars = ax.barh(words, counts, color=color, alpha=0.8, edgecolor='white', linewidth=2)
        
        ax.set_title(f'Most Frequent Words - {sentiment if sentiment else "All"} Feedback', 
                    fontsize=14, fontweight='bold', pad=20)
        ax.set_xlabel('Frequency', fontweight='bold')
        
        # Add value labels
        for bar in bars:
            width = bar.get_width()
            ax.text(width + 0.1, bar.get_y() + bar.get_height()/2, 
                   f'{int(width)}', ha='left', va='center', fontweight='bold')
        
        ax.grid(axis='x', alpha=0.3)
        plt.tight_layout()
        
        img_buffer = BytesIO()
        plt.savefig(img_buffer, format='png', bbox_inches='tight', dpi=120, facecolor='white')
        plt.close()
        
        img_buffer.seek(0)
        img_str = base64.b64encode(img_buffer.read()).decode()
        return f"data:image/png;base64,{img_str}"
    
    def create_sentiment_breakdown(self, df):
        # Create a detailed breakdown chart
        sentiment_stats = df.groupby('sentiment_label').agg({
            'polarity': ['mean', 'std'],
            'subjectivity': 'mean'
        }).round(3)
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        # Polarity by sentiment
        sentiments = sentiment_stats.index
        polarity_means = sentiment_stats[('polarity', 'mean')]
        
        bars1 = ax1.bar(sentiments, polarity_means, 
                       color=[self.colors[s] for s in sentiments],
                       alpha=0.8, edgecolor='white', linewidth=2)
        ax1.set_title('Average Polarity by Sentiment', fontweight='bold')
        ax1.set_ylabel('Polarity Score', fontweight='bold')
        
        for bar in bars1:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{height:.3f}', ha='center', va='bottom', fontweight='bold')
        
        # Subjectivity by sentiment
        subjectivity_means = sentiment_stats[('subjectivity', 'mean')]
        bars2 = ax2.bar(sentiments, subjectivity_means,
                       color=[self.colors[s] for s in sentiments],
                       alpha=0.8, edgecolor='white', linewidth=2)
        ax2.set_title('Average Subjectivity by Sentiment', fontweight='bold')
        ax2.set_ylabel('Subjectivity Score', fontweight='bold')
        
        for bar in bars2:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{height:.3f}', ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        
        img_buffer = BytesIO()
        plt.savefig(img_buffer, format='png', bbox_inches='tight', dpi=120, facecolor='white')
        plt.close()
        
        img_buffer.seek(0)
        img_str = base64.b64encode(img_buffer.read()).decode()
        return f"data:image/png;base64,{img_str}"
    
    def _create_empty_chart(self, message):
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.text(0.5, 0.5, message, ha='center', va='center', 
                transform=ax.transAxes, fontsize=14, style='italic')
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')
        
        img_buffer = BytesIO()
        plt.savefig(img_buffer, format='png', bbox_inches='tight', dpi=120, facecolor='white')
        plt.close()
        
        img_buffer.seek(0)
        img_str = base64.b64encode(img_buffer.read()).decode()
        return f"data:image/png;base64,{img_str}"

analyzer = ProfessionalAnalyzer()
viz = ProfessionalViz()

@app.route('/')
def home():
    return '''
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Policy Feedback Sentiment Analysis</title>
        <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/css/bootstrap.min.css" rel="stylesheet">
        <link href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css" rel="stylesheet">
        <style>
            :root {
                --primary: #2E86AB;
                --secondary: #A23B72;
                --success: #00D26A;
                --warning: #FFB800;
                --danger: #FF4757;
                --dark: #1A1A2E;
                --light: #F8F9FA;
            }
            
            body {
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                min-height: 100vh;
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            }
            
            .glass-card {
                background: rgba(255, 255, 255, 0.95);
                backdrop-filter: blur(10px);
                border-radius: 15px;
                border: 1px solid rgba(255, 255, 255, 0.2);
                box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
            }
            
            .stat-card {
                border-radius: 12px;
                border: none;
                transition: transform 0.3s ease, box-shadow 0.3s ease;
            }
            
            .stat-card:hover {
                transform: translateY(-5px);
                box-shadow: 0 12px 20px rgba(0, 0, 0, 0.15);
            }
            
            .dashboard-header {
                background: linear-gradient(135deg, var(--dark) 0%, var(--primary) 100%);
                color: white;
                padding: 3rem 0;
                margin-bottom: 2rem;
                border-radius: 0 0 30px 30px;
            }
            
            .chart-container {
                background: white;
                border-radius: 12px;
                padding: 20px;
                margin-bottom: 20px;
                box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
                border: 1px solid rgba(0, 0, 0, 0.05);
            }
            
            .btn-gradient {
                background: linear-gradient(135deg, var(--primary), var(--secondary));
                border: none;
                color: white;
                font-weight: 600;
                padding: 12px 30px;
                border-radius: 8px;
                transition: all 0.3s ease;
            }
            
            .btn-gradient:hover {
                transform: translateY(-2px);
                box-shadow: 0 6px 12px rgba(0, 0, 0, 0.15);
                color: white;
            }
            
            .feature-icon {
                width: 60px;
                height: 60px;
                background: linear-gradient(135deg, var(--primary), var(--secondary));
                border-radius: 12px;
                display: flex;
                align-items: center;
                justify-content: center;
                margin: 0 auto 15px;
                color: white;
                font-size: 24px;
            }
            
            .nav-tabs .nav-link {
                border: none;
                color: var(--dark);
                font-weight: 600;
                padding: 12px 25px;
            }
            
            .nav-tabs .nav-link.active {
                background: linear-gradient(135deg, var(--primary), var(--secondary));
                color: white;
                border-radius: 8px;
            }
        </style>
    </head>
    <body>
        <div class="dashboard-header">
            <div class="container">
                <div class="row align-items-center">
                    <div class="col-md-8">
                        <h1 class="display-4 fw-bold"><i class="fas fa-chart-line me-3"></i>Policy Feedback Analytics</h1>
                        <p class="lead">Advanced Sentiment Analysis for Public Policy Evaluation</p>
                    </div>
                    <div class="col-md-4 text-end">
                        <div class="glass-card p-3 d-inline-block">
                            <small class="text-muted">Real-time Analysis</small>
                            <div class="h5 mb-0" id="liveClock">Loading...</div>
                        </div>
                    </div>
                </div>
            </div>
        </div>

        <div class="container">
            <div class="row">
                <div class="col-12">
                    <div class="glass-card p-4 mb-4">
                        <div class="row">
                            <div class="col-md-8">
                                <h3><i class="fas fa-upload me-2"></i>Upload Feedback Data</h3>
                                <p class="text-muted">Analyze public policy feedback with advanced sentiment analysis</p>
                            </div>
                            <div class="col-md-4 text-end">
                                <span class="badge bg-success fs-6">AI-Powered</span>
                            </div>
                        </div>
                        
                        <form action="/upload" method="post" enctype="multipart/form-data" class="mt-4">
                            <div class="row">
                                <div class="col-md-6">
                                    <div class="mb-3">
                                        <label class="form-label fw-bold">Upload CSV File</label>
                                        <input class="form-control" type="file" name="file" accept=".csv" required>
                                        <div class="form-text">File should contain a column with feedback text</div>
                                    </div>
                                </div>
                                <div class="col-md-4">
                                    <div class="mb-3">
                                        <label class="form-label fw-bold">Text Column Name</label>
                                        <input type="text" class="form-control" name="text_column" value="feedback" required>
                                    </div>
                                </div>
                                <div class="col-md-2">
                                    <div class="mb-3">
                                        <label class="form-label">&nbsp;</label>
                                        <button type="submit" class="btn btn-gradient w-100">
                                            <i class="fas fa-rocket me-2"></i>Analyze
                                        </button>
                                    </div>
                                </div>
                            </div>
                        </form>
                        
                        <div class="row mt-4">
                            <div class="col-md-3 text-center">
                                <div class="feature-icon">
                                    <i class="fas fa-brain"></i>
                                </div>
                                <h6>AI Analysis</h6>
                                <small class="text-muted">Machine Learning Powered</small>
                            </div>
                            <div class="col-md-3 text-center">
                                <div class="feature-icon">
                                    <i class="fas fa-chart-pie"></i>
                                </div>
                                <h6>Multiple Charts</h6>
                                <small class="text-muted">Comprehensive Visualization</small>
                            </div>
                            <div class="col-md-3 text-center">
                                <div class="feature-icon">
                                    <i class="fas fa-bolt"></i>
                                </div>
                                <h6>Real-time</h6>
                                <small class="text-muted">Instant Results</small>
                            </div>
                            <div class="col-md-3 text-center">
                                <div class="feature-icon">
                                    <i class="fas fa-download"></i>
                                </div>
                                <h6>Export Ready</h6>
                                <small class="text-muted">Professional Reports</small>
                            </div>
                        </div>
                    </div>
                </div>
            </div>

            <div class="row">
                <div class="col-md-4">
                    <a href="/test" class="btn btn-outline-light w-100 mb-3">
                        <i class="fas fa-vial me-2"></i>Load Test Data
                    </a>
                </div>
                <div class="col-md-4">
                    <a href="/dashboard" class="btn btn-gradient w-100 mb-3">
                        <i class="fas fa-tachometer-alt me-2"></i>View Dashboard
                    </a>
                </div>
                <div class="col-md-4">
                    <a href="/create-sample" class="btn btn-outline-light w-100 mb-3">
                        <i class="fas fa-file-csv me-2"></i>Create Sample
                    </a>
                </div>
            </div>
        </div>

        <script>
            function updateClock() {
                const now = new Date();
                document.getElementById('liveClock').textContent = 
                    now.toLocaleDateString() + ' ' + now.toLocaleTimeString();
            }
            setInterval(updateClock, 1000);
            updateClock();
        </script>
    </body>
    </html>
    '''

@app.route('/upload', methods=['POST'])
def upload_file():
    global current_data
    
    try:
        if 'file' not in request.files:
            return "No file uploaded", 400
        
        file = request.files['file']
        text_column = request.form.get('text_column', 'feedback')
        
        if file.filename == '':
            return "No file selected", 400
        
        print(f"📁 Processing: {file.filename}")
        
        # Read CSV
        df = pd.read_csv(file)
        print(f"✅ Loaded {len(df)} rows. Columns: {list(df.columns)}")
        
        # Check if text column exists
        if text_column not in df.columns:
            return f"Error: Column '{text_column}' not found. Available: {list(df.columns)}", 400
        
        # Analyze sentiment
        analyzed_df = analyzer.analyze_dataframe(df, text_column)
        
        # Calculate comprehensive summary
        total = len(analyzed_df)
        positive = len(analyzed_df[analyzed_df['sentiment_label'] == 'Positive'])
        negative = len(analyzed_df[analyzed_df['sentiment_label'] == 'Negative'])
        neutral = len(analyzed_df[analyzed_df['sentiment_label'] == 'Neutral'])
        
        avg_polarity = analyzed_df['polarity'].mean()
        avg_subjectivity = analyzed_df['subjectivity'].mean()
        
        summary = {
            'total_feedback': total,
            'positive_count': positive,
            'negative_count': negative,
            'neutral_count': neutral,
            'positive_percentage': round((positive/total)*100, 2),
            'negative_percentage': round((negative/total)*100, 2),
            'neutral_percentage': round((neutral/total)*100, 2),
            'avg_polarity': round(avg_polarity, 3),
            'avg_subjectivity': round(avg_subjectivity, 3),
            'analysis_timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        
        print(f"📊 Analysis Complete: {summary}")
        
        # STORE DATA GLOBALLY
        current_data = {
            'analyzed_df': analyzed_df,
            'summary': summary,
            'filename': file.filename,
            'upload_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        
        return f'''
        <!DOCTYPE html>
        <html>
        <head>
            <title>Analysis Complete</title>
            <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/css/bootstrap.min.css" rel="stylesheet">
            <link href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css" rel="stylesheet">
            <style>
                body {{
                    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                    min-height: 100vh;
                    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                }}
                .glass-card {{
                    background: rgba(255, 255, 255, 0.95);
                    backdrop-filter: blur(10px);
                    border-radius: 15px;
                    border: 1px solid rgba(255, 255, 255, 0.2);
                }}
            </style>
        </head>
        <body>
            <div class="container mt-5">
                <div class="glass-card p-5">
                    <div class="text-center mb-4">
                        <i class="fas fa-check-circle text-success" style="font-size: 4rem;"></i>
                        <h2 class="mt-3">Analysis Complete!</h2>
                        <p class="lead text-muted">Your feedback data has been successfully analyzed</p>
                    </div>
                    
                    <div class="row mb-4">
                        <div class="col-md-6">
                            <div class="card border-0 shadow-sm">
                                <div class="card-body text-center">
                                    <h5><i class="fas fa-file-alt me-2"></i>File Information</h5>
                                    <p class="mb-1"><strong>Filename:</strong> {file.filename}</p>
                                    <p class="mb-1"><strong>Total Entries:</strong> {total}</p>
                                    <p class="mb-0"><strong>Analysis Time:</strong> {summary['analysis_timestamp']}</p>
                                </div>
                            </div>
                        </div>
                        <div class="col-md-6">
                            <div class="card border-0 shadow-sm">
                                <div class="card-body text-center">
                                    <h5><i class="fas fa-chart-bar me-2"></i>Quick Stats</h5>
                                    <p class="mb-1"><strong>Positive:</strong> {positive} ({summary['positive_percentage']}%)</p>
                                    <p class="mb-1"><strong>Neutral:</strong> {neutral} ({summary['neutral_percentage']}%)</p>
                                    <p class="mb-0"><strong>Negative:</strong> {negative} ({summary['negative_percentage']}%)</p>
                                </div>
                            </div>
                        </div>
                    </div>
                    
                    <div class="text-center">
                        <a href="/dashboard" class="btn btn-primary btn-lg px-5">
                            <i class="fas fa-tachometer-alt me-2"></i>View Detailed Dashboard
                        </a>
                        <a href="/" class="btn btn-outline-secondary btn-lg ms-2">
                            <i class="fas fa-upload me-2"></i>Upload New File
                        </a>
                    </div>
                </div>
            </div>
        </body>
        </html>
        '''
        
    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        print(f"❌ Error: {str(e)}")
        return f'''
        <div class="alert alert-danger">
            <h4><i class="fas fa-exclamation-triangle"></i> Error Processing File</h4>
            <p>{str(e)}</p>
            <a href="/" class="btn btn-primary">Try Again</a>
        </div>
        ''', 500

@app.route('/dashboard')
def dashboard():
    global current_data
    
    if current_data is None:
        return '''
        <!DOCTYPE html>
        <html>
        <head>
            <title>Dashboard - No Data</title>
            <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/css/bootstrap.min.css" rel="stylesheet">
            <link href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css" rel="stylesheet">
            <style>
                body {
                    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                    min-height: 100vh;
                    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                }
                .glass-card {
                    background: rgba(255, 255, 255, 0.95);
                    backdrop-filter: blur(10px);
                    border-radius: 15px;
                }
            </style>
        </head>
        <body>
            <div class="container mt-5">
                <div class="glass-card p-5 text-center">
                    <i class="fas fa-chart-bar text-muted" style="font-size: 4rem;"></i>
                    <h2 class="mt-3">No Data Available</h2>
                    <p class="lead text-muted">Please upload a feedback file to view analytics</p>
                    <a href="/" class="btn btn-primary btn-lg">
                        <i class="fas fa-upload me-2"></i>Upload File
                    </a>
                    <a href="/test" class="btn btn-success btn-lg ms-2">
                        <i class="fas fa-vial me-2"></i>Load Test Data
                    </a>
                </div>
            </div>
        </body>
        </html>
        '''
    
    # WE HAVE DATA - SHOW PROFESSIONAL DASHBOARD
    analyzed_df = current_data['analyzed_df']
    summary = current_data['summary']
    filename = current_data['filename']
    
    # Generate all visualizations
    pie_chart = viz.create_pie_chart(analyzed_df)
    bar_chart = viz.create_bar_chart(analyzed_df)
    polarity_histogram = viz.create_polarity_histogram(analyzed_df)
    timeseries_chart = viz.create_timeseries_chart(analyzed_df)
    top_words_positive = viz.create_top_words_chart(analyzed_df, 'Positive')
    top_words_negative = viz.create_top_words_chart(analyzed_df, 'Negative')
    top_words_neutral = viz.create_top_words_chart(analyzed_df, 'Neutral')
    sentiment_breakdown = viz.create_sentiment_breakdown(analyzed_df)
    
    return f'''
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Professional Dashboard - {filename}</title>
        <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/css/bootstrap.min.css" rel="stylesheet">
        <link href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css" rel="stylesheet">
        <style>
            :root {{
                --primary: #2E86AB;
                --secondary: #A23B72;
                --success: #00D26A;
                --warning: #FFB800;
                --danger: #FF4757;
                --dark: #1A1A2E;
            }}
            
            body {{
                background: #f8f9fa;
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            }}
            
            .dashboard-header {{
                background: linear-gradient(135deg, var(--dark) 0%, var(--primary) 100%);
                color: white;
                padding: 2.5rem 0;
                margin-bottom: 2rem;
                border-radius: 0 0 25px 25px;
            }}
            
            .stat-card {{
                border-radius: 12px;
                border: none;
                transition: transform 0.3s ease, box-shadow 0.3s ease;
                margin-bottom: 20px;
            }}
            
            .stat-card:hover {{
                transform: translateY(-5px);
                box-shadow: 0 8px 25px rgba(0,0,0,0.15);
            }}
            
            .chart-container {{
                background: white;
                border-radius: 12px;
                padding: 25px;
                margin-bottom: 25px;
                box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
                border: 1px solid #e9ecef;
            }}
            
            .nav-tabs .nav-link {{
                border: none;
                color: var(--dark);
                font-weight: 600;
                padding: 12px 25px;
            }}
            
            .nav-tabs .nav-link.active {{
                background: linear-gradient(135deg, var(--primary), var(--secondary));
                color: white;
                border-radius: 8px;
            }}
            
            .metric-value {{
                font-size: 2.2rem;
                font-weight: 700;
                margin-bottom: 0;
            }}
            
            .metric-label {{
                font-size: 0.9rem;
                color: #6c757d;
                font-weight: 600;
            }}
        </style>
    </head>
    <body>
        <!-- Dashboard Header -->
        <div class="dashboard-header">
            <div class="container">
                <div class="row align-items-center">
                    <div class="col-md-8">
                        <h1 class="display-5 fw-bold"><i class="fas fa-tachometer-alt me-3"></i>Analytics Dashboard</h1>
                        <p class="lead mb-0">Comprehensive sentiment analysis for policy feedback evaluation</p>
                    </div>
                    <div class="col-md-4 text-end">
                        <div class="bg-white bg-opacity-10 p-3 rounded d-inline-block">
                            <small class="text-white-50">Analyzing</small>
                            <div class="h6 mb-0 text-white">{filename}</div>
                        </div>
                    </div>
                </div>
            </div>
        </div>

        <div class="container">
            <!-- Quick Stats Row -->
            <div class="row mb-4">
                <div class="col-md-3">
                    <div class="stat-card text-white bg-success">
                        <div class="card-body text-center p-4">
                            <i class="fas fa-smile fa-2x mb-3"></i>
                            <div class="metric-value">{summary['positive_percentage']}%</div>
                            <div class="metric-label">POSITIVE FEEDBACK</div>
                            <small>{summary['positive_count']} responses</small>
                        </div>
                    </div>
                </div>
                <div class="col-md-3">
                    <div class="stat-card text-white bg-warning">
                        <div class="card-body text-center p-4">
                            <i class="fas fa-meh fa-2x mb-3"></i>
                            <div class="metric-value">{summary['neutral_percentage']}%</div>
                            <div class="metric-label">NEUTRAL FEEDBACK</div>
                            <small>{summary['neutral_count']} responses</small>
                        </div>
                    </div>
                </div>
                <div class="col-md-3">
                    <div class="stat-card text-white bg-danger">
                        <div class="card-body text-center p-4">
                            <i class="fas fa-frown fa-2x mb-3"></i>
                            <div class="metric-value">{summary['negative_percentage']}%</div>
                            <div class="metric-label">NEGATIVE FEEDBACK</div>
                            <small>{summary['negative_count']} responses</small>
                        </div>
                    </div>
                </div>
                <div class="col-md-3">
                    <div class="stat-card text-white bg-primary">
                        <div class="card-body text-center p-4">
                            <i class="fas fa-chart-line fa-2x mb-3"></i>
                            <div class="metric-value">{summary['total_feedback']}</div>
                            <div class="metric-label">TOTAL RESPONSES</div>
                            <small>Analyzed</small>
                        </div>
                    </div>
                </div>
            </div>

            <!-- Navigation Tabs -->
            <ul class="nav nav-tabs mb-4" id="dashboardTabs" role="tablist">
                <li class="nav-item" role="presentation">
                    <button class="nav-link active" id="overview-tab" data-bs-toggle="tab" data-bs-target="#overview" type="button" role="tab">
                        <i class="fas fa-chart-pie me-2"></i>Overview
                    </button>
                </li>
                <li class="nav-item" role="presentation">
                    <button class="nav-link" id="analysis-tab" data-bs-toggle="tab" data-bs-target="#analysis" type="button" role="tab">
                        <i class="fas fa-chart-bar me-2"></i>Detailed Analysis
                    </button>
                </li>
                <li class="nav-item" role="presentation">
                    <button class="nav-link" id="words-tab" data-bs-toggle="tab" data-bs-target="#words" type="button" role="tab">
                        <i class="fas fa-font me-2"></i>Text Analysis
                    </button>
                </li>
            </ul>

            <!-- Tab Content -->
            <div class="tab-content" id="dashboardTabsContent">
                <!-- Overview Tab -->
                <div class="tab-pane fade show active" id="overview" role="tabpanel">
                    <div class="row">
                        <div class="col-md-6">
                            <div class="chart-container">
                                <h5><i class="fas fa-chart-pie me-2"></i>Sentiment Distribution</h5>
                                <img src="{pie_chart}" alt="Sentiment Pie Chart" class="img-fluid w-100">
                            </div>
                        </div>
                        <div class="col-md-6">
                            <div class="chart-container">
                                <h5><i class="fas fa-chart-bar me-2"></i>Sentiment Analysis Results</h5>
                                <img src="{bar_chart}" alt="Sentiment Bar Chart" class="img-fluid w-100">
                            </div>
                        </div>
                    </div>
                    <div class="row">
                        <div class="col-md-6">
                            <div class="chart-container">
                                <h5><i class="fas fa-wave-square me-2"></i>Polarity Distribution</h5>
                                <img src="{polarity_histogram}" alt="Polarity Histogram" class="img-fluid w-100">
                            </div>
                        </div>
                        <div class="col-md-6">
                            <div class="chart-container">
                                <h5><i class="fas fa-trend-up me-2"></i>Sentiment Trends</h5>
                                <img src="{timeseries_chart}" alt="Time Series Chart" class="img-fluid w-100">
                            </div>
                        </div>
                    </div>
                </div>

                <!-- Detailed Analysis Tab -->
                <div class="tab-pane fade" id="analysis" role="tabpanel">
                    <div class="row">
                        <div class="col-12">
                            <div class="chart-container">
                                <h5><i class="fas fa-chart-line me-2"></i>Sentiment Metrics Breakdown</h5>
                                <img src="{sentiment_breakdown}" alt="Sentiment Breakdown" class="img-fluid w-100">
                            </div>
                        </div>
                    </div>
                    <div class="row">
                        <div class="col-md-6">
                            <div class="chart-container">
                                <h5><i class="fas fa-smile me-2 text-success"></i>Positive Feedback Analysis</h5>
                                <img src="{top_words_positive}" alt="Positive Words" class="img-fluid w-100">
                            </div>
                        </div>
                        <div class="col-md-6">
                            <div class="chart-container">
                                <h5><i class="fas fa-frown me-2 text-danger"></i>Negative Feedback Analysis</h5>
                                <img src="{top_words_negative}" alt="Negative Words" class="img-fluid w-100">
                            </div>
                        </div>
                    </div>
                </div>

                <!-- Text Analysis Tab -->
                <div class="tab-pane fade" id="words" role="tabpanel">
                    <div class="row">
                        <div class="col-md-4">
                            <div class="chart-container">
                                <h5 class="text-success"><i class="fas fa-smile me-2"></i>Positive Words</h5>
                                <img src="{top_words_positive}" alt="Positive Words" class="img-fluid w-100">
                            </div>
                        </div>
                        <div class="col-md-4">
                            <div class="chart-container">
                                <h5 class="text-warning"><i class="fas fa-meh me-2"></i>Neutral Words</h5>
                                <img src="{top_words_neutral}" alt="Neutral Words" class="img-fluid w-100">
                            </div>
                        </div>
                        <div class="col-md-4">
                            <div class="chart-container">
                                <h5 class="text-danger"><i class="fas fa-frown me-2"></i>Negative Words</h5>
                                <img src="{top_words_negative}" alt="Negative Words" class="img-fluid w-100">
                            </div>
                        </div>
                    </div>
                </div>
            </div>

            <!-- Action Buttons -->
            <div class="text-center mt-4 mb-5">
                <a href="/" class="btn btn-primary btn-lg px-5">
                    <i class="fas fa-upload me-2"></i>Upload New File
                </a>
                <a href="/test" class="btn btn-success btn-lg px-5 ms-2">
                    <i class="fas fa-vial me-2"></i>Load Test Data
                </a>
            </div>
        </div>

        <!-- Bootstrap JS -->
        <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/js/bootstrap.bundle.min.js"></script>
        <script>
            // Add some interactivity
            document.addEventListener('DOMContentLoaded', function() {{
                console.log('Professional Dashboard Loaded Successfully!');
            }});
        </script>
    </body>
    </html>
    '''

@app.route('/test')
def test_data():
    global current_data
    
    # Create comprehensive test data
    test_df = pd.DataFrame({
        'feedback': [
            "EXCELLENT policy implementation! This will significantly improve public services and benefit our community tremendously.",
            "TERRIBLE decision that will negatively impact small businesses and local economy. Very disappointing approach.",
            "Reasonable policy framework with good intentions, though implementation details need more careful planning.",
            "Outstanding work by our government representatives! This initiative addresses critical community needs effectively.",
            "Complete disaster - ignores expert recommendations and will likely create more problems than solutions.",
            "Well-balanced policy approach that considers various stakeholder perspectives and long-term impacts.",
            "Very disappointing policy direction that fails to address the core issues facing our community.",
            "Fantastic initiative demonstrating innovative thinking and strong commitment to public welfare.",
            "Poorly conceived policy framework that lacks proper consultation and risk assessment.",
            "Moderate support for this proposal, though several aspects require further refinement and clarification."
        ]
    })
    
    analyzed_df = analyzer.analyze_dataframe(test_df)
    
    total = len(analyzed_df)
    positive = len(analyzed_df[analyzed_df['sentiment_label'] == 'Positive'])
    negative = len(analyzed_df[analyzed_df['sentiment_label'] == 'Negative'])
    neutral = len(analyzed_df[analyzed_df['sentiment_label'] == 'Neutral'])
    
    current_data = {
        'analyzed_df': analyzed_df,
        'summary': {
            'total_feedback': total,
            'positive_count': positive,
            'negative_count': negative,
            'neutral_count': neutral,
            'positive_percentage': round((positive/total)*100, 2),
            'negative_percentage': round((negative/total)*100, 2),
            'neutral_percentage': round((neutral/total)*100, 2),
            'avg_polarity': round(analyzed_df['polarity'].mean(), 3),
            'avg_subjectivity': round(analyzed_df['subjectivity'].mean(), 3),
            'analysis_timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        },
        'filename': 'comprehensive_test_data.csv',
        'upload_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }
    
    return '''
    <!DOCTYPE html>
    <html>
    <head>
        <title>Test Data Loaded</title>
        <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/css/bootstrap.min.css" rel="stylesheet">
        <link href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css" rel="stylesheet">
        <style>
            body {
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                min-height: 100vh;
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            }
            .glass-card {
                background: rgba(255, 255, 255, 0.95);
                backdrop-filter: blur(10px);
                border-radius: 15px;
            }
        </style>
    </head>
    <body>
        <div class="container mt-5">
            <div class="glass-card p-5 text-center">
                <i class="fas fa-check-circle text-success" style="font-size: 4rem;"></i>
                <h2 class="mt-3">Test Data Loaded Successfully!</h2>
                <p class="lead text-muted">Comprehensive test dataset has been analyzed and is ready for exploration.</p>
                <div class="mt-4">
                    <a href="/dashboard" class="btn btn-primary btn-lg px-5">
                        <i class="fas fa-tachometer-alt me-2"></i>Explore Dashboard
                    </a>
                    <a href="/" class="btn btn-outline-secondary btn-lg ms-2">
                        <i class="fas fa-upload me-2"></i>Upload Real Data
                    </a>
                </div>
            </div>
        </div>
    </body>
    </html>
    '''

@app.route('/create-sample')
def create_sample():
    # Create professional sample CSV file
    sample_data = {
        'feedback': [
            "This policy represents a significant step forward in addressing urban infrastructure challenges with innovative solutions.",
            "Concerning approach that may disproportionately affect low-income communities without adequate support mechanisms.",
            "Well-researched policy framework that demonstrates thorough analysis of stakeholder needs and potential impacts.",
            "Disappointing lack of consultation with industry experts, potentially undermining policy effectiveness.",
            "Comprehensive strategy that balances economic growth with environmental sustainability considerations.",
            "Implementation timeline appears overly ambitious without sufficient resource allocation planning.",
            "Strong alignment with international best practices and evidence-based policy development approaches.",
            "Inadequate consideration of rural community needs in what appears to be an urban-focused initiative.",
            "Robust monitoring and evaluation framework that ensures accountability and continuous improvement.",
            "Potential regulatory overreach that may stifle innovation and private sector participation."
        ]
    }
    
    df = pd.DataFrame(sample_data)
    df.to_csv('professional_policy_feedback.csv', index=False)
    
    return '''
    <!DOCTYPE html>
    <html>
    <head>
        <title>Sample Created</title>
        <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/css/bootstrap.min.css" rel="stylesheet">
        <link href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css" rel="stylesheet">
        <style>
            body {
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                min-height: 100vh;
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            }
            .glass-card {
                background: rgba(255, 255, 255, 0.95);
                backdrop-filter: blur(10px);
                border-radius: 15px;
            }
        </style>
    </head>
    <body>
        <div class="container mt-5">
            <div class="glass-card p-5 text-center">
                <i class="fas fa-file-csv text-info" style="font-size: 4rem;"></i>
                <h2 class="mt-3">Professional Sample Created</h2>
                <p class="lead text-muted">Sample file 'professional_policy_feedback.csv' has been generated with comprehensive policy feedback data.</p>
                <div class="mt-4">
                    <a href="/" class="btn btn-primary btn-lg px-5">
                        <i class="fas fa-upload me-2"></i>Upload Sample File
                    </a>
                    <a href="/test" class="btn btn-success btn-lg ms-2">
                        <i class="fas fa-vial me-2"></i>Load Test Data
                    </a>
                </div>
            </div>
        </div>
    </body>
    </html>
    '''

if __name__ == '__main__':
    print("🚀 Starting PROFESSIONAL Sentiment Analysis App...")
    print("📍 Open: http://localhost:5000")
    print("🎨 Professional UI with Multiple Visualizations")
    print("📊 Comprehensive Analytics Dashboard")
    print("✅ COMPATIBLE with Python 3.6+")
    print("🔥 Project-Ready Implementation!")
    app.run(debug=True, host='0.0.0.0', port=5000)