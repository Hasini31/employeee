from flask import Flask, request, jsonify
from flask_cors import CORS
import sqlite3
import joblib
import os
import json
import re
from textblob import TextBlob
from datetime import datetime, timedelta
import numpy as np
import hashlib
import secrets
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

import google.generativeai as genai
from dotenv import load_dotenv

import matplotlib
matplotlib.use('Agg')  # Headless backend
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import io
import base64

# Load environment variables
load_dotenv()

app = Flask(__name__)
CORS(app)

# Database path
DB_PATH = os.path.join(os.path.dirname(__file__), 'database.db')
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'burnout_model.pkl')

# Simple session store (in production, use Redis or similar)
sessions = {}

# Configure Gemini API
genai.configure(api_key=os.getenv('GEMINI_API_KEY'))


# --- Auth Utilities ---

def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

def generate_token():
    return secrets.token_hex(32)

def verify_session(token, role='employee'):
    if token in sessions:
        session = sessions[token]
        if session['role'] == role or role == 'any':
            return session
    return None


# --- Database Setup ---

def init_db():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    cursor.execute('''
        CREATE TABLE IF NOT EXISTS employees (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            email TEXT UNIQUE NOT NULL,
            password TEXT NOT NULL,
            name TEXT NOT NULL,
            department TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')

    cursor.execute('''
        CREATE TABLE IF NOT EXISTS managers (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            email TEXT UNIQUE NOT NULL,
            password TEXT NOT NULL,
            name TEXT NOT NULL,
            department TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')

    cursor.execute('''
        CREATE TABLE IF NOT EXISTS password_resets (
            token TEXT PRIMARY KEY,
            email TEXT NOT NULL,
            expires_at TIMESTAMP NOT NULL
        )
    ''')

    cursor.execute('''
        CREATE TABLE IF NOT EXISTS records (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            employee_id INTEGER,
            employee_name TEXT,
            mood TEXT,
            work_hours REAL,
            fatigue INTEGER,
            experience REAL,
            feedback TEXT,
            sentiment TEXT,
            sentiment_score REAL,
            burnout_score REAL,
            burnout_level TEXT,
            suggestions TEXT,
            week_number INTEGER,
            day_of_week INTEGER,
            weekly_trend TEXT,
            submission_date TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (employee_id) REFERENCES employees(id)
        )
    ''')

    # Migrations: add columns if they don't exist
    for col_def in [
        ("submission_date", "TEXT"),
        ("sleep_hours", "REAL"),
        ("stress_level", "INTEGER")
    ]:
        try:
            cursor.execute(f"ALTER TABLE records ADD COLUMN {col_def[0]} {col_def[1]}")
        except Exception:
            pass  # Column already exists

    # Startup data integrity cleanup
    cursor.execute("DELETE FROM records WHERE employee_id IS NULL")

    # Backfill submission_date for existing records
    cursor.execute("""
        UPDATE records 
        SET submission_date = date(created_at)
        WHERE submission_date IS NULL OR submission_date = ''
    """)

    # Default manager
    try:
        cursor.execute('''
            INSERT OR IGNORE INTO managers (email, password, name, department)
            VALUES (?, ?, ?, ?)
        ''', ('admin@company.com', hash_password('admin123'), 'Admin Manager', 'HR'))
    except Exception:
        pass

    conn.commit()
    conn.close()


def load_model():
    if os.path.exists(MODEL_PATH):
        return joblib.load(MODEL_PATH)
    return None


# --- Sentiment ---

# Burnout-specific keyword dictionaries to supplement TextBlob
_NEGATIVE_KEYWORDS = {
    'exhausted', 'exhausting', 'drained', 'burnt', 'burned', 'burnout',
    'overwhelmed', 'overwhelm', 'stressed', 'stressful', 'stress',
    'tired', 'fatigue', 'fatigued', 'anxious', 'anxiety', 'depressed',
    'depression', 'hopeless', 'helpless', 'miserable', 'awful', 'terrible',
    'horrible', 'unbearable', 'struggling', 'struggle', 'frustrated',
    'frustration', 'demotivated', 'unmotivated', 'overloaded', 'overworked',
    'sleepless', 'insomnia', 'bored', 'disengaged', 'irritated', 'angry',
    'resentful', 'dejected', 'broken', 'lonely', 'isolated', 'numb'
}

_POSITIVE_KEYWORDS = {
    'happy', 'great', 'excellent', 'amazing', 'fantastic', 'motivated',
    'energized', 'refreshed', 'productive', 'focused', 'excited', 'positive',
    'good', 'wonderful', 'calm', 'relaxed', 'rested', 'recovered', 'better',
    'improving', 'confident', 'satisfied', 'engaged', 'balanced', 'hopeful'
}

def analyze_sentiment(text):
    if not text or text.strip() == '':
        return 'Neutral', 0.0

    # Keyword-based detection (burnout domain boost)
    words = set(text.lower().split())
    neg_hits = words & _NEGATIVE_KEYWORDS
    pos_hits = words & _POSITIVE_KEYWORDS

    # TextBlob polarity
    blob = TextBlob(text)
    polarity = blob.sentiment.polarity

    # Boost polarity based on keyword matches
    if neg_hits:
        polarity = min(polarity, -0.3)  # Force at least -0.3 if negative keyword found
    if pos_hits and not neg_hits:
        polarity = max(polarity, 0.3)   # Force at least +0.3 if positive keyword found

    if polarity > 0.1:
        return 'Positive', round(polarity, 4)
    elif polarity < -0.1:
        return 'Negative', round(polarity, 4)
    else:
        return 'Neutral', round(polarity, 4)


# --- Burnout Calculation ---

def encode_mood(mood):
    return {'Happy': 0, 'Okay': 1, 'Stressed': 2}.get(mood, 1)

def calculate_burnout(mood, work_hours, fatigue, experience, sentiment_score, model=None):
    mood_encoded = encode_mood(mood)
    
    # We deduce missing dataset fields from existing ones so UI doesn't need to change
    stressLevel = float(fatigue)
    sleepHours = max(4.0, 10.0 - float(fatigue) * 0.5)
    
    if model is not None:
        prediction = model.predict([[stressLevel, float(work_hours), float(fatigue), sleepHours, mood_encoded]])
        base_score = float(prediction[0])
    else:
        # Fallback if model not loaded
        base_score = mood_encoded * 2.0
        if work_hours > 8:
            base_score += min((work_hours - 8) * 0.5, 3.0)
        base_score += fatigue * 0.3
        if experience < 1:
            base_score += 1.0
        elif experience > 5:
            base_score -= 0.5
        if sentiment_score < -0.2:
            base_score += abs(sentiment_score) * 2.0
            
    return min(max(base_score, 0.0), 10.0)

def get_burnout_level(score):
    if score >= 7:
        return 'High'
    elif score >= 4:
        return 'Medium'
    else:
        return 'Low'


# --- Weekly Trend (legacy, for /predict) ---

def get_week_info():
    now = datetime.now()
    return now.isocalendar()[1], now.weekday()

def get_weekly_data(employee_id, week_number):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('''
        SELECT day_of_week, burnout_score, fatigue, work_hours, mood, sentiment
        FROM records 
        WHERE employee_id = ? AND week_number = ?
        ORDER BY day_of_week
    ''', (employee_id, week_number))
    rows = cursor.fetchall()
    conn.close()
    return rows

def analyze_weekly_trend(employee_id, current_day, current_score, week_number):
    past_data = get_weekly_data(employee_id, week_number)
    if not past_data:
        return {'trend': 'baseline', 'message': 'First assessment this week - establishing baseline.',
                'pattern': 'No pattern yet', 'avg_score': current_score, 'comparison': None}
    past_scores = [row[1] for row in past_data if row[0] < current_day]
    if not past_scores:
        return {'trend': 'baseline', 'message': 'First assessment this week - establishing baseline.',
                'pattern': 'No pattern yet', 'avg_score': current_score, 'comparison': None}
    avg_past = sum(past_scores) / len(past_scores)
    diff = current_score - avg_past
    if diff > 10:
        trend = 'worsening'
        message = f'Your burnout level is increasing. Score is {abs(diff):.1f} points higher than your weekly average.'
    elif diff < -10:
        trend = 'improving'
        message = f'Great progress! Your burnout level is {abs(diff):.1f} points lower than your weekly average.'
    else:
        trend = 'stable'
        message = 'Your burnout level is relatively stable this week.'
    patterns = []
    fatigue_values = [row[2] for row in past_data]
    work_hours_values = [row[3] for row in past_data]
    if len(fatigue_values) >= 2:
        if all(fatigue_values[i] <= fatigue_values[i+1] for i in range(len(fatigue_values)-1)):
            patterns.append('Fatigue increasing throughout the week')
        elif all(fatigue_values[i] >= fatigue_values[i+1] for i in range(len(fatigue_values)-1)):
            patterns.append('Fatigue decreasing - good recovery')
    if len(work_hours_values) >= 2:
        avg_hours = sum(work_hours_values) / len(work_hours_values)
        if avg_hours > 9:
            patterns.append('Consistently high work hours')
    pattern_str = '; '.join(patterns) if patterns else 'No significant patterns detected'
    return {
        'trend': trend, 'message': message, 'pattern': pattern_str,
        'avg_score': avg_past, 'days_analyzed': len(past_scores),
        'comparison': {'current': current_score, 'average': avg_past, 'difference': diff}
    }


# ============================================================
# CORE: Employee Trend Engine
# ============================================================

def compute_employee_trend(employee_id):
    """
    Compute an employee's burnout trend based on their last 5-7 records.
    Returns: trend label, percentage change, insight text, and raw scores.
    """
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('''
        SELECT burnout_score, submission_date, created_at
        FROM records
        WHERE employee_id = ?
        ORDER BY created_at DESC
        LIMIT 7
    ''', (employee_id,))
    rows = cursor.fetchall()
    conn.close()

    if len(rows) < 2:
        return {
            'trend': 'No Data',
            'trend_arrow': '→',
            'percentage_change': 0,
            'insight': 'Not enough data to determine trend.',
            'scores': [r[0] for r in rows],
            'record_count': len(rows)
        }

    scores = [r[0] for r in rows]
    # rows are DESC ordered; latest is index 0, oldest is last
    latest = scores[0]
    oldest = scores[-1]
    trend_score = latest - oldest

    if len(scores) > 1:
        pct_change = ((latest - oldest) / max(oldest, 0.1)) * 100
    else:
        pct_change = 0

    if trend_score > 0.5:
        trend = 'Increasing'
        arrow = '↑'
        insight = f'Burnout increased by {abs(pct_change):.1f}% over last {len(scores)} records.'
    elif trend_score < -0.5:
        trend = 'Decreasing'
        arrow = '↓'
        insight = f'Burnout decreased by {abs(pct_change):.1f}% over last {len(scores)} records. Good progress!'
    else:
        trend = 'Stable'
        arrow = '→'
        insight = f'Burnout level is stable over last {len(scores)} records.'

    return {
        'trend': trend,
        'trend_arrow': arrow,
        'percentage_change': round(pct_change, 1),
        'insight': insight,
        'scores': list(reversed(scores)),  # chronological order
        'record_count': len(scores)
    }


def get_employee_last_record(employee_id):
    """Get the most recent record for an employee."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('''
        SELECT burnout_score, burnout_level, fatigue, work_hours, mood, submission_date, created_at
        FROM records WHERE employee_id = ? ORDER BY created_at DESC LIMIT 1
    ''', (employee_id,))
    row = cursor.fetchone()
    conn.close()
    return row


# ============================================================
# CORE: Pattern Detection (Rule-Based)
# ============================================================

def detect_patterns(employee_id, current_fatigue, current_work_hours, current_burnout_score, current_sentiment_score):
    """
    Detect behavioral patterns using recent history + current values.
    Returns a pattern dictionary for AI context.
    """
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('''
        SELECT fatigue, work_hours, burnout_score
        FROM records WHERE employee_id = ?
        ORDER BY created_at DESC LIMIT 7
    ''', (employee_id,))
    recent = cursor.fetchall()
    conn.close()

    patterns = {
        'fatigue_increasing': False,
        'work_hours_high': False,
        'burnout_spike': False,
        'negative_sentiment': False
    }

    patterns['negative_sentiment'] = current_sentiment_score < -0.2
    patterns['work_hours_high'] = current_work_hours > 9

    if len(recent) >= 2:
        fatigue_values = [r[0] for r in recent]
        # Check if fatigue has been increasing (values in DESC order)
        patterns['fatigue_increasing'] = fatigue_values[0] >= fatigue_values[-1] and current_fatigue >= fatigue_values[0]

        if len(recent) >= 1:
            prev_score = recent[0][2]
            patterns['burnout_spike'] = (current_burnout_score - prev_score) >= 2

    return patterns


# ============================================================
# CORE: Hybrid AI Suggestions
# ============================================================

def generate_suggestions(burnout_level, sentiment, weekly_trend=None, user_feedback="",
                          work_hours=8, fatigue=5, patterns=None, history_summary=None,
                          employee_id=None, burnout_score=None):
    """
    Hybrid AI system: rule-based pattern detection + Gemini structured JSON output.
    """
    if patterns is None:
        patterns = {'fatigue_increasing': False, 'work_hours_high': False,
                    'burnout_spike': False, 'negative_sentiment': False}

    # Determine trend direction from history_summary
    trend_direction = 'stable'
    if history_summary:
        if 'decreased' in history_summary.lower() or 'improving' in history_summary.lower():
            trend_direction = 'improving'
        elif 'increased' in history_summary.lower() or 'worsening' in history_summary.lower():
            trend_direction = 'worsening'

    # Build history context
    history_text = ""
    if history_summary:
        history_text = f"\nLast 7 days trend: {history_summary}"

    score_text = f"\n- Burnout Score: {burnout_score:.1f}/10" if burnout_score is not None else ""
    active_patterns = [k for k, v in patterns.items() if v]
    patterns_text = ', '.join(active_patterns) if active_patterns else 'none detected'

    try:
        model = genai.GenerativeModel('gemini-2.0-flash')

        prompt = f"""
You are a mental health and workplace wellness AI assistant. Analyze an employee's burnout data and return a structured JSON response.

Employee Data:
- Burnout Level: {burnout_level} (Low/Medium/High){score_text}
- Overall Trend: {trend_direction} (improving = getting better, worsening = getting worse, stable = unchanged)
- Work Hours: {work_hours} hours/day
- Fatigue Level: {fatigue}/10
- Sentiment: {sentiment}
- Employee Feedback: "{user_feedback}"
- Detected Patterns: {patterns_text}
{history_text}

IMPORTANT RULES FOR SUGGESTIONS:
1. If burnout level is Low AND trend is improving: Give ENCOURAGING, maintenance-focused advice. Celebrate progress. Do NOT give crisis-level advice.
2. If burnout level is Low AND trend is worsening: Give gentle early-intervention advice.
3. If burnout level is Medium AND trend is improving: Acknowledge progress, give sustaining advice.
4. If burnout level is Medium AND trend is worsening: Give moderate intervention advice.
5. If burnout level is High: Give immediate action advice regardless of trend.
6. Negative sentiment alone should NOT override a good burnout score + improving trend.
7. Always give exactly 4 suggestions, tailored to the actual score and trend.

Respond ONLY with valid JSON in this exact format (no markdown, no extra text):
{{
  "pattern_detected": "brief description of the main pattern",
  "risk_reason": "one sentence explaining the current situation (positive or negative)",
  "suggestions": [
    "specific actionable suggestion 1",
    "specific actionable suggestion 2",
    "specific actionable suggestion 3",
    "specific actionable suggestion 4"
  ]
}}
"""

        response = model.generate_content(prompt)
        text = response.text.strip()

        # Strip markdown code fences if present
        text = re.sub(r'^```(?:json)?\s*', '', text)
        text = re.sub(r'\s*```$', '', text)

        parsed = json.loads(text)
        suggestions = parsed.get('suggestions', [])
        if suggestions and len(suggestions) >= 2:
            return suggestions[:4]
        raise ValueError("Insufficient suggestions from AI")

    except Exception as e:
        print(f"Gemini API error: {e}")
        return get_fallback_suggestions(burnout_level, sentiment, weekly_trend, user_feedback, work_hours, fatigue, patterns, burnout_score)


def get_fallback_suggestions(burnout_level, sentiment, weekly_trend=None, user_feedback="",
                              work_hours=8, fatigue=5, patterns=None, burnout_score=None):
    """Rule-based fallback suggestions — fully context-aware (score + trend + patterns)."""
    if patterns is None:
        patterns = {}

    feedback_lower = user_feedback.lower()
    score = burnout_score if burnout_score is not None else (3.0 if burnout_level == 'Low' else (5.5 if burnout_level == 'Medium' else 8.0))

    # Determine trend direction from weekly_trend dict or string
    trend_direction = 'stable'
    if isinstance(weekly_trend, dict):
        t = weekly_trend.get('trend', 'stable').lower()
    else:
        t = str(weekly_trend or '').lower()
    if t in ('improving', 'decreasing'):
        trend_direction = 'improving'
    elif t in ('worsening', 'increasing'):
        trend_direction = 'worsening'

    suggestions = []

    # ── HIGH burnout: always urgent action ───────────────────────────────────
    if burnout_level == 'High':
        if patterns.get('burnout_spike'):
            suggestions.append("Your burnout spiked sharply today — block tomorrow morning for low-intensity tasks or rest.")
        suggestions.append("Speak with your manager this week about immediate workload relief — your score needs urgent attention.")
        suggestions.append("Schedule 30+ minutes of physical activity today to release built-up stress hormones.")
        suggestions.append("Practice box breathing (4-4-4-4) when overwhelmed — it activates your body's calm response.")
        if fatigue >= 7:
            suggestions.append("Prioritize 8 hours of sleep tonight; avoid screens 1 hour before bed to improve sleep quality.")
        else:
            suggestions.append("Consider requesting a mental health day to fully recharge before burnout deepens.")
        return suggestions[:4]

    # ── LOW burnout + IMPROVING trend: celebrate & maintain ──────────────────
    if burnout_level == 'Low' and trend_direction == 'improving':
        suggestions.append("Great progress — your burnout is declining! Keep maintaining the habits that are working.")
        if sentiment == 'Negative' or any(w in feedback_lower for w in ['tired', 'exhausted', 'stressed']):
            suggestions.append("You mentioned feeling tired today — this is normal during recovery. Honour your rest time tonight.")
        else:
            suggestions.append("Your wellbeing is on a positive trajectory. Celebrate this win with a small reward today.")
        suggestions.append("Continue keeping work hours balanced and protect your evenings from work spillover.")
        suggestions.append("Regular check-ins like this help you catch early warning signs — keep the habit going!")
        if fatigue >= 6:
            suggestions.append("Even with a low burnout score, moderate fatigue is worth watching — aim for 7-8 hours of sleep.")
        else:
            suggestions.append("Stay connected with teammates — social support is key to sustaining your recovery.")
        return suggestions[:4]

    # ── LOW burnout + WORSENING trend: gentle early intervention ─────────────
    if burnout_level == 'Low' and trend_direction == 'worsening':
        suggestions.append("Your score is still low, but notice the upward trend early — small adjustments now prevent bigger issues later.")
        if patterns.get('work_hours_high') or work_hours > 9:
            suggestions.append("Try capping your work day at 8-9 hours this week to prevent the creeping fatigue from building further.")
        if sentiment == 'Negative' or any(w in feedback_lower for w in ['tired', 'exhausted', 'stressed']):
            suggestions.append("Your feedback signals stress is building — talk to a colleague or journal your concerns to reduce mental load.")
        suggestions.append("Schedule a 10-minute mid-day walk to break the mental tension before it accumulates.")
        suggestions.append("Identify your top 3 priorities for tomorrow to reduce decision fatigue and reclaim a sense of control.")
        return suggestions[:4]

    # ── MEDIUM burnout + IMPROVING trend: sustain momentum ──────────────────
    if burnout_level == 'Medium' and trend_direction == 'improving':
        suggestions.append("You're making real progress — your burnout trend is heading in the right direction. Stay consistent.")
        suggestions.append("Set a firm end-of-work time today and protect it — consistent boundaries are what's driving your recovery.")
        if fatigue >= 6:
            suggestions.append("Fatigue is still moderate — prioritise 7-8 hours of sleep and limit caffeine after 2pm.")
        else:
            suggestions.append("Take a 10-minute walk mid-afternoon to sustain your energy through the rest of the day.")
        suggestions.append("Connect with a supportive colleague this week — social engagement reinforces emotional resilience.")
        return suggestions[:4]

    # ── MEDIUM burnout + WORSENING or STABLE trend: moderate intervention ───
    if burnout_level == 'Medium':
        if patterns.get('work_hours_high') and patterns.get('fatigue_increasing'):
            suggestions.append("Fatigue rising with long hours — cap work at 9 hours today and schedule mandatory breaks every 90 minutes.")
        else:
            suggestions.append("Set a hard stop time for work today and honour it — your medium burnout needs active boundary-setting.")
        if patterns.get('burnout_spike'):
            suggestions.append("Today's score jumped — block tomorrow's morning for lighter tasks to avoid a full burnout spiral.")
        if sentiment == 'Negative' or any(w in feedback_lower for w in ['overwhelmed', 'stressed', 'anxious', 'tired', 'exhausted']):
            suggestions.append("Talk with your manager about redistributing 1-2 urgent tasks this week to ease your current load.")
        suggestions.append("Take a 10-minute walk mid-afternoon to reset your nervous system and regain focus.")
        suggestions.append("Identify your top 3 priorities for tomorrow — reducing decision fatigue helps lower medium-level burnout.")
        return suggestions[:4]

    # ── DEFAULT FALLBACK (Low + stable with no specific patterns) ───────────
    suggestions = [
        "Keep up your current habits — consistency is the foundation of long-term wellness.",
        "Stay connected with your team; social support is a proven buffer against burnout.",
        "Celebrate small wins today — positive reinforcement builds long-term resilience.",
        "Continue regular check-ins to catch any early signs of burnout before they escalate."
    ]
    return suggestions[:4]


# ============================================================
# CORE: Weekly Analytics Engine
# ============================================================

def compute_weekly_analytics():
    """
    Compute team-level weekly analytics:
    - avg burnout per day (last 7 days)
    - avg fatigue per day
    - high-risk count per day
    - week-over-week burnout change %
    """
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    today = datetime.now().date()
    seven_days_ago = today - timedelta(days=7)
    fourteen_days_ago = today - timedelta(days=14)

    # Last 7 days
    cursor.execute('''
        SELECT submission_date, AVG(burnout_score), AVG(fatigue), 
               SUM(CASE WHEN burnout_level = 'High' THEN 1 ELSE 0 END),
               COUNT(*)
        FROM records
        WHERE submission_date >= ?
        GROUP BY submission_date
        ORDER BY submission_date ASC
    ''', (str(seven_days_ago),))
    this_week_rows = cursor.fetchall()

    # Previous 7 days (for comparison)
    cursor.execute('''
        SELECT AVG(burnout_score) FROM records
        WHERE submission_date >= ? AND submission_date < ?
    ''', (str(fourteen_days_ago), str(seven_days_ago)))
    prev_week_avg_row = cursor.fetchone()
    prev_week_avg = prev_week_avg_row[0] if prev_week_avg_row and prev_week_avg_row[0] else None

    # This week's overall avg
    cursor.execute('''
        SELECT AVG(burnout_score) FROM records WHERE submission_date >= ?
    ''', (str(seven_days_ago),))
    this_week_avg_row = cursor.fetchone()
    this_week_avg = this_week_avg_row[0] if this_week_avg_row and this_week_avg_row[0] else 0

    conn.close()

    daily_data = []
    for row in this_week_rows:
        date_str = row[0] or ''
        daily_data.append({
            'date': date_str,
            'avg_burnout': round(row[1], 2) if row[1] else 0,
            'avg_fatigue': round(row[2], 2) if row[2] else 0,
            'high_risk_count': int(row[3]) if row[3] else 0,
            'submission_count': int(row[4]) if row[4] else 0
        })

    # Compute week-over-week change
    burnout_change_pct = 0
    if prev_week_avg and prev_week_avg > 0 and this_week_avg:
        burnout_change_pct = round(((this_week_avg - prev_week_avg) / prev_week_avg) * 100, 1)

    # Generate insights
    insights = []
    if burnout_change_pct > 10:
        insights.append(f"Team burnout increased by {burnout_change_pct}% compared to last week.")
    elif burnout_change_pct < -10:
        insights.append(f"Team burnout improved by {abs(burnout_change_pct)}% compared to last week.")
    else:
        insights.append("Team burnout levels are stable week-over-week.")

    # Check fatigue trend
    if len(daily_data) >= 3:
        recent_fatigue = [d['avg_fatigue'] for d in daily_data[-3:]]
        if all(recent_fatigue[i] <= recent_fatigue[i+1] for i in range(len(recent_fatigue)-1)):
            insights.append("Team fatigue has been rising over the last 3 days.")

    high_risk_total = sum(d['high_risk_count'] for d in daily_data)
    if high_risk_total == 0:
        insights.append("No high-risk burnout submissions this week.")

    return {
        'daily_data': daily_data,
        'this_week_avg': round(this_week_avg, 2),
        'prev_week_avg': round(prev_week_avg, 2) if prev_week_avg else None,
        'burnout_change_pct': burnout_change_pct,
        'insights': insights
    }


# ============================================================
# CORE: Extended Analytics Engines
# ============================================================

def compute_custom_analytics(start_date_str, end_date_str, analysis_type='daily'):
    """
    Returns custom team avg burnout + fatigue for a specific date range.
    analysis_type:
      'daily' -> timeline of exact days
      'weekly_grouped' -> groups by Day of Week (Mon-Sun)
      'monthly_grouped' -> groups by Week of Month (Week 1-4)
    """
    try:
        start_date = datetime.strptime(start_date_str, '%Y-%m-%d').date()
        end_date = datetime.strptime(end_date_str, '%Y-%m-%d').date()
    except ValueError:
        today = datetime.now().date()
        start_date = today - timedelta(days=7)
        end_date = today

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    cursor.execute('''
        SELECT submission_date, AVG(burnout_score), AVG(fatigue),
               SUM(CASE WHEN burnout_level = 'High' THEN 1 ELSE 0 END),
               COUNT(*)
        FROM records
        WHERE submission_date >= ? AND submission_date <= ?
        GROUP BY submission_date
        ORDER BY submission_date ASC
    ''', (str(start_date), str(end_date)))
    rows = cursor.fetchall()
    conn.close()

    if analysis_type == 'weekly_grouped':
        # Group by ISO week and display explicit date ranges.
        from collections import defaultdict
        week_data = defaultdict(lambda: {'burnouts': [], 'fatigues': [], 'dates': []})

        for row in rows:
            date_str = row[0]
            if not date_str:
                continue
            try:
                d = datetime.strptime(date_str, '%Y-%m-%d').date()
                iso = d.isocalendar()
                week_key = f"{iso[0]}-W{iso[1]:02d}"
                week_data[week_key]['burnouts'].append(row[1] if row[1] else 0)
                week_data[week_key]['fatigues'].append(row[2] if row[2] else 0)
                week_data[week_key]['dates'].append(d)
            except Exception:
                pass

        result_data = []
        for week_key in sorted(week_data.keys()):
            b_list = week_data[week_key]['burnouts']
            f_list = week_data[week_key]['fatigues']
            dates = week_data[week_key]['dates']
            if dates:
                min_date = min(dates).strftime('%Y-%m-%d')
                max_date = max(dates).strftime('%Y-%m-%d')
                label = min_date if min_date == max_date else f"{min_date} to {max_date}"
            else:
                label = week_key

            result_data.append({
                'label': label,
                'avg_burnout': round(sum(b_list) / len(b_list), 2) if b_list else 0,
                'avg_fatigue': round(sum(f_list) / len(f_list), 2) if f_list else 0
            })

        return {'grouped_data': result_data, 'view_type': 'weekly_grouped'}

    elif analysis_type == 'monthly_grouped':
        # Group by Week of Month (Week 1, Week 2, Week 3, Week 4)
        from collections import defaultdict
        wom_data = defaultdict(lambda: {'burnouts': [], 'fatigues': []})

        for row in rows:
            date_str = row[0]
            if not date_str: continue
            try:
                d = datetime.strptime(date_str, '%Y-%m-%d').date()
                week_of_month = (d.day - 1) // 7 + 1
                if week_of_month > 4: week_of_month = 4 # cap at 4 
                wom_data[week_of_month]['burnouts'].append(row[1] if row[1] else 0)
                wom_data[week_of_month]['fatigues'].append(row[2] if row[2] else 0)
            except Exception:
                pass
        
        result_data = []
        for i in range(1, 5):
            b_list = wom_data[i]['burnouts']
            f_list = wom_data[i]['fatigues']
            result_data.append({
                'label': f"Week {i}",
                'avg_burnout': round(sum(b_list) / len(b_list), 2) if b_list else 0,
                'avg_fatigue': round(sum(f_list) / len(f_list), 2) if f_list else 0
            })
        return {'grouped_data': result_data, 'view_type': 'monthly_grouped'}

    else:
        # Normal Daily Timeline
        daily_data = []
        for row in rows:
            daily_data.append({
                'date': row[0] or '',
                'avg_burnout': round(row[1], 2) if row[1] else 0,
                'avg_fatigue': round(row[2], 2) if row[2] else 0,
                'high_risk_count': int(row[3]) if row[3] else 0,
                'submission_count': int(row[4]) if row[4] else 0
            })
        return {'daily_data': daily_data, 'view_type': 'daily'}
# ============================================================
# CORE: Alerts Engine
# ============================================================

def compute_alerts():
    """
    Generate alerts for employees with:
    - burnout_score >= 7
    - fatigue >= 8
    - increasing trend (3+ records)
    - sudden spike (score increase >= 2 in consecutive records)
    """
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    # Get all employees with records
    cursor.execute('''
        SELECT DISTINCT employee_id, employee_name FROM records WHERE employee_id IS NOT NULL
    ''')
    employees = cursor.fetchall()

    alerts = []
    for emp_id, emp_name in employees:
        cursor.execute('''
            SELECT burnout_score, fatigue, submission_date, created_at
            FROM records WHERE employee_id = ?
            ORDER BY created_at DESC LIMIT 5
        ''', (emp_id,))
        recs = cursor.fetchall()

        if not recs:
            continue

        latest = recs[0]
        latest_score = latest[0]
        latest_fatigue = latest[1]

        if latest_score >= 7:
            alerts.append({
                'employee_id': emp_id,
                'employee_name': emp_name,
                'type': 'high_burnout',
                'severity': 'critical',
                'message': f'{emp_name} has a high burnout score of {latest_score:.1f}.',
                'value': latest_score
            })

        if latest_fatigue >= 8:
            alerts.append({
                'employee_id': emp_id,
                'employee_name': emp_name,
                'type': 'high_fatigue',
                'severity': 'warning',
                'message': f'{emp_name} reported extreme fatigue ({latest_fatigue}/10).',
                'value': latest_fatigue
            })

        # Sudden spike detection
        if len(recs) >= 2:
            prev_score = recs[1][0]
            if latest_score - prev_score >= 2:
                alerts.append({
                    'employee_id': emp_id,
                    'employee_name': emp_name,
                    'type': 'sudden_spike',
                    'severity': 'critical',
                    'message': f'{emp_name} had a sudden burnout spike (+{latest_score - prev_score:.1f} points).',
                    'value': latest_score - prev_score
                })

        # Increasing trend (3+ records)
        if len(recs) >= 3:
            scores = [r[0] for r in recs[:3]]  # last 3, DESC order → [latest, prev, older]
            if scores[0] > scores[1] > scores[2]:
                alerts.append({
                    'employee_id': emp_id,
                    'employee_name': emp_name,
                    'type': 'increasing_trend',
                    'severity': 'warning',
                    'message': f'{emp_name} has shown consistently increasing burnout over 3 consecutive records.',
                    'value': scores[0]
                })

    conn.close()
    # Deduplicate: one alert per employee per type
    seen = set()
    unique_alerts = []
    for a in alerts:
        key = (a['employee_id'], a['type'])
        if key not in seen:
            seen.add(key)
            unique_alerts.append(a)

    return sorted(unique_alerts, key=lambda x: {'critical': 0, 'warning': 1}.get(x['severity'], 2))


# ============================================================
# CORE: Correlation Insight
# ============================================================

def compute_correlation_insight():
    """Compute Pearson correlation between work_hours and fatigue."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('SELECT work_hours, fatigue FROM records WHERE employee_id IS NOT NULL')
    rows = cursor.fetchall()
    conn.close()

    if len(rows) < 5:
        return {'correlation': None, 'insight': 'Not enough data for correlation analysis.'}

    work_hours = np.array([r[0] for r in rows])
    fatigue = np.array([r[1] for r in rows])

    if np.std(work_hours) == 0 or np.std(fatigue) == 0:
        return {'correlation': 0, 'insight': 'Insufficient variance for correlation analysis.'}

    corr = np.corrcoef(work_hours, fatigue)[0, 1]
    corr_rounded = round(float(corr), 2)

    if corr > 0.6:
        insight = f"Strong correlation ({corr_rounded}) between long work hours and high fatigue. Reducing hours may directly lower fatigue."
    elif corr > 0.3:
        insight = f"Moderate correlation ({corr_rounded}) between work hours and fatigue. Monitor employees working over 9 hours daily."
    elif corr > 0:
        insight = f"Weak positive correlation ({corr_rounded}) between work hours and fatigue."
    else:
        insight = f"No significant correlation ({corr_rounded}) between work hours and fatigue in current data."

    return {'correlation': corr_rounded, 'insight': insight}


# ============================================================
# CORE: Manager Recommendations
# ============================================================

def compute_manager_recommendations(alerts, top_risk, weekly_analytics):
    """Generate natural-language manager recommendations."""
    recs = []

    critical_count = sum(1 for a in alerts if a['severity'] == 'critical')
    warning_count = sum(1 for a in alerts if a['severity'] == 'warning')

    if critical_count > 0:
        recs.append(f"Urgently schedule 1:1 meetings with {critical_count} employee(s) flagged as critical burnout risk.")
    if warning_count > 0:
        recs.append(f"Monitor {warning_count} employee(s) showing warning signs — consider redistributing their workload.")
    if weekly_analytics.get('burnout_change_pct', 0) > 10:
        recs.append("Team burnout spiked this week — consider a team-wide check-in or reducing sprint scope.")
    if top_risk and len(top_risk) > 0 and top_risk[0].get('burnout_score', 0) >= 8:
        recs.append(f"Consider granting a mental health day to {top_risk[0]['name']} (score: {top_risk[0]['burnout_score']:.1f}).")
    if not recs:
        recs.append("Team appears stable. Continue regular check-ins to maintain wellness.")

    return recs


# ============================================================
# Initialize DB and Model
# ============================================================

init_db()
model = load_model()


# ============================================================
# Routes - Auth
# ============================================================

@app.route('/health')
def health():
    return jsonify({'status': 'healthy', 'model_loaded': model is not None})


@app.route('/auth/employee/register', methods=['POST'])
def employee_register():
    try:
        data = request.json
        email = data.get('email', '').strip().lower()
        password = data.get('password', '')
        name = data.get('name', '').strip()
        department = data.get('department', '').strip()

        if not email or not password or not name:
            return jsonify({'error': 'Email, password, and name are required'}), 400
        if len(password) < 6:
            return jsonify({'error': 'Password must be at least 6 characters'}), 400

        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute('SELECT id FROM employees WHERE email = ?', (email,))
        if cursor.fetchone():
            conn.close()
            return jsonify({'error': 'Email already registered'}), 400

        cursor.execute('''
            INSERT INTO employees (email, password, name, department) VALUES (?, ?, ?, ?)
        ''', (email, hash_password(password), name, department))
        conn.commit()
        employee_id = cursor.lastrowid
        conn.close()

        token = generate_token()
        sessions[token] = {'id': employee_id, 'email': email, 'name': name, 'department': department, 'role': 'employee'}
        return jsonify({'message': 'Registration successful', 'token': token,
                        'user': {'id': employee_id, 'email': email, 'name': name, 'department': department, 'role': 'employee'}})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/auth/employee/login', methods=['POST'])
def employee_login():
    try:
        data = request.json
        email = data.get('email', '').strip().lower()
        password = data.get('password', '').strip()

        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute('SELECT id, email, name, department FROM employees WHERE email = ? AND password = ?',
                       (email, hash_password(password)))
        user = cursor.fetchone()
        conn.close()

        if not user:
            return jsonify({'error': 'Invalid email or password'}), 401

        token = generate_token()
        sessions[token] = {'id': user[0], 'email': user[1], 'name': user[2], 'department': user[3], 'role': 'employee'}
        return jsonify({'message': 'Login successful', 'token': token,
                        'user': {'id': user[0], 'email': user[1], 'name': user[2], 'department': user[3], 'role': 'employee'}})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/auth/manager/login', methods=['POST'])
def manager_login():
    try:
        data = request.json
        email = data.get('email', '').strip().lower()
        password = data.get('password', '').strip()

        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute('SELECT id, email, name, department FROM managers WHERE email = ? AND password = ?',
                       (email, hash_password(password)))
        user = cursor.fetchone()
        conn.close()

        if not user:
            return jsonify({'error': 'Invalid email or password'}), 401

        token = generate_token()
        sessions[token] = {'id': user[0], 'email': user[1], 'name': user[2], 'department': user[3], 'role': 'manager'}
        return jsonify({'message': 'Login successful', 'token': token,
                        'user': {'id': user[0], 'email': user[1], 'name': user[2], 'department': user[3], 'role': 'manager'}})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/auth/logout', methods=['POST'])
def logout():
    token = request.headers.get('Authorization', '').replace('Bearer ', '')
    if token in sessions:
        del sessions[token]
    return jsonify({'message': 'Logged out successfully'})


@app.route('/auth/verify', methods=['GET'])
def verify_token():
    token = request.headers.get('Authorization', '').replace('Bearer ', '')
    session = verify_session(token, 'any')
    if session:
        return jsonify({'valid': True, 'user': session})
    return jsonify({'valid': False}), 401


# ============================================================
# Routes - Employee
# ============================================================

@app.route('/predict', methods=['POST'])
def predict():
    try:
        token = request.headers.get('Authorization', '').replace('Bearer ', '')
        session = verify_session(token, 'employee')
        if not session:
            return jsonify({'error': 'Unauthorized. Please login.'}), 401

        data = request.json
        employee_id = session['id']
        employee_name = session['name']
        mood = data.get('mood', 'Okay')
        work_hours = float(data.get('work_hours', 8))
        fatigue = int(data.get('fatigue', 5))
        experience = float(data.get('experience', 1))
        feedback = data.get('feedback', '')

        if not (0 <= fatigue <= 10):
            return jsonify({'error': 'Fatigue must be between 0 and 10'}), 400
        if not (0 <= work_hours <= 24):
            return jsonify({'error': 'Work hours must be between 0 and 24'}), 400
        if experience < 0:
            return jsonify({'error': 'Experience cannot be negative'}), 400

        week_number, day_of_week = get_week_info()
        today_str = datetime.now().strftime('%Y-%m-%d')

        sentiment, sentiment_score = analyze_sentiment(feedback)
        burnout_score = calculate_burnout(mood, work_hours, fatigue, experience, sentiment_score, model)
        burnout_level = get_burnout_level(burnout_score)

        # Weekly trend (legacy)
        weekly_trend_data = analyze_weekly_trend(employee_id, day_of_week, burnout_score, week_number)
        weekly_trend_label = weekly_trend_data.get('trend', 'baseline')

        # Pattern detection
        patterns = detect_patterns(employee_id, fatigue, work_hours, burnout_score, sentiment_score)

        # Employee trend for insight
        emp_trend = compute_employee_trend(employee_id)
        history_summary = emp_trend.get('insight', '')

        # Suggestions
        suggestions = generate_suggestions(
            burnout_level, sentiment, weekly_trend_data, feedback,
            work_hours, fatigue, patterns, history_summary, employee_id,
            burnout_score=burnout_score
        )

        # Store record
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO records (employee_id, employee_name, mood, work_hours, fatigue, experience,
                               feedback, sentiment, sentiment_score, burnout_score, burnout_level,
                               suggestions, week_number, day_of_week, weekly_trend, submission_date)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (employee_id, employee_name, mood, work_hours, fatigue, experience,
              feedback, sentiment, sentiment_score, burnout_score, burnout_level,
              '|'.join(suggestions), week_number, day_of_week, weekly_trend_label, today_str))
        conn.commit()
        record_id = cursor.lastrowid
        conn.close()

        day_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']

        return jsonify({
            'id': record_id,
            'employee_name': employee_name,
            'mood': mood,
            'work_hours': work_hours,
            'fatigue': fatigue,
            'experience': experience,
            'feedback': feedback,
            'sentiment': sentiment,
            'sentiment_score': sentiment_score,
            'burnout_score': round(burnout_score, 1),
            'burnout_level': burnout_level,
            'suggestions': suggestions,
            'patterns': patterns,
            'trend_insight': emp_trend.get('insight', ''),
            'trend_arrow': emp_trend.get('trend_arrow', '→'),
            'weekly_analysis': {
                'current_day': day_names[day_of_week],
                'week_number': week_number,
                'trend': weekly_trend_data.get('trend'),
                'trend_message': weekly_trend_data.get('message'),
                'pattern': weekly_trend_data.get('pattern'),
                'days_analyzed': weekly_trend_data.get('days_analyzed', 0),
                'comparison': weekly_trend_data.get('comparison')
            }
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/my-records')
def get_my_records():
    try:
        token = request.headers.get('Authorization', '').replace('Bearer ', '')
        session = verify_session(token, 'employee')
        if not session:
            return jsonify({'error': 'Unauthorized'}), 401

        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute('''
            SELECT id, mood, work_hours, fatigue, experience, feedback, sentiment, sentiment_score,
                   burnout_score, burnout_level, suggestions, week_number, day_of_week, weekly_trend,
                   submission_date, created_at
            FROM records WHERE employee_id = ? ORDER BY created_at DESC
        ''', (session['id'],))
        rows = cursor.fetchall()
        conn.close()

        records = []
        for row in rows:
            raw_date = row[14] or row[15] or ''
            try:
                if raw_date:
                    parsed_date = datetime.fromisoformat(raw_date.split('T')[0]).strftime('%Y-%m-%d')
                else:
                    parsed_date = ''
            except Exception:
                parsed_date = raw_date[:10] if raw_date else ''

            records.append({
                'id': row[0], 'mood': row[1], 'work_hours': row[2], 'fatigue': row[3],
                'experience': row[4], 'feedback': row[5], 'sentiment': row[6],
                'sentiment_score': row[7], 'burnout_score': row[8], 'burnout_level': row[9],
                'suggestions': row[10].split('|') if row[10] else [],
                'week_number': row[11], 'day_of_week': row[12], 'weekly_trend': row[13],
                'submission_date': parsed_date,
                'created_at': row[15]
            })

        # Compute trend data to attach to the response
        trend_data = compute_employee_trend(session['id'])

        return jsonify({'records': records, 'trend': trend_data})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


# ============================================================
# Routes - Manager
# ============================================================

@app.route('/records')
def get_records():
    try:
        token = request.headers.get('Authorization', '').replace('Bearer ', '')
        session = verify_session(token, 'manager')
        if not session:
            return jsonify({'error': 'Unauthorized. Manager access required.'}), 401

        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute('SELECT * FROM records ORDER BY created_at DESC')
        rows = cursor.fetchall()
        conn.close()

        records = []
        for row in rows:
            records.append({
                'id': row[0], 'employee_id': row[1], 'employee_name': row[2],
                'mood': row[3], 'work_hours': row[4], 'fatigue': row[5], 'experience': row[6],
                'feedback': row[7], 'sentiment': row[8], 'sentiment_score': row[9],
                'burnout_score': row[10], 'burnout_level': row[11],
                'suggestions': row[12].split('|') if row[12] else [],
                'week_number': row[13], 'day_of_week': row[14], 'weekly_trend': row[15],
                'created_at': row[17]
            })
        return jsonify(records)
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/stats')
def get_stats():
    try:
        token = request.headers.get('Authorization', '').replace('Bearer ', '')
        session = verify_session(token, 'manager')
        if not session:
            return jsonify({'error': 'Unauthorized. Manager access required.'}), 401

        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()

        cursor.execute('SELECT COUNT(DISTINCT employee_id) FROM records')
        total_employees = cursor.fetchone()[0]
        cursor.execute('SELECT COUNT(*) FROM records')
        total_assessments = cursor.fetchone()[0]
        cursor.execute("SELECT COUNT(*) FROM records WHERE burnout_level = 'High'")
        high_burnout = cursor.fetchone()[0]
        cursor.execute("SELECT COUNT(*) FROM records WHERE burnout_level = 'Medium'")
        medium_burnout = cursor.fetchone()[0]
        cursor.execute("SELECT COUNT(*) FROM records WHERE burnout_level = 'Low'")
        low_burnout = cursor.fetchone()[0]
        cursor.execute('SELECT AVG(burnout_score) FROM records')
        avg_burnout = cursor.fetchone()[0] or 0
        cursor.execute('SELECT work_hours, fatigue, burnout_level FROM records')
        scatter_data = cursor.fetchall()
        cursor.execute("SELECT COUNT(*) FROM records WHERE weekly_trend = 'worsening'")
        worsening_trends = cursor.fetchone()[0]
        cursor.execute("SELECT COUNT(*) FROM records WHERE weekly_trend = 'improving'")
        improving_trends = cursor.fetchone()[0]
        conn.close()

        return jsonify({
            'total_employees': total_employees, 'total_assessments': total_assessments,
            'high_burnout': high_burnout, 'medium_burnout': medium_burnout, 'low_burnout': low_burnout,
            'avg_burnout': round(avg_burnout, 1), 'worsening_trends': worsening_trends,
            'improving_trends': improving_trends,
            'scatter_data': [{'work_hours': r[0], 'fatigue': r[1], 'burnout_level': r[2]} for r in scatter_data]
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/manager-insights')
def manager_insights():
    """
    Comprehensive manager intelligence endpoint.
    Returns: summary, alerts, top_risk, employees, weekly_analytics, correlation_insight, recommendations
    """
    try:
        token = request.headers.get('Authorization', '').replace('Bearer ', '')
        session = verify_session(token, 'manager')
        if not session:
            return jsonify({'error': 'Unauthorized. Manager access required.'}), 401

        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()

        # All unique employees with records
        cursor.execute('''
            SELECT DISTINCT employee_id, employee_name
            FROM records WHERE employee_id IS NOT NULL
        ''')
        employee_rows = cursor.fetchall()

        # Build employee summary list
        employees = []
        total_burnout_sum = 0
        high_risk_count = 0

        for emp_id, emp_name in employee_rows:
            last_record = get_employee_last_record(emp_id)
            trend_data = compute_employee_trend(emp_id)

            if last_record:
                burnout_score = last_record[0]
                burnout_level = last_record[1]
                fatigue = last_record[2]
                work_hours = last_record[3]
                mood = last_record[4]
                last_updated = last_record[5] or (last_record[6][:10] if last_record[6] else '')

                total_burnout_sum += burnout_score
                if burnout_level == 'High':
                    high_risk_count += 1

                employees.append({
                    'employee_id': emp_id,
                    'name': emp_name,
                    'burnout_score': round(burnout_score, 1),
                    'burnout_level': burnout_level,
                    'fatigue': fatigue,
                    'work_hours': work_hours,
                    'mood': mood,
                    'last_updated': last_updated,
                    'trend': trend_data['trend'],
                    'trend_arrow': trend_data['trend_arrow'],
                    'trend_insight': trend_data['insight'],
                    'percentage_change': trend_data['percentage_change']
                })

        # Sort by burnout score desc
        employees.sort(key=lambda x: x['burnout_score'], reverse=True)

        total_employees = len(employees)
        avg_burnout = round(total_burnout_sum / total_employees, 2) if total_employees > 0 else 0

        conn.close()

        alerts = compute_alerts()
        weekly_analytics = compute_weekly_analytics()
        correlation = compute_correlation_insight()
        top_risk = employees[:5]
        recommendations = compute_manager_recommendations(alerts, top_risk, weekly_analytics)

        # Team-level trend
        team_trend = 'Stable'
        if weekly_analytics.get('burnout_change_pct', 0) > 5:
            team_trend = 'Increasing'
        elif weekly_analytics.get('burnout_change_pct', 0) < -5:
            team_trend = 'Decreasing'

        return jsonify({
            'summary': {
                'total_employees': total_employees,
                'high_risk_employees': high_risk_count,
                'avg_burnout': avg_burnout,
                'team_trend': team_trend,
                'team_trend_arrow': '↑' if team_trend == 'Increasing' else ('↓' if team_trend == 'Decreasing' else '→')
            },
            'alerts': alerts,
            'top_risk': top_risk,
            'employees': employees,
            'weekly_analytics': weekly_analytics,
            'correlation_insight': correlation,
            'team_insights': weekly_analytics.get('insights', []),
            'recommendations': recommendations
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/analytics/team-range')
def team_analytics_range():
    """
    Returns extended analytics data for the manager dashboard.
    Query param: startDate, endDate (YYYY-MM-DD)
    """
    try:
        token = request.headers.get('Authorization', '').replace('Bearer ', '')
        session = verify_session(token, 'manager')
        if not session:
            return jsonify({'error': 'Unauthorized'}), 401

        start_date = request.args.get('startDate', '')
        end_date = request.args.get('endDate', '')
        analysis_type = request.args.get('analysisType', 'daily')
        
        data = compute_custom_analytics(start_date, end_date, analysis_type)
        return jsonify(data)
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/employee/<int:employee_id>/history')
def employee_history(employee_id):
    """
    Full employee analysis endpoint.
    Supports query params: startDate, endDate for date-range filtering.
    Returns daily records, weekly aggregation, monthly aggregation, and insights.
    """
    try:
        token = request.headers.get('Authorization', '').replace('Bearer ', '')
        session = verify_session(token, 'any')
        if not session:
            return jsonify({'error': 'Unauthorized'}), 401

        if session['role'] == 'employee' and session['id'] != employee_id:
            return jsonify({'error': 'Forbidden'}), 403

        start_date = request.args.get('startDate', '')
        end_date = request.args.get('endDate', '')

        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()

        # Build query with optional date range
        query = '''
            SELECT id, mood, work_hours, fatigue, burnout_score, burnout_level,
                   weekly_trend, submission_date, created_at, sentiment, feedback,
                   sleep_hours, stress_level
            FROM records
            WHERE employee_id = ?
        '''
        params = [employee_id]

        if start_date:
            query += ' AND submission_date >= ?'
            params.append(start_date)
        if end_date:
            query += ' AND submission_date <= ?'
            params.append(end_date)

        query += ' ORDER BY submission_date ASC, created_at ASC'
        cursor.execute(query, params)
        rows = cursor.fetchall()

        cursor.execute('SELECT name, department, email FROM employees WHERE id = ?', (employee_id,))
        emp_row = cursor.fetchone()
        conn.close()

        records = []
        for row in rows:
            raw_date = row[7] or (row[8][:10] if row[8] else '')
            parsed_date = raw_date[:10] if raw_date else ''
            records.append({
                'id': row[0], 'mood': row[1], 'work_hours': row[2], 'fatigue': row[3],
                'burnout_score': row[4], 'burnout_level': row[5], 'weekly_trend': row[6],
                'submission_date': parsed_date, 'created_at': row[8],
                'sentiment': row[9], 'feedback': row[10],
                'sleep_hours': row[11] if row[11] is not None else 7.0,
                'stress_level': row[12] if row[12] is not None else int(row[4]) if row[4] else 5
            })

        # WEEKLY AGGREGATION: group by ISO week
        weekly_agg = {}
        for r in records:
            try:
                d = datetime.strptime(r['submission_date'], '%Y-%m-%d')
                iso = d.isocalendar()
                week_key = f"{iso[0]}-W{iso[1]:02d}"
                week_label = f"Week {iso[1]}"
            except Exception:
                week_key = 'unknown'
                week_label = 'Unknown'
            if week_key not in weekly_agg:
                weekly_agg[week_key] = {'label': week_label, 'records': []}
            weekly_agg[week_key]['records'].append(r)

        weekly_data = []
        for wk_key in sorted(weekly_agg.keys()):
            wk = weekly_agg[wk_key]
            recs = wk['records']
            weekly_data.append({
                'week': wk['label'],
                'week_key': wk_key,
                'avg_burnout': round(sum(r['burnout_score'] for r in recs) / len(recs), 2),
                'avg_fatigue': round(sum(r['fatigue'] for r in recs) / len(recs), 2),
                'avg_stress': round(sum(r['stress_level'] for r in recs) / len(recs), 2),
                'avg_sleep': round(sum(r['sleep_hours'] for r in recs) / len(recs), 2),
                'avg_work_hours': round(sum(r['work_hours'] for r in recs) / len(recs), 2),
                'count': len(recs)
            })

        # OVERALL ANALYSIS
        if records:
            all_burnout = [r['burnout_score'] for r in records]
            all_fatigue = [r['fatigue'] for r in records]
            all_sleep = [r['sleep_hours'] for r in records]
            all_stress = [r['stress_level'] for r in records]
            avg_burnout = round(sum(all_burnout) / len(all_burnout), 2)
            max_burnout = max(all_burnout)
            min_mood_idx = min(range(len(records)), key=lambda i: {'Happy': 2, 'Okay': 1, 'Stressed': 0}.get(records[i]['mood'], 1))
            highest_burnout_record = max(records, key=lambda r: r['burnout_score'])
            lowest_mood_record = records[min_mood_idx]

            # Risk level (0-10 scale)
            if avg_burnout >= 7:
                risk_level = 'High'
                risk_color = 'red'
            elif avg_burnout >= 4:
                risk_level = 'Medium'
                risk_color = 'yellow'
            else:
                risk_level = 'Low'
                risk_color = 'green'

            # INSIGHT GENERATION
            insights = []
            # Burnout-sleep correlation
            high_burnout_days = [r for r in records if r['burnout_score'] >= 7]
            if high_burnout_days:
                avg_sleep_high_burnout = sum(r['sleep_hours'] for r in high_burnout_days) / len(high_burnout_days)
                if avg_sleep_high_burnout < 6.5:
                    insights.append('High burnout days correlate with low sleep hours (avg {:.1f}h sleep on high-burnout days).'.format(avg_sleep_high_burnout))

            # Fatigue-work hours
            long_hour_days = [r for r in records if r['work_hours'] > 9]
            if long_hour_days:
                avg_fatigue_long = sum(r['fatigue'] for r in long_hour_days) / len(long_hour_days)
                avg_fatigue_normal = sum(r['fatigue'] for r in records) / len(records)
                if avg_fatigue_long > avg_fatigue_normal * 1.2:
                    insights.append('Increased fatigue correlates with long working hours (>{:.0f}h days show {:.0f}% higher fatigue).'.format(9, ((avg_fatigue_long / max(avg_fatigue_normal, 0.1)) - 1) * 100))

            # Weekly trend
            if len(weekly_data) >= 2:
                max_week = max(weekly_data, key=lambda w: w['avg_burnout'])
                insights.append('"{}" shows the highest average burnout score ({}).'.format(max_week['week'], max_week['avg_burnout']))
                min_sleep_week = min(weekly_data, key=lambda w: w['avg_sleep'])
                if min_sleep_week['avg_sleep'] < 6.5:
                    insights.append('Sleep dropped significantly in {} (avg {:.1f}h).'.format(min_sleep_week['week'], min_sleep_week['avg_sleep']))

            # Stress pattern
            stressed_days = [r for r in records if r['mood'] == 'Stressed']
            if len(stressed_days) > len(records) * 0.4:
                insights.append('Over 40% of days show "Stressed" mood -- consistent stress pattern detected.')

            if not insights:
                insights.append('No significant patterns detected in the selected range.')

            # RECOMMENDATIONS
            recommendations = []
            if risk_level == 'High':
                recommendations.append('Urgently reduce workload for this employee.')
                recommendations.append('Schedule a 1:1 wellness check-in within 48 hours.')
                recommendations.append('Consider granting a mental health day.')
            elif risk_level == 'Medium':
                recommendations.append('Monitor this employee closely over the next week.')
                recommendations.append('Encourage taking regular breaks throughout the day.')
                recommendations.append('Review task distribution to avoid overload.')
            else:
                recommendations.append('Continue positive engagement and support.')
                recommendations.append('Recognize good work-life balance practices.')

            overall = {
                'avg_burnout': avg_burnout,
                'highest_burnout_day': highest_burnout_record['submission_date'],
                'highest_burnout_score': highest_burnout_record['burnout_score'],
                'lowest_mood_day': lowest_mood_record['submission_date'],
                'lowest_mood': lowest_mood_record['mood'],
                'risk_level': risk_level,
                'risk_color': risk_color,
                'total_days': len(records),
                'recommendations': recommendations
            }
        else:
            insights = ['No data available for the selected date range.']
            overall = {'avg_burnout': 0, 'risk_level': 'Low', 'risk_color': 'green', 'total_days': 0, 'recommendations': []}

        trend_data = compute_employee_trend(employee_id)

        return jsonify({
            'employee_id': employee_id,
            'name': emp_row[0] if emp_row else 'Unknown',
            'department': emp_row[1] if emp_row else '',
            'email': emp_row[2] if emp_row else '',
            'records': records,
            'weekly_data': weekly_data,
            'overall': overall,
            'insights': insights,
            'trend': trend_data,
            'pattern_summary': '; '.join(insights[:3]) if insights else 'No data'
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/employee/<int:employee_id>/analysis-plots')
def employee_analysis_plots(employee_id):
    """
    Generate advanced statistical plots using Seaborn/Matplotlib.
    Returns base64 encoded images.
    """
    try:
        token = request.headers.get('Authorization', '').replace('Bearer ', '')
        session = verify_session(token, 'any')
        if not session:
            return jsonify({'error': 'Unauthorized'}), 401

        start_date = request.args.get('startDate', '')
        end_date = request.args.get('endDate', '')

        conn = sqlite3.connect(DB_PATH)
        query = 'SELECT burnout_score, fatigue, work_hours, sleep_hours, stress_level, submission_date FROM records WHERE employee_id = ?'
        params = [employee_id]
        if start_date:
            query += ' AND submission_date >= ?'
            params.append(start_date)
        if end_date:
            query += ' AND submission_date <= ?'
            params.append(end_date)
        
        df = pd.read_sql_query(query, conn, params=params)
        conn.close()

        if df.empty:
            return jsonify({'error': 'No data available for plots'}), 404

        # Clean column names for the plot
        df.columns = ['Burnout', 'Fatigue', 'Work Hours', 'Sleep', 'Stress', 'Date']
        
        # 1. Correlation Heatmap
        plt.figure(figsize=(8, 6))
        sns.set_theme(style="white")
        # Ensure we only correlate numeric columns
        numeric_df = df.select_dtypes(include=[np.number])
        corr = numeric_df.corr()
        
        mask = np.triu(np.ones_like(corr, dtype=bool))
        sns.heatmap(corr, mask=mask, annot=True, cmap="coolwarm", fmt=".2f", linewidths=.5, cbar_kws={"shrink": .8})
        plt.title('Feature Correlation Matrix', fontsize=14, fontweight='bold', pad=20)
        
        img_heatmap = io.BytesIO()
        plt.savefig(img_heatmap, format='png', bbox_inches='tight', dpi=100)
        img_heatmap.seek(0)
        plot_heatmap = base64.b64encode(img_heatmap.getvalue()).decode()
        plt.close()

        # 2. Burnout Distribution vs Fatigue Grouped
        plt.figure(figsize=(10, 5))
        sns.kdeplot(data=df, x="Burnout", fill=True, color="red", label="Burnout Density")
        sns.kdeplot(data=df, x="Fatigue", fill=True, color="orange", label="Fatigue Density")
        plt.title('Score Distribution Analysis', fontsize=14, fontweight='bold')
        plt.xlabel('Risk Value (0-10)')
        plt.legend()
        
        img_dist = io.BytesIO()
        plt.savefig(img_dist, format='png', bbox_inches='tight', dpi=100)
        img_dist.seek(0)
        plot_dist = base64.b64encode(img_dist.getvalue()).decode()
        plt.close()

        # 3. Work Hours vs Sleep Scatter (Regression)
        plt.figure(figsize=(10, 5))
        sns.regplot(data=df, x="Work Hours", y="Sleep", color="blue", scatter_kws={'alpha':0.5})
        plt.title('Work Hours vs Sleep Pattern', fontsize=14, fontweight='bold')
        
        img_reg = io.BytesIO()
        plt.savefig(img_reg, format='png', bbox_inches='tight', dpi=100)
        img_reg.seek(0)
        plot_reg = base64.b64encode(img_reg.getvalue()).decode()
        plt.close()

        return jsonify({
            'heatmap': plot_heatmap,
            'distribution': plot_dist,
            'regression': plot_reg
        })

    except Exception as e:
        print(f"Plot Error: {e}")
        return jsonify({'error': str(e)}), 500


# ============================================================
# CORE: Password Reset Routes
# ============================================================

@app.route('/api/auth/forgot-password', methods=['POST'])
def forgot_password():
    data = request.json
    email = data.get('email')
    
    if not email:
        return jsonify({'error': 'Email is required'}), 400

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # Check if user exists (managers or employees)
    cursor.execute('SELECT id FROM managers WHERE email = ?', (email,))
    user = cursor.fetchone()
    if not user:
        cursor.execute('SELECT id FROM employees WHERE email = ?', (email,))
        user = cursor.fetchone()
        
    if not user:
        conn.close()
        # Return success anyway to prevent email enumeration attacks
        return jsonify({'message': 'Reset link sent to your email'})

    # Generate secure token
    token = secrets.token_hex(32)
    expires_at = datetime.now() + timedelta(minutes=15)
    
    cursor.execute('INSERT INTO password_resets (token, email, expires_at) VALUES (?, ?, ?)',
                  (token, email, expires_at.strftime('%Y-%m-%d %H:%M:%S')))
    conn.commit()
    conn.close()

    # Send Email securely
    try:
        sender_email = os.getenv('SMTP_SENDER_EMAIL')
        sender_password = os.getenv('SMTP_APP_PASSWORD')
        
        if sender_email and sender_password:
            msg = MIMEMultipart()
            msg['From'] = sender_email
            msg['To'] = email
            msg['Subject'] = 'Password Reset - Burnout Intelligence'
            
            reset_link = f"http://localhost:3000/reset-password/{token}"
            body = f"Click the link below to reset your password:\n\n{reset_link}\n\nThis link expires in 15 minutes."
            
            msg.attach(MIMEText(body, 'plain'))
            
            server = smtplib.SMTP('smtp.gmail.com', 587)
            server.starttls()
            server.login(sender_email, sender_password)
            server.send_message(msg)
            server.quit()
        else:
            print(f"Warning: SMTP credentials missing. Reset link for {email}: http://localhost:3000/reset-password/{token}")
    except Exception as e:
        print(f"SMTP Error: {e}")
        return jsonify({'error': 'Failed to send email. Check credentials.'}), 500

    return jsonify({'message': 'Reset link sent to your email'})


@app.route('/api/auth/reset-password/<token>', methods=['POST'])
def reset_password(token):
    data = request.json
    new_password = data.get('password')
    
    if not new_password:
        return jsonify({'error': 'Password is required'}), 400

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    cursor.execute('SELECT email, expires_at FROM password_resets WHERE token = ?', (token,))
    reset_record = cursor.fetchone()
    
    if not reset_record:
        conn.close()
        return jsonify({'error': 'Invalid or expired token'}), 400
        
    email, expires_at_str = reset_record
    expires_at = datetime.strptime(expires_at_str, '%Y-%m-%d %H:%M:%S')
    
    if datetime.now() > expires_at:
        cursor.execute('DELETE FROM password_resets WHERE token = ?', (token,))
        conn.commit()
        conn.close()
        return jsonify({'error': 'Invalid or expired token'}), 400
        
    hashed_pwd = hash_password(new_password)
    
    # Update password in respective table
    cursor.execute('UPDATE managers SET password = ? WHERE email = ?', (hashed_pwd, email))
    if cursor.rowcount == 0:
        cursor.execute('UPDATE employees SET password = ? WHERE email = ?', (hashed_pwd, email))
        
    # Invalidate token
    cursor.execute('DELETE FROM password_resets WHERE token = ?', (token,))
    
    conn.commit()
    conn.close()
    
    return jsonify({'message': 'Password reset successful'})


if __name__ == '__main__':
    app.run(debug=True, port=5000)
