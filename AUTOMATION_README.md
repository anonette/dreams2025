# Dreams Project Automation Guide

This guide explains how to automate the Dreams research workflow for continuous data collection and analysis.

## 🚀 Quick Start

### One-Time Run
```bash
# Run full pipeline once (50 dreams per language)
python automate_dreams.py --mode once --dreams 50

# Run analysis only on existing data
python automate_dreams.py --mode analysis-only
```

### Scheduled Automation
```bash
# Predefined smart schedule
python automate_dreams.py --schedule

# Daily runs at 2 AM
python automate_dreams.py --mode daily --dreams 25

# Weekly runs (Wednesday 2 AM)
python automate_dreams.py --mode weekly --dreams 100

# Continuous runs every 6 hours
python automate_dreams.py --mode continuous --interval 6 --dreams 25
```

## 🛠️ Automation Features

### Complete Pipeline
1. **Dream Generation**: Generate dreams across all languages
2. **Cultural Analysis**: Run comprehensive Hall-Van de Castle + cultural scripts analysis  
3. **Statistical Analysis**: Generate research-grade statistics
4. **Research Reports**: Create publication-ready reports
5. **Git Commits**: Automatically commit results with timestamps
6. **Logging**: Detailed logs in `automation_logs/`

### Smart Scheduling
- **Daily**: Quick analysis of existing data (no new dreams)
- **Weekly**: Full pipeline with new dream generation 
- **Monthly**: Comprehensive analysis and reporting
- **Continuous**: Custom intervals (hourly, every 6 hours, etc.)

## 📋 Automation Options

### Command Line Arguments

| Argument | Options | Description |
|----------|---------|-------------|
| `--mode` | `once`, `daily`, `weekly`, `continuous`, `analysis-only` | Automation mode |
| `--dreams` | `25`, `50`, `100`, etc. | Dreams per language per run |
| `--languages` | `english`, `basque`, `hebrew`, `serbian`, `slovenian` | Specific languages (default: all) |
| `--interval` | `1`, `6`, `12`, `24` | Hours between runs (continuous mode) |
| `--no-git` | flag | Disable automatic git commits |
| `--schedule` | flag | Use predefined smart schedule |

### Example Commands

```bash
# Quick daily analysis (existing data only)
python automate_dreams.py --mode analysis-only

# Generate 25 dreams per language, analyze, commit
python automate_dreams.py --mode once --dreams 25

# English and Basque only, every 12 hours
python automate_dreams.py --mode continuous --interval 12 --languages english basque --dreams 10

# Weekly runs with 200 dreams per language
python automate_dreams.py --mode weekly --dreams 200

# Smart schedule (daily analysis + weekly generation + monthly reports)
python automate_dreams.py --schedule
```

## 📊 Output Structure

Each automated run creates timestamped analysis in:
```
analysis_output/
├── 20250627/
│   ├── cultural_dream_analysis_104500/  # 10:45:00 run
│   └── cultural_dream_analysis_220300/  # 22:03:00 run
├── 20250628/
│   └── cultural_dream_analysis_093000/  # Next day
└── automation_logs/
    ├── automation_20250627.log
    └── automation_20250628.log
```

## 🔧 Setup for Production

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Set API Keys
```bash
export OPENAI_API_KEY="your-key"
export ANTHROPIC_API_KEY="your-key"
export OPENROUTER_API_KEY="your-key"
```

### 3. Test Setup
```bash
# Test with small run
python automate_dreams.py --mode once --dreams 5 --languages english
```

### 4. Start Automation
```bash
# Recommended: Smart schedule
python automate_dreams.py --schedule

# Or custom schedule
python automate_dreams.py --mode daily --dreams 50
```

## 🖥️ Running as Background Service

### Linux/Mac (systemd)
Create `/etc/systemd/system/dreams-automation.service`:
```ini
[Unit]
Description=Dreams Research Automation
After=network.target

[Service]
Type=simple
User=your-username
WorkingDirectory=/path/to/Dreams
ExecStart=/path/to/python automate_dreams.py --schedule
Restart=always
Environment=OPENAI_API_KEY=your-key
Environment=ANTHROPIC_API_KEY=your-key

[Install]
WantedBy=multi-user.target
```

Enable and start:
```bash
sudo systemctl enable dreams-automation
sudo systemctl start dreams-automation
sudo systemctl status dreams-automation
```

### Windows (Task Scheduler)
1. Open Task Scheduler
2. Create Basic Task
3. Set trigger (daily/weekly)
4. Set action: `python C:\path\to\Dreams\automate_dreams.py --schedule`

### Docker (Recommended)
```dockerfile
FROM python:3.11-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

ENV OPENAI_API_KEY=""
ENV ANTHROPIC_API_KEY=""

CMD ["python", "automate_dreams.py", "--schedule"]
```

Build and run:
```bash
docker build -t dreams-automation .
docker run -d --name dreams-auto \
  -e OPENAI_API_KEY=$OPENAI_API_KEY \
  -e ANTHROPIC_API_KEY=$ANTHROPIC_API_KEY \
  -v $(pwd)/analysis_output:/app/analysis_output \
  -v $(pwd)/logs:/app/logs \
  dreams-automation
```

## 📈 Monitoring Automation

### Log Files
```bash
# View today's automation log
tail -f automation_logs/automation_$(date +%Y%m%d).log

# Check last runs
grep "COMPLETED" automation_logs/*.log
```

### Git History
```bash
# See automated commits
git log --oneline --grep="Automated analysis"

# Check latest analysis files
ls -la analysis_output/$(date +%Y%m%d)/
```

### Health Checks
```bash
# Quick health check
python -c "
import pandas as pd
from pathlib import Path
recent = max(Path('analysis_output').rglob('*.csv'))
df = pd.read_csv(recent)
print(f'Latest analysis: {recent.parent.name}')
print(f'Dreams analyzed: {len(df)}')
print(f'Languages: {df.language.unique()}')
"
```

## ⚡ Performance Optimization

### Resource Management
- **Memory**: ~2GB RAM for 100+ dreams analysis
- **CPU**: Multi-core helpful for TF-IDF clustering
- **Disk**: ~10MB per 100 dreams (CSV + JSON)
- **Network**: API calls ~1-5 min for 100 dreams

### Scheduling Recommendations
- **Research**: Weekly/monthly full runs
- **Continuous Monitoring**: Daily analysis-only
- **High Activity**: Every 6-12 hours with 25-50 dreams
- **Storage Conscious**: Analysis-only mode (no new dreams)

### API Rate Limiting
- OpenAI: Built-in rate limiting with backoff
- Anthropic: Respectful request timing
- Parallel generation across languages

## 🔍 Troubleshooting

### Common Issues

**"No dream data found"**
```bash
# Check logs directory
ls -la logs/*/gpt-4o/
# Run generation first
python batch_dream_generator.py --dreams-per-language 10
```

**Git commit failures**
```bash
# Check git status
git status
# Run without git
python automate_dreams.py --mode once --no-git
```

**API failures**
```bash
# Check API keys
python check_keys.py
# Check rate limits in logs
grep "rate limit" automation_logs/*.log
```

**Memory issues**
- Reduce dreams per run: `--dreams 25`
- Run analysis-only mode
- Check available RAM: `free -h` (Linux) or Activity Monitor (Mac)

### Error Recovery
Automation handles errors gracefully:
- API failures: Retry with backoff
- Partial failures: Continue with available data
- Git issues: Log warnings but don't fail pipeline
- Analysis errors: Skip optional steps

## 📚 Advanced Configuration

### Custom Schedules
```python
# In automate_dreams.py, modify create_scheduled_automation()
schedule.every().tuesday.at("14:30").do(
    automation.run_full_pipeline, languages=["english", "hebrew"]
)
```

### Email Notifications
Add to automation class:
```python
import smtplib
from email.mime.text import MIMEText

def send_notification(self, subject, message):
    # Configure email settings
    # Send completion notifications
```

### Webhook Integration
```python
import requests

def notify_webhook(self, status, results):
    requests.post("https://your-webhook.com/dreams", json={
        "status": status,
        "timestamp": datetime.now().isoformat(),
        "dreams_analyzed": results.get("total_dreams", 0)
    })
```

## 🎯 Production Deployment

### Recommended Setup
1. **Server**: Linux VPS or cloud instance
2. **Monitoring**: systemd service + log rotation
3. **Storage**: Regular backups of `analysis_output/`
4. **Alerts**: Log monitoring for failures
5. **Updates**: Git pulls for code updates

### Security
- Store API keys in environment variables
- Use least-privilege user account
- Regular security updates
- Monitor API usage/costs

### Scaling
- Multiple instances for different language sets
- Database storage for large datasets
- API load balancing for high volume
- Distributed analysis for very large corpora

---

## Quick Commands Reference

| Task | Command |
|------|---------|
| One-time full run | `python automate_dreams.py --mode once --dreams 50` |
| Analysis only | `python automate_dreams.py --mode analysis-only` |
| Smart schedule | `python automate_dreams.py --schedule` |
| Daily automation | `python automate_dreams.py --mode daily --dreams 25` |
| Check logs | `tail -f automation_logs/automation_$(date +%Y%m%d).log` |
| View results | `ls -la analysis_output/$(date +%Y%m%d)/` |

**Happy Automating! 🤖✨**
