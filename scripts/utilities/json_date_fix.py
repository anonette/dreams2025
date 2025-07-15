#!/usr/bin/env python3
"""
Fix for JSON serialization issue with date objects in generate_research_report.py
"""

import json
from datetime import datetime, date
import pandas as pd

class DateTimeEncoder(json.JSONEncoder):
    """Custom JSON encoder that handles datetime and date objects."""
    
    def default(self, obj):
        if isinstance(obj, (datetime, date)):
            return obj.isoformat()
        elif isinstance(obj, pd.Timestamp):
            return obj.isoformat()
        elif hasattr(obj, 'item'):  # numpy types
            return obj.item()
        elif hasattr(obj, 'tolist'):  # numpy arrays
            return obj.tolist()
        return super().default(obj)

def convert_dates_to_strings(data):
    """Recursively convert date objects to strings in nested dictionaries."""
    if isinstance(data, dict):
        return {
            str(k) if isinstance(k, (datetime, date)) else k: convert_dates_to_strings(v)
            for k, v in data.items()
        }
    elif isinstance(data, list):
        return [convert_dates_to_strings(item) for item in data]
    elif isinstance(data, (datetime, date)):
        return data.isoformat()
    elif isinstance(data, pd.Timestamp):
        return data.isoformat()
    else:
        return data

# Monkey patch the json module to use our encoder
original_dumps = json.dumps
json.dumps = lambda obj, **kwargs: original_dumps(obj, cls=DateTimeEncoder, **kwargs)

print("JSON date serialization fix applied!")
print("Now run: python generate_research_report.py --sessions 20250705_194838") 