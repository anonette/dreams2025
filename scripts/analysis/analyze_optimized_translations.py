#!/usr/bin/env python3
"""
Create Translations for Optimized Dreams
Specifically designed to work with logs_optimized_v2 directory
"""

import pandas as pd
import re
import json
from pathlib import Path
from datetime import datetime
import time
from deep_translator import GoogleTranslator
import sys

class OptimizedTranslationCreator:
    def __init__(self, logs_dir="logs_optimized_v2"):
        self.logs_dir = Path(logs_dir)
        self.translations_dir = Path('translations')
        self.translations_dir.mkdir(exist_ok=True)
        
        self.language_codes = {
            'english': 'en',
            'basque': 'eu',
            'hebrew': 'auto',  # Use auto-detection for Hebrew
            'serbian': 'sr',
            'slovenian': 'sl'
        }
        
        print(f"🔍 Optimized Translation Creator initialized")
        print(f"📁 Looking for data in: {self.logs_dir}")
        print(f"📝 Translations will be saved to: {self.translations_dir}")
    
    def find_latest_session(self):
        """Find the latest session across all languages"""
        sessions = []
        
        for lang in ['english', 'basque', 'hebrew', 'serbian', 'slovenian']:
            lang_dir = self.logs_dir / lang / 'gpt-4o'
            if lang_dir.exists():
                for session_dir in lang_dir.iterdir():
                    if session_dir.is_dir() and session_dir.name.startswith('session_'):
                        sessions.append(session_dir.name)
        
        if sessions:
            latest = sorted(list(set(sessions)))[-1]
            print(f"📅 Latest session found: {latest}")
            return latest
        else:
            print("❌ No sessions found!")
            return None
    
    def translate_text(self, text, source_lang, target_lang='en', max_retries=3):
        """Translate text using Google Translate"""
        if source_lang == target_lang:
            return text
        
        if not text or len(text.strip()) < 3:
            return text
        
        for attempt in range(max_retries):
            try:
                translator = GoogleTranslator(source=source_lang, target=target_lang)
                
                # Handle long text
                max_chunk_size = 4500
                if len(text) <= max_chunk_size:
                    translated = translator.translate(text)
                    return translated if translated else text
                else:
                    # Split into chunks
                    sentences = text.split('. ')
                    translated_chunks = []
                    current_chunk = ""
                    
                    for sentence in sentences:
                        if len(current_chunk + sentence) < max_chunk_size:
                            current_chunk += sentence + ". "
                        else:
                            if current_chunk:
                                chunk_translation = translator.translate(current_chunk.strip())
                                translated_chunks.append(chunk_translation if chunk_translation else current_chunk)
                                time.sleep(0.1)
                            current_chunk = sentence + ". "
                    
                    if current_chunk:
                        chunk_translation = translator.translate(current_chunk.strip())
                        translated_chunks.append(chunk_translation if chunk_translation else current_chunk)
                    
                    return " ".join(translated_chunks)
                    
            except Exception as e:
                print(f"    ⚠️ Translation attempt {attempt + 1} failed: {str(e)}")
                if attempt < max_retries - 1:
                    time.sleep(1)
                else:
                    print(f"    ❌ Translation failed after {max_retries} attempts")
                    return text
        
        return text
    
    def save_translations(self, language, dreams_data, session_id):
        """Save translations to files"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save as JSON
        json_file = self.translations_dir / f"{language}_translations_{session_id}.json"
        json_data = {
            'language': language,
            'session_id': session_id,
            'timestamp': timestamp,
            'total_dreams': len(dreams_data),
            'dreams': dreams_data
        }
        
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(json_data, f, indent=2, ensure_ascii=False)
        
        # Save as CSV
        csv_file = self.translations_dir / f"{language}_translations_{session_id}.csv"
        csv_data = []
        for dream in dreams_data:
            csv_data.append({
                'dream_id': dream['dream_id'],
                'language': dream['language'],
                'original_text': dream['original_text'],
                'translated_text': dream.get('translated_text', ''),
                'word_count': dream['word_count'],
                'char_count': dream['char_count']
            })
        
        df = pd.DataFrame(csv_data)
        df.to_csv(csv_file, index=False, encoding='utf-8')
        
        print(f"    ✅ Saved {len(dreams_data)} translations")
        print(f"       📄 JSON: {json_file.name}")
        print(f"       📊 CSV: {csv_file.name}")
    
    def check_existing_translations(self, language, session_id):
        """Check if translations already exist"""
        json_file = self.translations_dir / f"{language}_translations_{session_id}.json"
        return json_file.exists()
    
    def create_translations_for_session(self, session_id):
        """Create translations for all languages in a session"""
        languages = ['english', 'basque', 'hebrew', 'serbian', 'slovenian']
        total_translated = 0
        
        print(f"\n🎯 Creating translations for session: {session_id}")
        print(f"📁 Source: {self.logs_dir}")
        
        for lang in languages:
            dreams_file = self.logs_dir / lang / "gpt-4o" / session_id / "dreams.csv"
            
            if not dreams_file.exists():
                print(f"    ❌ No dreams file found for {lang}")
                continue
            
            # Check if translations already exist
            if self.check_existing_translations(lang, session_id):
                print(f"    ⏭️ {lang.upper()}: Translations already exist, skipping")
                continue
            
            print(f"    🔄 Processing {lang.upper()}...")
            
            try:
                df = pd.read_csv(dreams_file)
                successful_dreams = df[df['status'] == 'success']
                print(f"        📊 Found {len(successful_dreams)} successful dreams")
                
                dreams_data = []
                
                for i, (_, row) in enumerate(successful_dreams.iterrows()):
                    original_text = str(row['dream'])
                    
                    if lang != 'english':
                        print(f"        🔄 Translating {lang} dream {i+1}/{len(successful_dreams)}...")
                        translated_text = self.translate_text(
                            original_text,
                            self.language_codes[lang],
                            'en'
                        )
                        total_translated += 1
                        time.sleep(0.2)  # Rate limiting
                    else:
                        translated_text = original_text
                    
                    dream_data = {
                        'dream_id': row.get('call_id', f"{lang}_{i}"),
                        'text': translated_text.lower(),
                        'original_text': original_text,
                        'translated_text': translated_text if lang != 'english' else None,
                        'word_count': len(translated_text.split()),
                        'char_count': len(translated_text),
                        'language': lang
                    }
                    
                    dreams_data.append(dream_data)
                
                # Save translations
                self.save_translations(lang, dreams_data, session_id)
                
                if lang == 'english':
                    print(f"    ✅ {lang.upper()} COMPLETE: {len(dreams_data)} dreams (native)")
                else:
                    print(f"    ✅ {lang.upper()} COMPLETE: {len(dreams_data)} translated, 0 failed")
                
            except Exception as e:
                print(f"    ❌ Error processing {lang}: {e}")
        
        print(f"\n🎉 Translation session complete!")
        print(f"   📊 Total dreams translated: {total_translated}")
        
        return total_translated > 0
    
    def create_summary(self, session_id):
        """Create a summary of the translation session"""
        summary_file = self.translations_dir / f"translation_summary_{session_id}.md"
        
        with open(summary_file, 'w', encoding='utf-8') as f:
            f.write(f"# Translation Summary - {session_id}\n\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Source: {self.logs_dir}\n\n")
            
            f.write("## Translation Files Created\n\n")
            
            for lang in ['english', 'basque', 'hebrew', 'serbian', 'slovenian']:
                json_file = self.translations_dir / f"{lang}_translations_{session_id}.json"
                csv_file = self.translations_dir / f"{lang}_translations_{session_id}.csv"
                
                if json_file.exists():
                    try:
                        with open(json_file, 'r', encoding='utf-8') as jf:
                            data = json.load(jf)
                            dream_count = data.get('total_dreams', 0)
                        
                        f.write(f"### {lang.title()}\n")
                        f.write(f"- **Dreams**: {dream_count}\n")
                        f.write(f"- **JSON**: `{json_file.name}`\n")
                        f.write(f"- **CSV**: `{csv_file.name}`\n")
                        if lang != 'english':
                            f.write(f"- **Method**: Google Translate ({self.language_codes[lang]})\n")
                        else:
                            f.write(f"- **Method**: Native English\n")
                        f.write("\n")
                    except:
                        f.write(f"### {lang.title()}\n")
                        f.write(f"- ❌ Error reading translation data\n\n")
                else:
                    f.write(f"### {lang.title()}\n")
                    f.write(f"- ❌ No translations created\n\n")
            
            f.write("## Usage\n\n")
            f.write("These translation files can now be used by:\n")
            f.write("- Dream Analysis Streamlit app\n")
            f.write("- Thematic analysis scripts\n")
            f.write("- Statistical analysis tools\n\n")
            
            f.write("## File Locations\n\n")
            f.write(f"- **Source Dreams**: `{self.logs_dir}/[language]/gpt-4o/{session_id}/dreams.csv`\n")
            f.write(f"- **Translations**: `{self.translations_dir}/[language]_translations_{session_id}.json`\n")
        
        print(f"📋 Summary created: {summary_file}")

def main():
    print("🌙 Optimized Dream Translation Creator")
    print("=" * 50)
    
    # Allow specifying logs directory
    logs_dir = "logs_optimized_v2"
    if len(sys.argv) > 1:
        logs_dir = sys.argv[1]
    
    creator = OptimizedTranslationCreator(logs_dir)
    
    # Find latest session
    latest_session = creator.find_latest_session()
    if not latest_session:
        print("❌ No sessions found to translate!")
        return
    
    # Create translations
    success = creator.create_translations_for_session(latest_session)
    
    if success:
        creator.create_summary(latest_session)
        print(f"\n✅ Translations complete for {latest_session}!")
        print(f"📁 Files saved to: {creator.translations_dir}")
        print(f"\n💡 You can now analyze this data in the Streamlit app:")
        print(f"   1. Select '{logs_dir}' in the directory dropdown")
        print(f"   2. The app will automatically use the new translations")
    else:
        print(f"\n⚠️ No new translations were created")

if __name__ == "__main__":
    main() 