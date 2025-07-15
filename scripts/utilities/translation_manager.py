#!/usr/bin/env python3
"""
Translation Manager - Clean, one-time translation system
Translates all languages except English and saves for reuse
"""

import pandas as pd
import json
import time
from datetime import datetime
from pathlib import Path
from deep_translator import GoogleTranslator
import numpy as np

class TranslationManager:
    def __init__(self):
        self.translations_dir = Path("translations")
        self.translations_dir.mkdir(exist_ok=True)
        
        self.language_codes = {
            'basque': 'eu',
            'hebrew': 'auto',  # Use auto-detection for Hebrew
            'serbian': 'sr',
            'slovenian': 'sl'
        }
        
        # Status tracking
        self.translation_status = {
            'total_dreams': 0,
            'translated_dreams': 0,
            'skipped_dreams': 0,
            'failed_dreams': 0,
            'languages_processed': [],
            'start_time': None,
            'end_time': None
        }
        
        print("🔧 Translation Manager initialized")
        print(f"📁 Translations directory: {self.translations_dir}")
    
    def find_all_sessions(self):
        """Find all available sessions with dream data"""
        languages = ['english', 'basque', 'hebrew', 'serbian', 'slovenian']
        all_sessions = set()
        
        for lang in languages:
            lang_dir = Path(f'logs/{lang}/gpt-4o')
            if lang_dir.exists():
                for session_dir in lang_dir.iterdir():
                    if session_dir.is_dir() and session_dir.name.startswith('session_'):
                        all_sessions.add(session_dir.name)
        
        if not all_sessions:
            raise Exception("No dream sessions found!")
        
        sorted_sessions = sorted(all_sessions)
        print(f"📊 Found {len(sorted_sessions)} sessions: {sorted_sessions}")
        return sorted_sessions
    
    def find_latest_session(self):
        """Find the most recent session with dream data"""
        all_sessions = self.find_all_sessions()
        latest_session = all_sessions[-1]
        print(f"📊 Using latest session: {latest_session}")
        return latest_session
    
    def check_existing_translations(self, session_id):
        """Check what translations already exist"""
        existing = {}
        languages = ['english', 'basque', 'hebrew', 'serbian', 'slovenian']
        
        for lang in languages:
            translation_file = self.translations_dir / f"{lang}_translations_{session_id}.json"
            if translation_file.exists():
                try:
                    with open(translation_file, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                    existing[lang] = len(data['dreams'])
                    print(f"  ✅ {lang}: {len(data['dreams'])} translations found")
                except Exception as e:
                    print(f"  ⚠️ {lang}: Error reading existing translations - {e}")
                    existing[lang] = 0
            else:
                existing[lang] = 0
                print(f"  ❌ {lang}: No translations found")
        
        return existing
    
    def translate_text(self, text, source_lang, max_retries=3):
        """Translate text using Google Translate with error handling"""
        if not text or len(text.strip()) < 3:
            return text
        
        for attempt in range(max_retries):
            try:
                translator = GoogleTranslator(source=source_lang, target='en')
                
                # Handle long text by splitting into chunks
                max_chunk_size = 4500
                if len(text) <= max_chunk_size:
                    translated = translator.translate(text)
                    return translated if translated else text
                else:
                    # Split into sentences and translate in chunks
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
                                time.sleep(0.1)  # Rate limiting
                            current_chunk = sentence + ". "
                    
                    # Translate final chunk
                    if current_chunk:
                        chunk_translation = translator.translate(current_chunk.strip())
                        translated_chunks.append(chunk_translation if chunk_translation else current_chunk)
                    
                    return " ".join(translated_chunks)
                    
            except Exception as e:
                print(f"    ⚠️ Translation attempt {attempt + 1} failed: {str(e)}")
                if attempt < max_retries - 1:
                    time.sleep(1)  # Wait before retry
                else:
                    print(f"    ❌ Translation failed after {max_retries} attempts")
                    return text
        
        return text
    
    def load_dreams_for_language(self, language, session_id):
        """Load dreams for a specific language"""
        dreams_file = Path(f'logs/{language}/gpt-4o/{session_id}/dreams.csv')
        
        if not dreams_file.exists():
            print(f"    ❌ No dreams file found for {language}")
            return []
        
        try:
            df = pd.read_csv(dreams_file)
            successful_dreams = df[df['status'] == 'success']
            print(f"    📊 Found {len(successful_dreams)} successful dreams")
            
            dreams_data = []
            for _, row in successful_dreams.iterrows():
                dream_data = {
                    'dream_id': row.get('call_id', f"{language}_{len(dreams_data)}"),
                    'original_text': str(row['dream']),
                    'language': language,
                    'timestamp': row.get('timestamp', ''),
                    'session_id': session_id
                }
                dreams_data.append(dream_data)
            
            return dreams_data
            
        except Exception as e:
            print(f"    ❌ Error loading dreams for {language}: {e}")
            return []
    
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
            'translation_method': 'Google Translate (auto-detection)' if language == 'hebrew' else 'Google Translate',
            'dreams': dreams_data
        }
        
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(json_data, f, indent=2, ensure_ascii=False)
        
        # Save as CSV for easy viewing
        csv_file = self.translations_dir / f"{language}_translations_{session_id}.csv"
        csv_data = []
        for dream in dreams_data:
            csv_data.append({
                'dream_id': dream['dream_id'],
                'language': dream['language'],
                'original_text': dream['original_text'],
                'translated_text': dream.get('translated_text', ''),
                'word_count': dream.get('word_count', 0),
                'char_count': dream.get('char_count', 0)
            })
        
        df = pd.DataFrame(csv_data)
        df.to_csv(csv_file, index=False, encoding='utf-8')
        
        print(f"    ✅ Saved {len(dreams_data)} translations")
        print(f"       📄 JSON: {json_file.name}")
        print(f"       📊 CSV: {csv_file.name}")
    
    def translate_all_languages(self, force_retranslate=False):
        """Translate all languages except English (latest session only - legacy method)"""
        print("\n⚠️ Using legacy single-session method. Consider using translate_new_batches_only() or translate_all_sessions()")
        
        # Find latest session and translate it
        session_id = self.find_latest_session()
        return self.translate_specific_session(session_id, force_retranslate)
    
    def generate_translation_summary(self, session_id):
        """Generate a comprehensive summary report"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        summary_file = self.translations_dir / f"translation_summary_{session_id}_{timestamp}.md"
        
        duration = self.translation_status['end_time'] - self.translation_status['start_time']
        
        with open(summary_file, 'w', encoding='utf-8') as f:
            f.write("# Comprehensive Dream Translation Summary\n\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Session: {session_id}\n")
            f.write(f"Duration: {duration}\n\n")
            
            f.write("## Translation Results\n\n")
            f.write(f"- **Total Dreams**: {self.translation_status['total_dreams']}\n")
            f.write(f"- **Translated**: {self.translation_status['translated_dreams']}\n")
            f.write(f"- **Skipped**: {self.translation_status['skipped_dreams']}\n")
            f.write(f"- **Failed**: {self.translation_status['failed_dreams']}\n")
            f.write(f"- **Success Rate**: {(self.translation_status['translated_dreams'] / max(1, self.translation_status['translated_dreams'] + self.translation_status['failed_dreams'])) * 100:.1f}%\n\n")
            
            f.write("## Languages Processed\n\n")
            for lang_status in self.translation_status['languages_processed']:
                f.write(f"- {lang_status}\n")
            f.write("\n")
            
            f.write("## Files Created\n\n")
            for file_path in sorted(self.translations_dir.glob(f'*_{session_id}.*')):
                if file_path.is_file():
                    f.write(f"- `{file_path.name}`\n")
            f.write("\n")
            
            f.write("## Next Steps\n\n")
            f.write("1. Use translations in thematic analysis\n")
            f.write("2. Apply semantic similarity for better detection\n")
            f.write("3. Run statistical analysis on translated data\n")
            f.write("4. Generate research reports\n\n")
        
        print(f"\n📄 Summary report generated: {summary_file}")
        return summary_file
    
    def get_translation_status(self):
        """Get current translation status"""
        return self.translation_status
    
    def clean_old_translations(self):
        """Clean up old/incomplete translation files"""
        print("\n🧹 Cleaning old translation files...")
        
        deleted_count = 0
        for file_path in self.translations_dir.glob('*_translations_*.json'):
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                # Check if translation is incomplete or corrupted
                if data.get('total_dreams', 0) == 0 or not data.get('dreams'):
                    print(f"    🗑️ Deleting incomplete: {file_path.name}")
                    file_path.unlink()
                    
                    # Also delete corresponding CSV
                    csv_path = file_path.with_suffix('.csv')
                    if csv_path.exists():
                        csv_path.unlink()
                    
                    deleted_count += 1
                    
            except Exception as e:
                print(f"    ❌ Error checking {file_path.name}: {e}")
        
        print(f"    ✅ Cleaned {deleted_count} incomplete translation files")
        return deleted_count

    def find_untranslated_sessions(self):
        """Find sessions that haven't been translated yet"""
        all_sessions = self.find_all_sessions()
        untranslated_sessions = []
        
        for session_id in all_sessions:
            existing_translations = self.check_existing_translations(session_id)
            # Check if any non-English language has translations
            non_english_translated = sum(1 for lang, count in existing_translations.items() 
                                       if lang != 'english' and count > 0)
            
            if non_english_translated == 0:
                untranslated_sessions.append(session_id)
        
        print(f"🔍 Found {len(untranslated_sessions)} untranslated sessions: {untranslated_sessions}")
        return untranslated_sessions
    
    def translate_specific_session(self, session_id, force_retranslate=False):
        """Translate a specific session"""
        print(f"\n🎯 Translating specific session: {session_id}")
        
        # Check existing translations
        existing_translations = self.check_existing_translations(session_id)
        
        # Languages to translate (excluding English)
        languages_to_translate = ['basque', 'hebrew', 'serbian', 'slovenian']
        
        session_status = {
            'session_id': session_id,
            'translated_dreams': 0,
            'skipped_dreams': 0,
            'failed_dreams': 0,
            'languages_processed': []
        }
        
        for language in languages_to_translate:
            print(f"\n🔄 Processing {language.upper()} for session {session_id}...")
            
            # Check if translation already exists and is complete
            if not force_retranslate and existing_translations.get(language, 0) > 0:
                print(f"    ✅ Translations already exist ({existing_translations[language]} dreams)")
                print(f"    ⏭️ Skipping {language} (use force_retranslate=True to override)")
                session_status['skipped_dreams'] += existing_translations[language]
                session_status['languages_processed'].append(f"{language} (skipped)")
                continue
            
            # Load dreams for this language
            dreams_data = self.load_dreams_for_language(language, session_id)
            if not dreams_data:
                continue
            
            # Translate each dream
            translated_count = 0
            failed_count = 0
            
            for i, dream in enumerate(dreams_data, 1):
                print(f"    🔄 Translating {language} dream {i}/{len(dreams_data)}...")
                
                try:
                    # Translate the dream text
                    translated_text = self.translate_text(
                        dream['original_text'], 
                        self.language_codes[language]
                    )
                    
                    # Add translation data to dream
                    dream['translated_text'] = translated_text
                    dream['word_count'] = len(translated_text.split())
                    dream['char_count'] = len(translated_text)
                    
                    translated_count += 1
                    session_status['translated_dreams'] += 1
                    
                    # Rate limiting
                    time.sleep(0.2)
                    
                except Exception as e:
                    print(f"    ❌ Failed to translate dream {i}: {e}")
                    dream['translated_text'] = dream['original_text']  # Fallback to original
                    dream['word_count'] = len(dream['original_text'].split())
                    dream['char_count'] = len(dream['original_text'])
                    failed_count += 1
                    session_status['failed_dreams'] += 1
            
            # Save translations for this language
            self.save_translations(language, dreams_data, session_id)
            
            # Update status
            session_status['languages_processed'].append(f"{language} ({translated_count} translated, {failed_count} failed)")
            
            print(f"    ✅ {language.upper()} COMPLETE: {translated_count} translated, {failed_count} failed")
        
        # Process English (no translation needed)
        print(f"\n📄 Processing ENGLISH for session {session_id} (native)...")
        english_dreams = self.load_dreams_for_language('english', session_id)
        if english_dreams:
            # Add English data without translation
            for dream in english_dreams:
                dream['translated_text'] = None  # No translation needed
                dream['word_count'] = len(dream['original_text'].split())
                dream['char_count'] = len(dream['original_text'])
            
            self.save_translations('english', english_dreams, session_id)
            session_status['languages_processed'].append(f"english ({len(english_dreams)} native)")
            print(f"    ✅ ENGLISH COMPLETE: {len(english_dreams)} dreams (native)")
        
        print(f"\n🎯 Session {session_id} completed!")
        return session_status
    
    def translate_all_sessions(self, force_retranslate=False):
        """Translate all available sessions"""
        self.translation_status['start_time'] = datetime.now()
        
        print("\n🌍 STARTING COMPREHENSIVE TRANSLATION - ALL SESSIONS")
        print("=" * 60)
        
        # Find all sessions
        all_sessions = self.find_all_sessions()
        
        # Process each session
        all_session_results = []
        
        for session_id in all_sessions:
            session_result = self.translate_specific_session(session_id, force_retranslate)
            all_session_results.append(session_result)
            
            # Update overall status
            self.translation_status['translated_dreams'] += session_result['translated_dreams']
            self.translation_status['skipped_dreams'] += session_result['skipped_dreams']
            self.translation_status['failed_dreams'] += session_result['failed_dreams']
        
        # Calculate total dreams across all sessions
        total_dreams = 0
        for session_id in all_sessions:
            languages = ['english', 'basque', 'hebrew', 'serbian', 'slovenian']
            for lang in languages:
                dreams = self.load_dreams_for_language(lang, session_id)
                total_dreams += len(dreams)
        
        self.translation_status['total_dreams'] = total_dreams
        self.translation_status['languages_processed'] = [f"All sessions: {len(all_sessions)} sessions processed"]
        
        # Final status
        self.translation_status['end_time'] = datetime.now()
        
        # Generate summary for all sessions
        self.generate_multi_session_summary(all_session_results)
        
        return self.translation_status
    
    def translate_new_batches_only(self):
        """Translate only new/untranslated sessions"""
        print("\n🆕 TRANSLATING NEW BATCHES ONLY")
        print("=" * 40)
        
        # Find untranslated sessions
        untranslated_sessions = self.find_untranslated_sessions()
        
        if not untranslated_sessions:
            print("✅ No new batches to translate - all sessions are already translated!")
            return {
                'total_dreams': 0,
                'translated_dreams': 0,
                'skipped_dreams': 0,
                'failed_dreams': 0,
                'new_sessions_processed': 0
            }
        
        # Translate each untranslated session
        self.translation_status['start_time'] = datetime.now()
        all_session_results = []
        
        for session_id in untranslated_sessions:
            print(f"\n🔄 Translating new batch: {session_id}")
            session_result = self.translate_specific_session(session_id, force_retranslate=False)
            all_session_results.append(session_result)
            
            # Update overall status
            self.translation_status['translated_dreams'] += session_result['translated_dreams']
            self.translation_status['skipped_dreams'] += session_result['skipped_dreams']
            self.translation_status['failed_dreams'] += session_result['failed_dreams']
        
        # Calculate total dreams for new sessions
        total_dreams = 0
        for session_id in untranslated_sessions:
            languages = ['english', 'basque', 'hebrew', 'serbian', 'slovenian']
            for lang in languages:
                dreams = self.load_dreams_for_language(lang, session_id)
                total_dreams += len(dreams)
        
        self.translation_status['total_dreams'] = total_dreams
        self.translation_status['languages_processed'] = [f"New batches: {len(untranslated_sessions)} sessions processed"]
        self.translation_status['new_sessions_processed'] = len(untranslated_sessions)
        
        # Final status
        self.translation_status['end_time'] = datetime.now()
        
        # Generate summary for new sessions
        self.generate_multi_session_summary(all_session_results, batch_type="new")
        
        print(f"\n🎉 NEW BATCH TRANSLATION COMPLETE!")
        print(f"✅ Processed {len(untranslated_sessions)} new sessions")
        
        return self.translation_status
    
    def generate_multi_session_summary(self, session_results, batch_type="all"):
        """Generate a comprehensive summary report for multiple sessions"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        summary_file = self.translations_dir / f"multi_session_summary_{batch_type}_{timestamp}.md"
        
        duration = self.translation_status['end_time'] - self.translation_status['start_time']
        
        with open(summary_file, 'w', encoding='utf-8') as f:
            f.write(f"# Multi-Session Dream Translation Summary ({batch_type.title()})\n\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Batch Type: {batch_type.title()}\n")
            f.write(f"Duration: {duration}\n\n")
            
            f.write("## Overall Translation Results\n\n")
            f.write(f"- **Total Dreams**: {self.translation_status['total_dreams']}\n")
            f.write(f"- **Translated**: {self.translation_status['translated_dreams']}\n")
            f.write(f"- **Skipped**: {self.translation_status['skipped_dreams']}\n")
            f.write(f"- **Failed**: {self.translation_status['failed_dreams']}\n")
            f.write(f"- **Sessions Processed**: {len(session_results)}\n")
            if 'new_sessions_processed' in self.translation_status:
                f.write(f"- **New Sessions**: {self.translation_status['new_sessions_processed']}\n")
            f.write(f"- **Success Rate**: {(self.translation_status['translated_dreams'] / max(1, self.translation_status['translated_dreams'] + self.translation_status['failed_dreams'])) * 100:.1f}%\n\n")
            
            f.write("## Session-by-Session Results\n\n")
            for session_result in session_results:
                f.write(f"### Session: {session_result['session_id']}\n")
                f.write(f"- Translated: {session_result['translated_dreams']}\n")
                f.write(f"- Skipped: {session_result['skipped_dreams']}\n")
                f.write(f"- Failed: {session_result['failed_dreams']}\n")
                f.write(f"- Languages: {', '.join(session_result['languages_processed'])}\n\n")
            
            f.write("## Files Created\n\n")
            for file_path in sorted(self.translations_dir.glob('*_translations_*.json')):
                if file_path.is_file():
                    f.write(f"- `{file_path.name}`\n")
            f.write("\n")
            
            f.write("## Usage Instructions\n\n")
            f.write("1. **Thematic Analysis**: Use translations for improved theme detection\n")
            f.write("2. **Semantic Similarity**: Apply advanced NLP techniques\n")
            f.write("3. **Statistical Analysis**: Run cross-cultural comparisons\n")
            f.write("4. **New Batches**: Use `translate_new_batches_only()` for incremental updates\n\n")
        
        print(f"\n📄 Multi-session summary generated: {summary_file}")
        return summary_file

def main():
    """Test the translation manager with new batch capabilities"""
    print("🚀 DREAM TRANSLATION MANAGER - BATCH CAPABLE")
    print("=" * 50)
    
    manager = TranslationManager()
    
    # Clean old translations first
    manager.clean_old_translations()
    
    # Show all available options
    print("\n🔍 SCANNING FOR SESSIONS...")
    try:
        all_sessions = manager.find_all_sessions()
        untranslated_sessions = manager.find_untranslated_sessions()
        
        print(f"\n📊 SESSION OVERVIEW:")
        print(f"   Total sessions found: {len(all_sessions)}")
        print(f"   Untranslated sessions: {len(untranslated_sessions)}")
        
        if untranslated_sessions:
            print(f"\n🆕 PROCESSING NEW BATCHES ONLY...")
            # Translate only new batches
            status = manager.translate_new_batches_only()
            
            print(f"\n🎯 NEW BATCH SUMMARY:")
            print(f"✅ Total dreams processed: {status['total_dreams']}")
            print(f"🔄 Newly translated: {status['translated_dreams']}")
            print(f"⏭️ Skipped (existing): {status['skipped_dreams']}")
            print(f"❌ Failed: {status['failed_dreams']}")
            print(f"🆕 New sessions processed: {status.get('new_sessions_processed', 0)}")
            
        else:
            print(f"\n✅ All sessions already translated!")
            print(f"   Use force_retranslate=True to retranslate existing sessions")
            
            # Show status of existing translations
            for session_id in all_sessions:
                existing = manager.check_existing_translations(session_id)
                print(f"\n   Session {session_id}:")
                for lang, count in existing.items():
                    status_icon = "✅" if count > 0 else "❌"
                    print(f"     {status_icon} {lang}: {count} translations")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return
    
    print(f"\n📁 All translations saved to: {manager.translations_dir}")
    print(f"🎉 Ready for thematic analysis with semantic similarity!")
    
    print(f"\n🔧 USAGE EXAMPLES:")
    print(f"   # Translate only new batches:")
    print(f"   manager.translate_new_batches_only()")
    print(f"   ")
    print(f"   # Translate specific session:")
    print(f"   manager.translate_specific_session('session_20250706_093349')")
    print(f"   ")
    print(f"   # Translate all sessions:")
    print(f"   manager.translate_all_sessions(force_retranslate=True)")

if __name__ == "__main__":
    main() 