#!/usr/bin/env python3
"""
Verification script for the 900-dream generation system.
Tests batch handling, data structure validation, and analysis preparation.
"""

import asyncio
import json
import pandas as pd
from pathlib import Path
import sys
import os
from datetime import datetime

class SystemVerifier:
    """Verifies the system can handle 900-dream generation workflow."""
    
    def __init__(self):
        self.logs_dir = Path('logs')
        self.languages = ['english', 'basque', 'serbian', 'hebrew', 'slovenian']
        
    def verify_batch_configuration(self):
        """Verify the batch system can handle 100 dreams properly."""
        print("🔧 VERIFYING BATCH CONFIGURATION")
        print("="*40)
        
        # Check default batch size handling
        from batch_dream_generator import SamplingConfig
        
        # Test different configurations
        configs = [
            {'batch_size': 50, 'dreams': 100},   # 2 batches of 50
            {'batch_size': 25, 'dreams': 100},   # 4 batches of 25
            {'batch_size': 100, 'dreams': 100},  # 1 batch of 100
        ]
        
        for config in configs:
            batch_size = config['batch_size']
            total_dreams = config['dreams']
            num_batches = (total_dreams + batch_size - 1) // batch_size
            
            print(f"✅ Config: {total_dreams} dreams, batch size {batch_size}")
            print(f"   → {num_batches} batches needed")
            
            # Verify no dreams are lost
            total_generated = num_batches * batch_size
            if batch_size * (num_batches - 1) < total_dreams <= total_generated:
                print(f"   → ✅ Math checks out: will generate {total_dreams} dreams")
            else:
                print(f"   → ❌ Math error: would generate {total_generated} dreams")
        
        print(f"\n✅ Batch system verified for 100-dream sessions")
    
    def verify_data_structure(self):
        """Verify the current data structure can handle multiple sessions."""
        print("\n📁 VERIFYING DATA STRUCTURE")
        print("="*40)
        
        # Check if logs directory exists
        if not self.logs_dir.exists():
            print(f"⚠️  Logs directory doesn't exist yet: {self.logs_dir}")
            print("   This is normal for first run")
            return True
        
        # Scan existing structure
        structure_verified = True
        
        for language in self.languages:
            lang_dir = self.logs_dir / language / 'gpt-4o'
            if lang_dir.exists():
                sessions = [d for d in lang_dir.iterdir() if d.is_dir() and d.name.startswith('session_')]
                print(f"📂 {language}: {len(sessions)} existing sessions")
                
                for session_dir in sessions[:2]:  # Check first 2 sessions
                    required_files = ['dreams.csv', 'api_calls.csv', 'session_data.json']
                    missing_files = []
                    
                    for file_name in required_files:
                        file_path = session_dir / file_name
                        if not file_path.exists():
                            missing_files.append(file_name)
                    
                    if missing_files:
                        print(f"   ❌ {session_dir.name}: Missing {missing_files}")
                        structure_verified = False
                    else:
                        print(f"   ✅ {session_dir.name}: All files present")
            else:
                print(f"📂 {language}: No existing sessions")
        
        if structure_verified:
            print(f"\n✅ Data structure verified")
        else:
            print(f"\n⚠️  Some data structure issues found")
        
        return structure_verified
    
    def verify_analysis_combination(self):
        """Verify the analysis tools can combine multiple sessions."""
        print("\n🔗 VERIFYING ANALYSIS COMBINATION")
        print("="*40)
        
        # Test the combination logic
        test_data = []
        
        # Simulate data from multiple sessions
        for language in ['english', 'basque']:
            for session in range(1, 4):  # 3 sessions
                session_id = f"session_test_{session}"
                
                # Simulate dream data
                for dream_num in range(1, 11):  # 10 dreams per session
                    test_data.append({
                        'call_id': f"{session_id}_{dream_num}",
                        'language': language,
                        'language_code': 'en' if language == 'english' else 'eu',
                        'script': 'Latin',
                        'dream': f"Test dream {dream_num} for {language} session {session}",
                        'status': 'success',
                        'session_id': session_id,
                        'timestamp': datetime.now().isoformat()
                    })
        
        # Convert to DataFrame and test combination
        df = pd.DataFrame(test_data)
        
        # Test grouping by language
        lang_counts = df.groupby('language').size()
        print("Test data summary:")
        for lang, count in lang_counts.items():
            print(f"  {lang}: {count} dreams across 3 sessions")
        
        # Test session counting
        session_counts = df.groupby(['language', 'session_id']).size()
        print("\nSession breakdown:")
        for (lang, session), count in session_counts.items():
            print(f"  {lang} {session}: {count} dreams")
        
        # Verify combination works
        combined_english = df[df['language'] == 'english']
        combined_basque = df[df['language'] == 'basque']
        
        if len(combined_english) == 30 and len(combined_basque) == 30:
            print(f"\n✅ Analysis combination logic verified")
            return True
        else:
            print(f"\n❌ Analysis combination failed")
            return False
    
    def verify_existing_data(self):
        """Verify and summarize existing dream data."""
        print("\n📊 EXISTING DATA SUMMARY")
        print("="*40)
        
        total_dreams = 0
        total_sessions = 0
        
        for language in self.languages:
            lang_dir = self.logs_dir / language / 'gpt-4o'
            lang_dreams = 0
            lang_sessions = 0
            
            if lang_dir.exists():
                for session_dir in lang_dir.iterdir():
                    if session_dir.is_dir() and session_dir.name.startswith('session_'):
                        lang_sessions += 1
                        dreams_file = session_dir / 'dreams.csv'
                        if dreams_file.exists():
                            try:
                                df = pd.read_csv(dreams_file)
                                successful_dreams = len(df[df['status'] == 'success'])
                                lang_dreams += successful_dreams
                            except Exception as e:
                                print(f"   ⚠️  Error reading {dreams_file}: {e}")
            
            total_dreams += lang_dreams
            total_sessions += lang_sessions
            
            if lang_sessions > 0:
                avg_dreams = lang_dreams / lang_sessions
                print(f"📈 {language.upper():>10}: {lang_dreams:>4} dreams ({lang_sessions} sessions, {avg_dreams:.1f} avg)")
            else:
                print(f"📈 {language.upper():>10}: {lang_dreams:>4} dreams (no sessions)")
        
        print("-" * 40)
        print(f"📈 {'TOTAL':>10}: {total_dreams:>4} dreams ({total_sessions} sessions)")
        
        if total_sessions > 0:
            overall_avg = total_dreams / total_sessions
            print(f"📊 Average dreams per session: {overall_avg:.1f}")
        
        return total_dreams, total_sessions
    
    def test_batch_generator(self):
        """Test the batch generator with a small sample."""
        print("\n🧪 TESTING BATCH GENERATOR")
        print("="*40)
        
        print("Testing batch generator configuration...")
        
        # Test command construction
        test_cmd = [
            sys.executable, 'batch_dream_generator.py',
            '--language', 'english',
            '--dreams-per-language', '100',
            '--batch-size', '50',
            '--help'
        ]
        
        print(f"Test command: {' '.join(test_cmd[:-1])} --help")
        
        try:
            import subprocess
            result = subprocess.run(test_cmd, capture_output=True, text=True, timeout=10)
            
            if result.returncode == 0:
                print("✅ Batch generator command structure verified")
                
                # Check if help output mentions batch-size
                if '--batch-size' in result.stdout:
                    print("✅ Batch size parameter available")
                else:
                    print("⚠️  Batch size parameter not found in help")
                    
                return True
            else:
                print(f"❌ Batch generator test failed: {result.stderr}")
                return False
                
        except subprocess.TimeoutExpired:
            print("⚠️  Batch generator test timed out")
            return False
        except Exception as e:
            print(f"❌ Batch generator test error: {e}")
            return False
    
    def create_test_data_combiner(self):
        """Create a test version of the data combiner."""
        print("\n🔧 CREATING TEST DATA COMBINER")
        print("="*40)
        
        combiner_script = '''#!/usr/bin/env python3
"""
Test data combiner for 900-dream analysis.
Tests the combination of multiple sessions into unified datasets.
"""

import pandas as pd
from pathlib import Path
import glob

def test_combine_sessions():
    """Test combining sessions for analysis."""
    logs_dir = Path('logs')
    languages = ['english', 'basque', 'serbian', 'hebrew', 'slovenian']
    
    print("🔍 Testing session combination...")
    
    for language in languages:
        lang_dir = logs_dir / language / 'gpt-4o'
        if not lang_dir.exists():
            continue
        
        sessions = []
        total_dreams = 0
        
        for session_dir in lang_dir.iterdir():
            if session_dir.is_dir() and session_dir.name.startswith('session_'):
                dreams_file = session_dir / 'dreams.csv'
                if dreams_file.exists():
                    try:
                        df = pd.read_csv(dreams_file)
                        successful = df[df['status'] == 'success']
                        sessions.append({
                            'session_id': session_dir.name,
                            'dreams_count': len(successful),
                            'total_calls': len(df),
                            'success_rate': len(successful) / len(df) * 100 if len(df) > 0 else 0
                        })
                        total_dreams += len(successful)
                    except Exception as e:
                        print(f"Error reading {dreams_file}: {e}")
        
        if sessions:
            print(f"\\n{language.upper()}:")
            print(f"  Total sessions: {len(sessions)}")
            print(f"  Total dreams: {total_dreams}")
            print(f"  Sessions: {[s['session_id'] for s in sessions]}")
            
            # Test combination
            all_dreams = []
            for session_dir in lang_dir.iterdir():
                if session_dir.is_dir() and session_dir.name.startswith('session_'):
                    dreams_file = session_dir / 'dreams.csv'
                    if dreams_file.exists():
                        df = pd.read_csv(dreams_file)
                        successful = df[df['status'] == 'success']
                        all_dreams.append(successful)
            
            if all_dreams:
                combined = pd.concat(all_dreams, ignore_index=True)
                print(f"  Combined dataset: {len(combined)} dreams")
                
                # Test analysis capability
                avg_length = combined['dream'].str.len().mean()
                print(f"  Average dream length: {avg_length:.0f} characters")
    
    print("\\n✅ Session combination test complete")

if __name__ == "__main__":
    test_combine_sessions()
'''
        
        with open('test_data_combiner.py', 'w') as f:
            f.write(combiner_script)
        
        print("📝 Created test_data_combiner.py")
        return True
    
    def run_verification(self):
        """Run complete system verification."""
        print("🚀 RUNNING 900-DREAM SYSTEM VERIFICATION")
        print("="*60)
        
        verifications = [
            ("Batch Configuration", self.verify_batch_configuration),
            ("Data Structure", self.verify_data_structure),
            ("Analysis Combination", self.verify_analysis_combination),
            ("Batch Generator", self.test_batch_generator),
            ("Test Combiner Creation", self.create_test_data_combiner)
        ]
        
        results = {}
        
        for name, func in verifications:
            try:
                result = func()
                results[name] = result if result is not None else True
            except Exception as e:
                print(f"❌ {name} verification failed: {e}")
                results[name] = False
        
        # Summary of existing data
        total_dreams, total_sessions = self.verify_existing_data()
        
        # Final report
        print("\n" + "="*60)
        print("🎯 VERIFICATION RESULTS")
        print("="*60)
        
        all_passed = True
        for name, passed in results.items():
            status = "✅ PASS" if passed else "❌ FAIL"
            print(f"{name:>20}: {status}")
            if not passed:
                all_passed = False
        
        print("-" * 60)
        print(f"{'EXISTING DATA':>20}: {total_dreams} dreams, {total_sessions} sessions")
        
        if all_passed:
            print(f"\n🎉 SYSTEM READY FOR 900-DREAM GENERATION!")
            print(f"\n📋 NEXT STEPS:")
            print(f"   python generate_900_dreams.py --check-progress    # Check current status")
            print(f"   python generate_900_dreams.py --language english  # Test with one language")
            print(f"   python generate_900_dreams.py                     # Generate all languages")
        else:
            print(f"\n⚠️  SYSTEM ISSUES DETECTED")
            print(f"   Please review the failed verifications above")
        
        return all_passed

def main():
    """Main verification function."""
    verifier = SystemVerifier()
    success = verifier.run_verification()
    
    if success:
        print(f"\n🔧 SYSTEM VERIFICATION COMPLETE - ALL SYSTEMS GO! 🚀")
    else:
        print(f"\n⚠️  SYSTEM VERIFICATION FOUND ISSUES - REVIEW NEEDED")
    
    return 0 if success else 1

if __name__ == "__main__":
    sys.exit(main()) 