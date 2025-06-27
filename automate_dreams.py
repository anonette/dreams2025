#!/usr/bin/env python3
"""
Dreams Project Automation System

This script automates the complete dreams research workflow:
1. Generate dreams across languages
2. Run cultural analysis
3. Generate research reports  
4. Commit results to git
5. Schedule regular runs

Usage:
    python automate_dreams.py --mode daily --dreams 50
    python automate_dreams.py --mode weekly --dreams 200 --full-analysis
    python automate_dreams.py --mode continuous --interval 6 --dreams 25
"""

import argparse
import subprocess
import schedule
import time
import logging
from datetime import datetime, timedelta
from pathlib import Path
import json
import os
import sys

class DreamsAutomation:
    """Automated Dreams research workflow manager"""
    
    def __init__(self, dreams_per_run: int = 50, enable_git: bool = True):
        self.dreams_per_run = dreams_per_run
        self.enable_git = enable_git
        self.setup_logging()
        
    def setup_logging(self):
        """Setup logging for automation tracking"""
        log_dir = Path("automation_logs")
        log_dir.mkdir(exist_ok=True)
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_dir / f"automation_{datetime.now().strftime('%Y%m%d')}.log"),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def run_dream_generation(self, languages: list = None) -> bool:
        """Generate dreams for specified languages"""
        try:
            if languages is None:
                languages = ["english", "basque", "hebrew", "serbian", "slovenian"]
            
            self.logger.info(f"Starting dream generation: {self.dreams_per_run} dreams per language")
            
            # Build command
            cmd = [
                "python", "batch_dream_generator.py",
                "--dreams-per-language", str(self.dreams_per_run)
            ]
            
            # Add specific languages if requested
            if languages != ["english", "basque", "hebrew", "serbian", "slovenian"]:
                cmd.extend(["--language"] + languages)
            
            # Run dream generation
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)
            
            if result.returncode == 0:
                self.logger.info("Dream generation completed successfully")
                return True
            else:
                self.logger.error(f"Dream generation failed: {result.stderr}")
                return False
                
        except subprocess.TimeoutExpired:
            self.logger.error("Dream generation timed out (1 hour limit)")
            return False
        except Exception as e:
            self.logger.error(f"Dream generation error: {e}")
            return False
    
    def run_cultural_analysis(self) -> bool:
        """Run comprehensive cultural analysis"""
        try:
            self.logger.info("Starting cultural analysis...")
            
            result = subprocess.run(
                ["python", "cultural_dream_analyst_persona.py"],
                capture_output=True, text=True, timeout=1800
            )
            
            if result.returncode == 0:
                self.logger.info("Cultural analysis completed successfully")
                return True
            else:
                self.logger.error(f"Cultural analysis failed: {result.stderr}")
                return False
                
        except subprocess.TimeoutExpired:
            self.logger.error("Cultural analysis timed out (30 minute limit)")
            return False
        except Exception as e:
            self.logger.error(f"Cultural analysis error: {e}")
            return False
    
    def run_statistical_analysis(self) -> bool:
        """Run statistical analysis if available"""
        try:
            # Check if we have recent session data
            logs_dir = Path("logs")
            recent_sessions = []
            
            for lang_dir in logs_dir.iterdir():
                if lang_dir.is_dir():
                    gpt_dir = lang_dir / "gpt-4o"
                    if gpt_dir.exists():
                        sessions = [d for d in gpt_dir.iterdir() if d.is_dir()]
                        if sessions:
                            latest = max(sessions, key=lambda x: x.name)
                            recent_sessions.append(latest.name)
            
            if not recent_sessions:
                self.logger.warning("No recent sessions found for statistical analysis")
                return True
            
            # Use most recent session ID
            session_id = max(recent_sessions)
            
            self.logger.info(f"Running statistical analysis for session: {session_id}")
            
            result = subprocess.run(
                ["python", "statistical_analysis.py", "--session-id", session_id],
                capture_output=True, text=True, timeout=600
            )
            
            if result.returncode == 0:
                self.logger.info("Statistical analysis completed successfully")
                return True
            else:
                self.logger.warning(f"Statistical analysis had issues: {result.stderr}")
                return True  # Don't fail the whole pipeline for this
                
        except Exception as e:
            self.logger.warning(f"Statistical analysis error: {e}")
            return True  # Don't fail the whole pipeline
    
    def generate_research_report(self) -> bool:
        """Generate research report if available"""
        try:
            # Find recent sessions for report generation
            logs_dir = Path("logs")
            recent_sessions = []
            
            for lang_dir in logs_dir.iterdir():
                if lang_dir.is_dir():
                    gpt_dir = lang_dir / "gpt-4o"
                    if gpt_dir.exists():
                        sessions = [d for d in gpt_dir.iterdir() if d.is_dir()]
                        if sessions:
                            latest = max(sessions, key=lambda x: x.name)
                            recent_sessions.append(latest.name)
            
            if not recent_sessions:
                self.logger.warning("No sessions found for research report")
                return True
            
            session_id = max(recent_sessions)
            
            self.logger.info(f"Generating research report for session: {session_id}")
            
            result = subprocess.run([
                "python", "generate_research_report.py",
                "--sessions", session_id,
                "--title", f"Automated Dreams Analysis - {datetime.now().strftime('%Y-%m-%d')}",
                "--authors", "Dreams Automation System"
            ], capture_output=True, text=True, timeout=300)
            
            if result.returncode == 0:
                self.logger.info("Research report generated successfully")
                return True
            else:
                self.logger.warning(f"Research report generation had issues: {result.stderr}")
                return True  # Don't fail pipeline
                
        except Exception as e:
            self.logger.warning(f"Research report error: {e}")
            return True
    
    def git_commit_results(self) -> bool:
        """Commit results to git"""
        if not self.enable_git:
            return True
            
        try:
            self.logger.info("Committing results to git...")
            
            # Add all new files
            subprocess.run(["git", "add", "."], check=True)
            
            # Create commit message
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            commit_msg = f"Automated analysis run - {timestamp}\n\n- Generated {self.dreams_per_run} dreams per language\n- Ran cultural analysis\n- Updated datasets and reports"
            
            # Commit
            result = subprocess.run(
                ["git", "commit", "-m", commit_msg],
                capture_output=True, text=True
            )
            
            if result.returncode == 0:
                self.logger.info("Git commit successful")
                return True
            else:
                # Check if it's just "nothing to commit"
                if "nothing to commit" in result.stdout:
                    self.logger.info("No new changes to commit")
                    return True
                else:
                    self.logger.warning(f"Git commit issues: {result.stderr}")
                    return True  # Don't fail pipeline
                    
        except Exception as e:
            self.logger.warning(f"Git commit error: {e}")
            return True
    
    def run_full_pipeline(self, languages: list = None) -> bool:
        """Run the complete dreams research pipeline"""
        start_time = datetime.now()
        self.logger.info("="*80)
        self.logger.info("STARTING AUTOMATED DREAMS RESEARCH PIPELINE")
        self.logger.info("="*80)
        
        success = True
        
        # Step 1: Generate Dreams
        if not self.run_dream_generation(languages):
            self.logger.error("Pipeline failed at dream generation step")
            return False
        
        # Step 2: Cultural Analysis
        if not self.run_cultural_analysis():
            self.logger.error("Pipeline failed at cultural analysis step")
            success = False
        
        # Step 3: Statistical Analysis (optional)
        self.run_statistical_analysis()
        
        # Step 4: Research Report (optional)
        self.generate_research_report()
        
        # Step 5: Git Commit
        self.git_commit_results()
        
        # Summary
        duration = datetime.now() - start_time
        self.logger.info("="*80)
        if success:
            self.logger.info(f"AUTOMATED PIPELINE COMPLETED SUCCESSFULLY in {duration}")
        else:
            self.logger.warning(f"PIPELINE COMPLETED WITH ISSUES in {duration}")
        self.logger.info("="*80)
        
        return success
    
    def run_quick_analysis_only(self) -> bool:
        """Run only analysis on existing data (no new dream generation)"""
        self.logger.info("Running analysis-only pipeline...")
        
        success = True
        
        # Run cultural analysis
        if not self.run_cultural_analysis():
            success = False
        
        # Run statistical analysis
        self.run_statistical_analysis()
        
        # Commit results
        self.git_commit_results()
        
        return success

def create_scheduled_automation():
    """Create scheduled automation jobs"""
    
    automation = DreamsAutomation(dreams_per_run=25, enable_git=True)
    
    # Daily quick analysis (no new dreams, just re-analyze existing)
    schedule.every().day.at("09:00").do(automation.run_quick_analysis_only)
    
    # Weekly full pipeline (generate new dreams + full analysis)
    schedule.every().wednesday.at("02:00").do(
        automation.run_full_pipeline, languages=None
    )
    
    # Monthly comprehensive run
    schedule.every().month.do(
        automation.run_full_pipeline, languages=None
    )
    
    print("📅 Scheduled automation jobs:")
    print("  • Daily at 09:00: Quick analysis of existing data")
    print("  • Weekly (Wednesday 02:00): Full pipeline with new dreams")
    print("  • Monthly: Comprehensive analysis")
    print("\n🔄 Running scheduler... (Ctrl+C to stop)")
    
    try:
        while True:
            schedule.run_pending()
            time.sleep(60)  # Check every minute
    except KeyboardInterrupt:
        print("\n⏹️  Automation scheduler stopped")

def main():
    parser = argparse.ArgumentParser(description="Dreams Project Automation")
    parser.add_argument("--mode", choices=["once", "daily", "weekly", "continuous", "analysis-only"], 
                       default="once", help="Automation mode")
    parser.add_argument("--dreams", type=int, default=50, 
                       help="Dreams per language per run")
    parser.add_argument("--languages", nargs="+", 
                       choices=["english", "basque", "hebrew", "serbian", "slovenian"],
                       help="Specific languages to generate (default: all)")
    parser.add_argument("--interval", type=int, default=6, 
                       help="Hours between runs in continuous mode")
    parser.add_argument("--no-git", action="store_true", 
                       help="Disable git commits")
    parser.add_argument("--schedule", action="store_true",
                       help="Run with predefined schedule")
    
    args = parser.parse_args()
    
    # Handle scheduled automation
    if args.schedule:
        create_scheduled_automation()
        return
    
    # Create automation instance
    automation = DreamsAutomation(
        dreams_per_run=args.dreams,
        enable_git=not args.no_git
    )
    
    if args.mode == "once":
        # Run once and exit
        success = automation.run_full_pipeline(args.languages)
        sys.exit(0 if success else 1)
        
    elif args.mode == "analysis-only":
        # Run analysis only
        success = automation.run_quick_analysis_only()
        sys.exit(0 if success else 1)
        
    elif args.mode == "daily":
        # Schedule daily runs
        schedule.every().day.at("02:00").do(
            automation.run_full_pipeline, languages=args.languages
        )
        print(f"📅 Scheduled daily runs at 02:00 with {args.dreams} dreams per language")
        
    elif args.mode == "weekly":
        # Schedule weekly runs
        schedule.every().wednesday.at("02:00").do(
            automation.run_full_pipeline, languages=args.languages
        )
        print(f"📅 Scheduled weekly runs (Wednesday 02:00) with {args.dreams} dreams per language")
        
    elif args.mode == "continuous":
        # Schedule continuous runs
        schedule.every(args.interval).hours.do(
            automation.run_full_pipeline, languages=args.languages
        )
        print(f"📅 Scheduled continuous runs every {args.interval} hours with {args.dreams} dreams per language")
    
    # Run scheduler
    print("🔄 Running scheduler... (Ctrl+C to stop)")
    try:
        while True:
            schedule.run_pending()
            time.sleep(60)
    except KeyboardInterrupt:
        print("\n⏹️  Automation stopped")

if __name__ == "__main__":
    main()
