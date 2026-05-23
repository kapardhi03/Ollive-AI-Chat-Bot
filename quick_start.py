#!/usr/bin/env python3
"""
Quick Start Demo - Shows Streamlit Apps Working
This script demonstrates the web interfaces without requiring model downloads
"""

import subprocess
import sys
import os
import time
import webbrowser
from threading import Thread

def check_streamlit():
    """Check if streamlit is available"""
    try:
        import streamlit
        return True
    except ImportError:
        print("❌ Streamlit not found. Install with: pip install streamlit")
        return False

def start_oss_app():
    """Start the OSS assistant app"""
    print("🤖 Starting OSS Assistant...")
    try:
        os.chdir("oss_assistant")
        subprocess.run([sys.executable, "-m", "streamlit", "run", "app.py",
                       "--server.port", "8501", "--server.headless", "true"])
    except Exception as e:
        print(f"Failed to start OSS app: {e}")
    finally:
        os.chdir("..")

def start_frontier_app():
    """Start the frontier assistant app"""
    print("🚀 Starting Frontier Assistant...")
    try:
        os.chdir("frontier_assistant")
        subprocess.run([sys.executable, "-m", "streamlit", "run", "app.py",
                       "--server.port", "8502", "--server.headless", "true"])
    except Exception as e:
        print(f"Failed to start Frontier app: {e}")
    finally:
        os.chdir("..")

def show_instructions():
    """Show manual start instructions"""
    print("🌐 Manual Start Instructions")
    print("=" * 40)
    print()
    print("To start the web interfaces manually:")
    print()
    print("1. OSS Assistant:")
    print("   cd oss_assistant")
    print("   streamlit run app.py --server.port 8501")
    print("   Then open: http://localhost:8501")
    print()
    print("2. Frontier Assistant:")
    print("   cd frontier_assistant")
    print("   streamlit run app.py --server.port 8502")
    print("   Then open: http://localhost:8502")
    print()
    print("Note: You'll need to add API keys for frontier models to work")
    print("The OSS assistant will attempt to download models on first use")

def main():
    """Main quick start function"""
    print("🎯 AI Assistant Comparison - Quick Start Demo")
    print("=" * 50)
    print()

    if not check_streamlit():
        return False

    print("This demo shows how to start the web interfaces.")
    print("The apps will handle model loading and API configuration automatically.")
    print()

    choice = input("Start web interfaces? (y/n): ").lower().strip()

    if choice == 'y':
        print("\n🚀 Starting both assistants...")
        print("This may take a moment for the first time...")
        print()

        # Start both apps in separate threads
        oss_thread = Thread(target=start_oss_app)
        frontier_thread = Thread(target=start_frontier_app)

        print("Starting OSS Assistant on http://localhost:8501...")
        oss_thread.start()

        time.sleep(2)  # Give first app time to start

        print("Starting Frontier Assistant on http://localhost:8502...")
        frontier_thread.start()

        print()
        print("✅ Both assistants should be starting up!")
        print("🌐 OSS Assistant: http://localhost:8501")
        print("🌐 Frontier Assistant: http://localhost:8502")
        print()
        print("Press Ctrl+C to stop both applications")

        try:
            # Wait for threads to complete
            oss_thread.join()
            frontier_thread.join()
        except KeyboardInterrupt:
            print("\n👋 Stopping applications...")
            return True

    else:
        show_instructions()

    return True

if __name__ == "__main__":
    try:
        success = main()
        exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")
        exit(0)