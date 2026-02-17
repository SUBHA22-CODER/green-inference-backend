#!/usr/bin/env python3
"""
Green-Inference: Main Entry Point
Choose to run dashboard, API server, or CLI mode
"""

import argparse
import sys
import os

# Add core to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'core'))


def run_dashboard():
    """Launch the CustomTkinter dashboard"""
    print("Launching Green-Inference Dashboard...")
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'ui'))
    
    import dashboard
    import yaml
    
    config_path = os.path.join(os.path.dirname(__file__), 'config', 'config.yaml')
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    app = dashboard.GreenInferenceDashboard(config)
    app.protocol("WM_DELETE_WINDOW", app.on_closing)
    app.mainloop()


def run_api():
    """Launch the FastAPI server"""
    print("Launching Green-Inference API Server...")
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'api'))
    
    import yaml
    import uvicorn
    from main import app
    
    config_path = os.path.join(os.path.dirname(__file__), 'config', 'config.yaml')
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    api_config = config.get('API', {})
    host = api_config.get('HOST', '127.0.0.1')
    port = api_config.get('PORT', 8000)
    
    print(f"Starting API server on {host}:{port}")
    print(f"API docs available at http://{host}:{port}/docs")
    
    uvicorn.run(app, host=host, port=port, log_level="info")


def run_cli():
    """Run in CLI mode"""
    print("Launching Green-Inference CLI Mode...")
    
    import yaml
    import time
    from loguru import logger
    from orchestrator import GreenInferenceOrchestrator
    
    config_path = os.path.join(os.path.dirname(__file__), 'config', 'config.yaml')
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    orchestrator = GreenInferenceOrchestrator(config)
    
    def log_event(event):
        print(f"[{event.event_type}] {event.details}")
    
    orchestrator.register_event_callback(log_event)
    orchestrator.start()
    
    print("\n" + "="*60)
    print("GREEN-INFERENCE CLI MODE")
    print("="*60)
    print("Commands:")
    print("  status - Show current status")
    print("  generate <prompt> - Generate AI response")
    print("  quit - Exit")
    print("="*60 + "\n")
    
    try:
        while True:
            cmd = input("\n> ").strip()
            
            if not cmd:
                continue
            
            if cmd == "quit" or cmd == "exit":
                break
            
            elif cmd == "status":
                print("\n" + orchestrator.get_detailed_status())
            
            elif cmd.startswith("generate "):
                prompt = cmd[9:].strip()
                if prompt:
                    print("\nGenerating response...")
                    response = orchestrator.generate(prompt)
                    print(f"\nResponse:\n{response}\n")
                else:
                    print("Please provide a prompt")
            
            else:
                print(f"Unknown command: {cmd}")
    
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
    
    finally:
        print("\nStopping orchestrator...")
        orchestrator.stop()
        print("Goodbye!")


def main():
    """Main entry point with argument parsing"""
    parser = argparse.ArgumentParser(
        description="Green-Inference: AI Power Orchestrator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py --dashboard    # Launch GUI dashboard
  python main.py --api          # Start API server
  python main.py --cli          # Run in CLI mode
  python main.py --help         # Show this help
        """
    )
    
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--dashboard', action='store_true', help='Launch GUI dashboard')
    group.add_argument('--api', action='store_true', help='Start API server')
    group.add_argument('--cli', action='store_true', help='Run in CLI mode')
    
    args = parser.parse_args()
    
    if args.dashboard:
        run_dashboard()
    elif args.api:
        run_api()
    elif args.cli:
        run_cli()


if __name__ == "__main__":
    main()
