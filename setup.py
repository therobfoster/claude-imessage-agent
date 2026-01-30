#!/usr/bin/env python3
"""
Agent Setup Script
Interactive setup for the Claude iMessage Agent.
Run this once to configure the agent for your environment.
"""

import subprocess
import json
import sys
from pathlib import Path

AGENT_DIR = Path(__file__).parent
CONFIG_DIR = AGENT_DIR / "config"
MEMORY_DIR = AGENT_DIR / "memory"

def print_header(text):
    print(f"\n{'='*50}")
    print(f"  {text}")
    print('='*50)

def print_step(num, text):
    print(f"\n[{num}] {text}")

def ask(prompt, default=None):
    if default:
        result = input(f"{prompt} [{default}]: ").strip()
        return result if result else default
    return input(f"{prompt}: ").strip()

def ask_yes_no(prompt, default=True):
    default_str = "Y/n" if default else "y/N"
    result = input(f"{prompt} [{default_str}]: ").strip().lower()
    if not result:
        return default
    return result in ('y', 'yes')

def setup_agent_name():
    print_step(1, "Agent Name")

    print("\n  Give your agent a name. This will be used in logs and responses.")
    print("  Examples: Jarvis, Friday, Max, Assistant")

    # Get current directory name as default
    default_name = AGENT_DIR.name.replace("-", " ").replace("_", " ").title()
    if default_name.lower() == "claude imessage agent":
        default_name = "Claude"

    name = ask("\n  Agent name", default=default_name)
    return name


def check_claude_cli():
    print_step(2, "Checking Claude Code CLI...")

    result = subprocess.run(["which", "claude"], capture_output=True, text=True)
    if result.returncode == 0:
        print(f"  ✓ Claude CLI found: {result.stdout.strip()}")
        return True
    else:
        print("  ✗ Claude Code CLI not found!")
        print("")
        print("  The agent requires Claude Code CLI to generate responses.")
        print("  Install it with:")
        print("")
        print("    npm install -g @anthropic-ai/claude-code")
        print("")
        print("  Then run 'claude' once to authenticate.")
        print("")
        if not ask_yes_no("  Continue setup anyway?", default=False):
            sys.exit(1)
        return False


def install_dependencies():
    print_step(3, "Installing Python dependencies...")

    requirements = AGENT_DIR / "requirements.txt"
    if not requirements.exists():
        print("  Creating requirements.txt...")
        requirements.write_text("""# Agent Dependencies
tiktoken>=0.5.0              # Token counting
chromadb>=0.4.0              # Vector database
google-generativeai>=0.3.0   # Gemini embeddings
""")

    print("  Running pip install...")
    result = subprocess.run(
        [sys.executable, "-m", "pip", "install", "-r", str(requirements), "-q"],
        capture_output=True, text=True
    )

    if result.returncode == 0:
        print("  ✓ Dependencies installed successfully")
        return True
    else:
        print(f"  ✗ Failed to install dependencies:")
        print(f"    {result.stderr}")
        return False

def setup_api_keys():
    print_step(4, "API Keys Configuration")

    secrets_file = CONFIG_DIR / "secrets.json"
    existing = {}

    if secrets_file.exists():
        try:
            existing = json.loads(secrets_file.read_text())
            print("  Found existing secrets.json")
        except:
            pass

    print("\n  You need a Gemini API key for embeddings and image processing.")
    print("  Get one free at: https://makersuite.google.com/app/apikey")

    current_key = existing.get("gemini_api_key", "")
    masked_key = f"{current_key[:10]}...{current_key[-4:]}" if len(current_key) > 14 else "(not set)"

    if current_key:
        print(f"\n  Current key: {masked_key}")
        if not ask_yes_no("  Change API key?", default=False):
            return existing

    new_key = ask("\n  Enter your Gemini API key")
    if new_key:
        existing["gemini_api_key"] = new_key

    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    secrets_file.write_text(json.dumps(existing, indent=2))
    print("  ✓ API key saved to config/secrets.json")

    return existing

def setup_allowed_senders():
    print_step(5, "Allowed Senders Configuration")

    print("\n  The agent will ONLY respond to messages from these contacts.")
    print("  Use phone numbers (+1234567890) or iCloud emails.")

    config_file = CONFIG_DIR / "user_config.json"
    existing = {}

    if config_file.exists():
        try:
            existing = json.loads(config_file.read_text())
            current_senders = existing.get("allowed_senders", [])
            if current_senders:
                print(f"\n  Current allowed senders:")
                for s in current_senders:
                    print(f"    - {s}")
                if not ask_yes_no("\n  Modify allowed senders?", default=False):
                    return existing
        except:
            pass

    senders = []
    print("\n  Enter allowed senders (one per line, empty line when done):")
    while True:
        sender = input("    > ").strip()
        if not sender:
            break
        senders.append(sender)

    if senders:
        existing["allowed_senders"] = senders
    elif "allowed_senders" not in existing:
        existing["allowed_senders"] = []

    return existing

def setup_permissions(config):
    print_step(6, "Permissions Configuration")

    print("\n  Configure what the agent can do automatically vs. with approval.")

    permissions = config.get("permissions", {})

    print("\n  Auto-approve shell commands (ls, cat, etc.)?")
    print("  If NO, agent will ask before running ANY command.")
    permissions["auto_approve_commands"] = ask_yes_no("  Auto-approve safe commands?", default=True)

    print("\n  Auto-approve BUILD requests (multi-file projects)?")
    print("  If NO, agent will ask before building anything.")
    permissions["auto_approve_builds"] = ask_yes_no("  Auto-approve builds?", default=False)

    print("\n  Auto-approve MODIFY_SELF requests (agent updating its own code)?")
    print("  WARNING: This allows the agent to modify itself without asking.")
    permissions["auto_approve_self_modify"] = ask_yes_no("  Auto-approve self-modification?", default=False)

    print("\n  Auto-approve file writes outside workspace?")
    permissions["auto_approve_file_writes"] = ask_yes_no("  Auto-approve file writes?", default=False)

    # Safe commands list
    if "safe_commands" not in permissions:
        permissions["safe_commands"] = [
            "ls", "pwd", "date", "whoami", "echo",
            "cat", "head", "tail", "wc", "grep", "find"
        ]

    config["permissions"] = permissions
    return config

def setup_poll_interval(config):
    print_step(7, "Poll Interval")

    current = config.get("poll_interval", 10)
    print(f"\n  How often to check for new messages (in seconds).")
    print(f"  Lower = faster response but more CPU. Recommended: 5-15")

    interval = ask("  Poll interval", default=str(current))
    try:
        config["poll_interval"] = int(interval)
    except:
        config["poll_interval"] = 10

    return config

def save_config(config):
    config_file = CONFIG_DIR / "user_config.json"
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    config_file.write_text(json.dumps(config, indent=2))
    print(f"\n  ✓ Configuration saved to {config_file}")

def create_directories():
    print_step(8, "Creating directories...")

    dirs = [
        AGENT_DIR / "memory",
        AGENT_DIR / "logs",
        AGENT_DIR / "workspace",
        AGENT_DIR / "tasks",
        AGENT_DIR / "config"
    ]

    for d in dirs:
        d.mkdir(parents=True, exist_ok=True)
        print(f"  ✓ {d.name}/")

def print_next_steps():
    print_header("Setup Complete!")

    print("""
  To start the agent:
    python3 agent.py

  To run in debug mode (verbose logging):
    AGENT_DEBUG=1 python3 agent.py

  To check memory system status:
    python3 agent.py --stats

  Logs are saved to:
    logs/agent.log

  Configuration files:
    config/user_config.json  - Your settings
    config/secrets.json      - API keys (keep private!)
    config/FEATURES.md       - Agent capabilities

  NOTE: The agent requires Full Disk Access to read iMessages.
  Go to: System Preferences > Privacy & Security > Full Disk Access
  Add: Terminal (or your terminal app)
""")

def main():
    print_header("Claude iMessage Agent Setup")
    print("\n  This script will configure the agent for your environment.")
    print("  Press Ctrl+C at any time to cancel.\n")

    try:
        # Step 1: Agent name
        agent_name = setup_agent_name()

        # Step 2: Check Claude CLI
        check_claude_cli()

        # Step 3: Install dependencies
        if not install_dependencies():
            print("\n  ⚠️  Dependency installation failed. Please install manually:")
            print("     pip3 install tiktoken chromadb google-generativeai")

        # Step 4: API Keys
        setup_api_keys()

        # Step 5: Allowed senders
        config = setup_allowed_senders()

        # Save agent name to config
        config["agent_name"] = agent_name

        # Step 6: Permissions
        config = setup_permissions(config)

        # Step 7: Poll interval
        config = setup_poll_interval(config)

        # Save config
        save_config(config)

        # Step 6: Create directories
        create_directories()

        # Done
        print_next_steps()

    except KeyboardInterrupt:
        print("\n\n  Setup cancelled.")
        sys.exit(1)

if __name__ == "__main__":
    main()
