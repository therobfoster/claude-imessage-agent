#!/usr/bin/env python3
"""
Claude iMessage Agent
Monitors for new iMessages and responds using Claude Code CLI.
Supports memory, reminders, scheduled tasks, and building things with approval.
"""

import sqlite3
import subprocess
import json
import time
import os
import sys
import logging
import re
import shlex
import base64
from pathlib import Path
from datetime import datetime, timedelta

# Browser automation (Playwright)
try:
    from playwright.sync_api import sync_playwright, TimeoutError as PlaywrightTimeout
    PLAYWRIGHT_AVAILABLE = True
except ImportError:
    PLAYWRIGHT_AVAILABLE = False

# Image pipeline (Gemini)
try:
    from image_pipeline import (
        process_image_request,
        get_image_attachments_for_message,
        get_recent_image_attachments,
        analyze_image_with_gemini,
        edit_image_with_gemini,
        generate_image_with_gemini,
        send_image_via_imessage,
        load_gemini_config
    )
    IMAGE_PIPELINE_AVAILABLE = True
except ImportError:
    IMAGE_PIPELINE_AVAILABLE = False

# Token counting for context management
try:
    from token_counter import estimate_tokens, check_limit, format_token_status
    TOKEN_COUNTER_AVAILABLE = True
except ImportError:
    TOKEN_COUNTER_AVAILABLE = False

# Memory manager for semantic search
try:
    from memory_manager import get_relevant_context, get_manager, init_memory
    MEMORY_MANAGER_AVAILABLE = True
except ImportError:
    MEMORY_MANAGER_AVAILABLE = False

# Memory flush for pre-compaction save
try:
    from memory_flush import should_flush, run_memory_flush, mark_flushed
    MEMORY_FLUSH_AVAILABLE = True
except ImportError:
    MEMORY_FLUSH_AVAILABLE = False

# Auto-compaction for context management
try:
    from compaction import check_and_compact, should_compact
    COMPACTION_AVAILABLE = True
except ImportError:
    COMPACTION_AVAILABLE = False

# Configuration
MESSAGES_DB = Path.home() / "Library/Messages/chat.db"
AGENT_DIR = Path(__file__).parent.resolve()  # Directory where agent.py is located
MEMORY_DIR = AGENT_DIR / "memory"
TASKS_DIR = AGENT_DIR / "tasks"
LOGS_DIR = AGENT_DIR / "logs"
WORKSPACE_DIR = AGENT_DIR / "workspace"
STATE_FILE = AGENT_DIR / ".agent_state.json"
PENDING_FILE = AGENT_DIR / ".pending_actions.json"

# Load user configuration
USER_CONFIG_FILE = AGENT_DIR / "config" / "user_config.json"

def load_user_config():
    """Load user configuration from file, with defaults."""
    defaults = {
        "allowed_senders": [],
        "poll_interval": 10,
        "permissions": {
            "auto_approve_commands": False,
            "auto_approve_builds": False,
            "auto_approve_self_modify": False,
            "auto_approve_file_writes": False,
            "safe_commands": ["ls", "cat", "head", "tail", "grep", "find", "pwd", "whoami",
                              "date", "cal", "echo", "which", "file", "wc", "du", "df"]
        }
    }
    if USER_CONFIG_FILE.exists():
        try:
            with open(USER_CONFIG_FILE) as f:
                user_config = json.load(f)
                # Merge with defaults
                for key, value in user_config.items():
                    if key == "permissions" and isinstance(value, dict):
                        defaults["permissions"].update(value)
                    else:
                        defaults[key] = value
        except Exception as e:
            logger.warning(f"Failed to load user config: {e}")
    return defaults

# Load config at startup
_user_config = load_user_config()
ALLOWED_SENDERS = _user_config["allowed_senders"]
POLL_INTERVAL = _user_config["poll_interval"]
PERMISSIONS = _user_config["permissions"]
SAFE_COMMANDS = PERMISSIONS.get("safe_commands", [])

# Dangerous patterns that ALWAYS need approval
DANGEROUS_PATTERNS = [
    r"rm\s+-rf", r"rm\s+-r", r"sudo", r"chmod", r"chown",
    r"mv\s+/", r"cp\s+.*\s+/", r">\s*/", r"curl.*\|.*sh",
    r"wget.*\|.*sh", r"eval", r"exec"
]

# Set up logging with rotation
from logging.handlers import RotatingFileHandler
import os

LOG_LEVEL = logging.DEBUG if os.environ.get("AGENT_DEBUG") else logging.INFO

# Ensure logs directory exists before setting up handlers
LOGS_DIR.mkdir(parents=True, exist_ok=True)

# File handler with rotation (10MB max, keep 5 backups)
file_handler = RotatingFileHandler(
    LOGS_DIR / "agent.log",
    maxBytes=10*1024*1024,  # 10MB
    backupCount=5
)
file_handler.setLevel(LOG_LEVEL)
file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))

# Console handler
console_handler = logging.StreamHandler()
console_handler.setLevel(LOG_LEVEL)
console_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))

# Configure root logger
logging.basicConfig(level=LOG_LEVEL, handlers=[file_handler, console_handler])
logger = logging.getLogger(__name__)


# ============ Single Instance Lock ============

PID_FILE = AGENT_DIR / ".agent.pid"

def acquire_lock():
    """
    Acquire a lock to ensure only one agent instance runs.
    Returns True if lock acquired, False if another instance is running.
    """
    # Check if PID file exists
    if PID_FILE.exists():
        try:
            old_pid = int(PID_FILE.read_text().strip())
            # Check if that process is still running
            os.kill(old_pid, 0)  # Signal 0 = check if process exists
            # Process exists - another instance is running
            return False
        except (ProcessLookupError, ValueError):
            # Process doesn't exist or invalid PID - stale lock file
            pass
        except PermissionError:
            # Process exists but we can't signal it - assume it's running
            return False

    # Write our PID
    PID_FILE.write_text(str(os.getpid()))
    return True


def release_lock():
    """Release the lock on exit."""
    try:
        if PID_FILE.exists():
            current_pid = int(PID_FILE.read_text().strip())
            if current_pid == os.getpid():
                PID_FILE.unlink()
    except Exception:
        pass


# ============ Directory Setup ============

def ensure_directories():
    """Create necessary directories if they don't exist."""
    for dir_path in [AGENT_DIR, MEMORY_DIR, TASKS_DIR, LOGS_DIR, WORKSPACE_DIR]:
        dir_path.mkdir(parents=True, exist_ok=True)


# ============ State Management ============

def load_state():
    """Load the last processed message ROWID."""
    if STATE_FILE.exists():
        with open(STATE_FILE, 'r') as f:
            return json.load(f)
    return {"last_rowid": 0, "last_reminder_check": None}


def save_state(state):
    """Save the current state."""
    with open(STATE_FILE, 'w') as f:
        json.dump(state, f)


# ============ Pending Actions (Approval Queue) ============

def load_pending_actions():
    """Load pending actions awaiting approval."""
    if PENDING_FILE.exists():
        with open(PENDING_FILE, 'r') as f:
            return json.load(f)
    return {"actions": []}


def save_pending_actions(pending):
    """Save pending actions."""
    with open(PENDING_FILE, 'w') as f:
        json.dump(pending, f, indent=2)


def add_pending_action(action_type, content, description):
    """Add an action to the pending queue."""
    pending = load_pending_actions()
    action = {
        "id": len(pending["actions"]) + 1,
        "type": action_type,
        "content": content,
        "description": description,
        "created": datetime.now().isoformat(),
        "status": "pending"
    }
    pending["actions"].append(action)
    save_pending_actions(pending)
    return action


def get_pending_action(action_id=None):
    """Get a pending action by ID, or the most recent one."""
    pending = load_pending_actions()
    pending_list = [a for a in pending["actions"] if a["status"] == "pending"]
    if not pending_list:
        return None
    if action_id:
        for a in pending_list:
            if a["id"] == action_id:
                return a
        return None
    return pending_list[-1]  # Most recent


def mark_action_completed(action_id, result="completed"):
    """Mark an action as completed or rejected."""
    pending = load_pending_actions()
    for a in pending["actions"]:
        if a["id"] == action_id:
            a["status"] = result
            a["completed_at"] = datetime.now().isoformat()
            break
    save_pending_actions(pending)


# ============ Memory System ============

def load_knowledge():
    """Load the knowledge base."""
    knowledge_file = MEMORY_DIR / "knowledge.json"
    if knowledge_file.exists():
        with open(knowledge_file, 'r') as f:
            return json.load(f)
    return {"facts": [], "preferences": [], "projects": [], "people": []}


def save_knowledge(knowledge):
    """Save the knowledge base."""
    knowledge_file = MEMORY_DIR / "knowledge.json"
    with open(knowledge_file, 'w') as f:
        json.dump(knowledge, f, indent=2)


def load_reminders():
    """Load reminders."""
    reminders_file = MEMORY_DIR / "reminders.json"
    if reminders_file.exists():
        with open(reminders_file, 'r') as f:
            return json.load(f)
    return {"reminders": []}


def save_reminders(reminders):
    """Save reminders."""
    reminders_file = MEMORY_DIR / "reminders.json"
    with open(reminders_file, 'w') as f:
        json.dump(reminders, f, indent=2)


def add_reminder(text, due_date=None, recurring=None):
    """Add a new reminder."""
    reminders = load_reminders()
    reminder = {
        "id": len(reminders["reminders"]) + 1,
        "text": text,
        "created": datetime.now().isoformat(),
        "due": due_date,
        "recurring": recurring,
        "completed": False
    }
    reminders["reminders"].append(reminder)
    save_reminders(reminders)
    return reminder


def get_due_reminders():
    """Get reminders that are due now or overdue."""
    reminders = load_reminders()
    now = datetime.now()
    due = []

    for r in reminders["reminders"]:
        if r["completed"]:
            continue
        if r["due"]:
            try:
                due_date = datetime.fromisoformat(r["due"])
                if due_date <= now:
                    due.append(r)
            except:
                pass

    return due


def complete_reminder(reminder_id):
    """Mark a reminder as completed."""
    reminders = load_reminders()
    for r in reminders["reminders"]:
        if r["id"] == reminder_id:
            r["completed"] = True
            r["completed_at"] = datetime.now().isoformat()
            break
    save_reminders(reminders)


def add_fact(fact, category="facts"):
    """Add a fact to knowledge base."""
    knowledge = load_knowledge()
    entry = {
        "text": fact,
        "added": datetime.now().isoformat()
    }
    if category in knowledge:
        knowledge[category].append(entry)
    else:
        knowledge["facts"].append(entry)
    save_knowledge(knowledge)
    return entry


# ============ Message Database ============

def get_new_messages(last_rowid):
    """Query the Messages database for new messages (including those with images)."""
    conn = sqlite3.connect(str(MESSAGES_DB))
    cursor = conn.cursor()

    # Modified query to also get messages with attachments (even if no text)
    query = """
    SELECT
        m.ROWID,
        m.text,
        datetime(m.date/1000000000 + 978307200, 'unixepoch', 'localtime') as date,
        h.id as sender,
        m.cache_has_attachments
    FROM message m
    JOIN handle h ON m.handle_id = h.ROWID
    WHERE m.ROWID > ?
    AND m.is_from_me = 0
    AND (m.text IS NOT NULL AND m.text != '' OR m.cache_has_attachments = 1)
    ORDER BY m.ROWID
    """

    cursor.execute(query, (last_rowid,))
    messages = cursor.fetchall()
    conn.close()
    return messages


# ============ iMessage Sending ============

def send_imessage(recipient, message):
    """Send an iMessage using AppleScript."""
    # Escape special characters for AppleScript
    escaped_message = message.replace('\\', '\\\\').replace('"', '\\"').replace('\n', '\\n')

    script = f'''
    tell application "Messages"
        activate
        set targetService to 1st account whose service type = iMessage
        set targetBuddy to participant "{recipient}" of targetService
        send "{escaped_message}" to targetBuddy
    end tell
    '''

    try:
        subprocess.run(['osascript', '-e', script], check=True, capture_output=True)
        logger.info(f"Sent message to {recipient}")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to send message: {e}")
        return False


# ============ Command Safety Check ============

def is_safe_command(command):
    """Check if a command is safe to run without approval."""
    # Check for dangerous patterns first
    for pattern in DANGEROUS_PATTERNS:
        if re.search(pattern, command, re.IGNORECASE):
            return False

    # Check if it starts with a safe command
    first_word = command.strip().split()[0] if command.strip() else ""
    return first_word in SAFE_COMMANDS


def is_safe_file_path(path):
    """Check if a file path is safe (within workspace)."""
    try:
        resolved = Path(path).resolve()
        workspace_resolved = WORKSPACE_DIR.resolve()
        return str(resolved).startswith(str(workspace_resolved))
    except:
        return False


# ============ Action Execution ============

def execute_command(command, needs_approval=True):
    """Execute a shell command, potentially with approval."""
    # Check if auto-approve is enabled or command is safe
    auto_approve = PERMISSIONS.get("auto_approve_commands", False)
    if needs_approval and not auto_approve and not is_safe_command(command):
        action = add_pending_action("command", command, f"Run command: {command[:50]}...")
        return None, f"⚠️ This command needs your approval:\n\n`{command}`\n\nReply 'yes' or 'approve' to run it, or 'no' to cancel."

    try:
        result = subprocess.run(
            command,
            shell=True,
            capture_output=True,
            text=True,
            timeout=60,
            cwd=str(WORKSPACE_DIR)
        )
        output = result.stdout + result.stderr
        return output[:1000], None  # Limit output length
    except subprocess.TimeoutExpired:
        return None, "Command timed out after 60 seconds"
    except Exception as e:
        return None, f"Error: {str(e)}"


def execute_file_write(path, content, needs_approval=True):
    """Write a file, potentially with approval."""
    # Always work within workspace unless it's a safe path
    if not path.startswith("/"):
        full_path = WORKSPACE_DIR / path
    else:
        full_path = Path(path)

    auto_approve = PERMISSIONS.get("auto_approve_file_writes", False)
    if needs_approval and not auto_approve and not is_safe_file_path(full_path):
        action = add_pending_action("file_write", {"path": str(full_path), "content": content},
                                    f"Create file: {path}")
        return None, f"⚠️ Creating files outside workspace needs approval:\n\nFile: `{path}`\n\nReply 'yes' to create it, or 'no' to cancel."

    try:
        full_path.parent.mkdir(parents=True, exist_ok=True)
        full_path.write_text(content)
        return f"Created: {full_path}", None
    except Exception as e:
        return None, f"Error: {str(e)}"


def execute_build_task(task_description, sender=None):
    """Run Claude Code to build something complex."""
    # Check if auto-approve is enabled
    if PERMISSIONS.get("auto_approve_builds", False):
        return run_approved_build(task_description, sender)

    # Add to pending for approval
    action = add_pending_action("build", task_description, f"Build task: {task_description[:50]}...")
    return None, f"🔨 Build task queued:\n\n{task_description}\n\nReply 'yes' or 'approve' to start building, or 'no' to cancel."


def execute_self_modify(description, sender=None):
    """Queue a self-modification task."""
    # Check if auto-approve is enabled
    if PERMISSIONS.get("auto_approve_self_modify", False):
        return run_approved_self_modify(description, sender)

    action = add_pending_action("self_modify", description, f"Self-modify: {description[:50]}...")
    return None, f"🔧 Self-modification queued:\n\n{description}\n\n⚠️ This will modify my own code. Reply 'yes' to proceed, or 'no' to cancel."


def run_approved_self_modify(description, sender):
    """Actually run Claude Code to modify the agent's own code."""
    logger.info(f"Running approved self-modification: {description[:50]}...")

    # Read current agent code for context
    agent_code = ""
    agent_file = AGENT_DIR / "agent.py"
    if agent_file.exists():
        agent_code = agent_file.read_text()

    prompt = f"""You are modifying your own agent code. The agent is located at: {AGENT_DIR}

Current agent.py has {len(agent_code.splitlines())} lines.

IMPORTANT FILES:
- {AGENT_DIR}/agent.py - Main agent code
- {AGENT_DIR}/config/personality.md - Personality/system prompt
- {AGENT_DIR}/tasks/ - Scheduled task definitions
- {AGENT_DIR}/run_task.py - Task runner

MODIFICATION REQUEST:
{description}

Make the requested changes. Be careful to:
1. Not break existing functionality
2. Maintain the same code style
3. Add logging for new features
4. Test any new patterns/regexes

After making changes, summarize what you modified."""

    try:
        result = subprocess.run(
            ["claude", "-p", prompt, "--dangerously-skip-permissions"],
            capture_output=True,
            text=True,
            timeout=300,
            cwd=str(AGENT_DIR)
        )

        output = result.stdout.strip()
        if output:
            if len(output) > 500:
                output = output[:500] + "...\n\n(output truncated)"
            return f"🔧 Self-modification complete!\n\n{output}\n\n⚠️ Restart me to apply changes: reply 'restart agent'"
        else:
            return "🔧 Self-modification completed (no output). Restart me to apply changes."

    except subprocess.TimeoutExpired:
        return "⏰ Self-modification timed out after 5 minutes"
    except Exception as e:
        logger.error(f"Self-modify error: {e}")
        return f"❌ Self-modification error: {str(e)}"


def run_approved_build(task_description, sender):
    """Actually run Claude Code to build something."""
    logger.info(f"Running approved build task: {task_description[:50]}...")

    prompt = f"""You are helping build something. Work in the directory: {WORKSPACE_DIR}

Task: {task_description}

Create the necessary files and provide a summary of what you built."""

    try:
        # Run Claude Code with full capabilities
        result = subprocess.run(
            ["claude", "-p", prompt, "--dangerously-skip-permissions"],
            capture_output=True,
            text=True,
            timeout=300,  # 5 minute timeout for builds
            cwd=str(WORKSPACE_DIR)
        )

        output = result.stdout.strip()
        if output:
            # Truncate if too long for iMessage
            if len(output) > 500:
                output = output[:500] + "...\n\n(output truncated)"
            return f"✅ Build complete!\n\n{output}"
        else:
            return "✅ Build task completed (no output)"

    except subprocess.TimeoutExpired:
        return "⏰ Build task timed out after 5 minutes"
    except Exception as e:
        logger.error(f"Build error: {e}")
        return f"❌ Build error: {str(e)}"


# ============ Browser Automation ============

def execute_browser_action(action, target):
    """Execute a browser automation action using Playwright."""
    if not PLAYWRIGHT_AVAILABLE:
        return None, "❌ Playwright not installed. Run: pip install playwright && playwright install chromium"

    logger.info(f"Browser action: {action} - {target[:50] if target else 'N/A'}...")

    try:
        with sync_playwright() as p:
            # Launch browser in headless mode
            browser = p.chromium.launch(headless=True)
            context = browser.new_context(
                viewport={"width": 1280, "height": 720},
                user_agent="Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"
            )
            page = context.new_page()
            page.set_default_timeout(30000)  # 30 second timeout

            result = None

            if action == "open":
                # Open a URL
                url = target.strip()
                if not url.startswith(("http://", "https://")):
                    url = "https://" + url
                page.goto(url, wait_until="domcontentloaded")
                result = f"✅ Opened: {page.url}\nTitle: {page.title()}"
                logger.info(f"Browser opened: {page.url}")

            elif action == "click":
                # Click an element by selector or text
                # Target format: "url|selector" or just "selector" if page already loaded
                if "|" in target:
                    url, selector = target.split("|", 1)
                    url = url.strip()
                    if not url.startswith(("http://", "https://")):
                        url = "https://" + url
                    page.goto(url, wait_until="domcontentloaded")
                    selector = selector.strip()
                else:
                    return None, "❌ Click action requires format: url|selector"

                # Try different click strategies
                try:
                    page.click(selector, timeout=5000)
                except:
                    # Try clicking by text content
                    page.click(f"text={selector}", timeout=5000)
                result = f"✅ Clicked: {selector}"
                logger.info(f"Browser clicked: {selector}")

            elif action == "type":
                # Type text into an element
                # Target format: "url|selector|text"
                parts = target.split("|")
                if len(parts) < 3:
                    return None, "❌ Type action requires format: url|selector|text to type"
                url, selector, text = parts[0].strip(), parts[1].strip(), "|".join(parts[2:])
                if not url.startswith(("http://", "https://")):
                    url = "https://" + url
                page.goto(url, wait_until="domcontentloaded")
                page.fill(selector, text)
                result = f"✅ Typed into {selector}: {text[:50]}..."
                logger.info(f"Browser typed into: {selector}")

            elif action == "screenshot":
                # Take a screenshot of a URL
                url = target.strip()
                if not url.startswith(("http://", "https://")):
                    url = "https://" + url
                page.goto(url, wait_until="domcontentloaded")
                # Wait a moment for any dynamic content
                page.wait_for_timeout(1000)

                # Save screenshot to workspace
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                screenshot_path = WORKSPACE_DIR / f"screenshot_{timestamp}.png"
                page.screenshot(path=str(screenshot_path), full_page=False)
                result = f"📸 Screenshot saved: {screenshot_path.name}\nURL: {page.url}"
                logger.info(f"Browser screenshot saved: {screenshot_path}")

            elif action == "read":
                # Read page content (text extraction)
                url = target.strip()
                if not url.startswith(("http://", "https://")):
                    url = "https://" + url
                page.goto(url, wait_until="domcontentloaded")

                # Extract main text content
                title = page.title()
                # Get visible text, avoiding scripts and styles
                text_content = page.evaluate("""() => {
                    const body = document.body;
                    const walker = document.createTreeWalker(
                        body,
                        NodeFilter.SHOW_TEXT,
                        {
                            acceptNode: function(node) {
                                const parent = node.parentElement;
                                if (!parent) return NodeFilter.FILTER_REJECT;
                                const tag = parent.tagName.toLowerCase();
                                if (['script', 'style', 'noscript'].includes(tag)) {
                                    return NodeFilter.FILTER_REJECT;
                                }
                                const style = window.getComputedStyle(parent);
                                if (style.display === 'none' || style.visibility === 'hidden') {
                                    return NodeFilter.FILTER_REJECT;
                                }
                                return NodeFilter.FILTER_ACCEPT;
                            }
                        }
                    );
                    let text = '';
                    while (walker.nextNode()) {
                        const trimmed = walker.currentNode.textContent.trim();
                        if (trimmed) text += trimmed + '\\n';
                    }
                    return text;
                }""")

                # Truncate if too long
                if len(text_content) > 1500:
                    text_content = text_content[:1500] + "...\n\n(content truncated)"

                result = f"📄 Page: {title}\nURL: {page.url}\n\n{text_content}"
                logger.info(f"Browser read: {page.url}")

            else:
                return None, f"❌ Unknown browser action: {action}. Use: open, click, type, screenshot, read"

            browser.close()
            return result, None

    except PlaywrightTimeout as e:
        logger.error(f"Browser timeout: {e}")
        return None, f"⏰ Browser action timed out: {str(e)[:100]}"
    except Exception as e:
        logger.error(f"Browser error: {e}")
        return None, f"❌ Browser error: {str(e)[:200]}"


# ============ Web Search ============

def execute_web_search(query, context=None):
    """Perform a web search and return summarized results."""
    logger.info(f"Executing web search: {query}")

    try:
        import urllib.request
        import urllib.parse
        from html.parser import HTMLParser

        # Use DuckDuckGo HTML search (no API key required)
        encoded_query = urllib.parse.quote_plus(query)
        url = f"https://html.duckduckgo.com/html/?q={encoded_query}"

        headers = {
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36'
        }

        req = urllib.request.Request(url, headers=headers)

        with urllib.request.urlopen(req, timeout=30) as response:
            html_content = response.read().decode('utf-8')

        # Simple HTML parser to extract search results
        class DuckDuckGoParser(HTMLParser):
            def __init__(self):
                super().__init__()
                self.results = []
                self.current_result = {}
                self.in_title = False
                self.in_snippet = False

            def handle_starttag(self, tag, attrs):
                attrs_dict = dict(attrs)
                classes = attrs_dict.get('class', '')

                if tag == 'a' and 'result__a' in classes:
                    self.in_title = True
                    self.current_result = {'title': '', 'url': attrs_dict.get('href', ''), 'snippet': ''}
                elif tag == 'a' and 'result__snippet' in classes:
                    self.in_snippet = True

            def handle_endtag(self, tag):
                if tag == 'a' and self.in_title:
                    self.in_title = False
                elif tag == 'a' and self.in_snippet:
                    self.in_snippet = False
                    if self.current_result.get('title'):
                        self.results.append(self.current_result)
                        self.current_result = {}

            def handle_data(self, data):
                if self.in_title:
                    self.current_result['title'] += data.strip()
                elif self.in_snippet:
                    self.current_result['snippet'] += data.strip()

        parser = DuckDuckGoParser()
        parser.feed(html_content)

        results = parser.results[:5]  # Limit to top 5 results

        if not results:
            logger.warning(f"No search results found for: {query}")
            return f"🔍 No results found for: {query}", None

        # Format results
        formatted_results = [f"🔍 Search results for: {query}\n"]
        for i, result in enumerate(results, 1):
            title = result.get('title', 'No title')
            snippet = result.get('snippet', 'No description')
            formatted_results.append(f"{i}. **{title}**\n   {snippet[:150]}{'...' if len(snippet) > 150 else ''}")

        result_text = "\n\n".join(formatted_results)

        # If context was provided, prepend it
        if context and context.strip():
            result_text = f"[Context: {context.strip()}]\n\n{result_text}"

        logger.info(f"Web search completed: {len(results)} results found")
        return result_text, None

    except urllib.error.URLError as e:
        logger.error(f"Web search network error: {e}")
        return None, f"❌ Search failed (network error): {str(e)}"
    except Exception as e:
        logger.error(f"Web search error: {e}")
        return None, f"❌ Search failed: {str(e)}"


# ============ TikTok/Apify Integration ============

def load_apify_config():
    """Load Apify API configuration."""
    config_file = AGENT_DIR / "config" / "apify.json"
    if config_file.exists():
        with open(config_file, 'r') as f:
            return json.load(f)
    return None


def execute_tiktok_search(query, max_results=20):
    """Search TikTok for trending content using Apify."""
    logger.info(f"TikTok search: {query}")

    config = load_apify_config()
    if not config or config.get("api_token") == "YOUR_APIFY_API_TOKEN_HERE":
        return None, "❌ Apify not configured. Add your API token to ~/Agent/config/apify.json"

    api_token = config["api_token"]
    actor_id = config.get("tiktok_actor", "clockworks/free-tiktok-scraper")

    try:
        import urllib.request
        import urllib.parse

        # Apify API endpoint to run actor (URL-encode the actor ID)
        encoded_actor = urllib.parse.quote(actor_id, safe='')
        url = f"https://api.apify.com/v2/acts/{encoded_actor}/run-sync-get-dataset-items"

        # Input for the TikTok free scraper
        input_data = json.dumps({
            "searchQueries": [query],
            "resultsPerPage": max_results,
            "excludePinnedPosts": False,
            "shouldDownloadVideos": False,
            "shouldDownloadCovers": False,
            "shouldDownloadSlideshowImages": False,
            "shouldDownloadSubtitles": False
        }).encode('utf-8')

        headers = {
            'Content-Type': 'application/json',
            'Authorization': f'Bearer {api_token}'
        }

        req = urllib.request.Request(url, data=input_data, headers=headers, method='POST')

        # This can take a while - Apify runs the scraper
        with urllib.request.urlopen(req, timeout=120) as response:
            results = json.loads(response.read().decode('utf-8'))

        if not results:
            return f"🔍 No TikTok results for: {query}", None

        # Format results
        formatted = [f"📱 TikTok trends for: {query}\n"]

        for i, video in enumerate(results[:10], 1):
            author = video.get('authorMeta', {}).get('name', 'Unknown')
            desc = video.get('text', 'No description')[:100]
            plays = video.get('playCount', 0)
            likes = video.get('diggCount', 0)
            comments = video.get('commentCount', 0)

            # Format large numbers
            def fmt_num(n):
                if n >= 1_000_000:
                    return f"{n/1_000_000:.1f}M"
                elif n >= 1_000:
                    return f"{n/1_000:.1f}K"
                return str(n)

            formatted.append(
                f"{i}. @{author}\n"
                f"   {desc}{'...' if len(video.get('text', '')) > 100 else ''}\n"
                f"   👀 {fmt_num(plays)} | ❤️ {fmt_num(likes)} | 💬 {fmt_num(comments)}"
            )

        # Extract trending hashtags/products mentioned
        all_text = ' '.join([v.get('text', '') for v in results])
        hashtags = re.findall(r'#(\w+)', all_text)
        if hashtags:
            # Count hashtag frequency
            from collections import Counter
            top_tags = Counter(hashtags).most_common(10)
            tag_str = ', '.join([f"#{tag}({count})" for tag, count in top_tags])
            formatted.append(f"\n🏷️ Top hashtags: {tag_str}")

        logger.info(f"TikTok search completed: {len(results)} results")
        return '\n\n'.join(formatted), None

    except urllib.error.HTTPError as e:
        if e.code == 401:
            return None, "❌ Apify auth failed. Check your API token."
        elif e.code == 402:
            return None, "❌ Apify credits exhausted. Check your account."
        logger.error(f"TikTok search HTTP error: {e}")
        return None, f"❌ TikTok search failed: HTTP {e.code}"
    except Exception as e:
        logger.error(f"TikTok search error: {e}")
        return None, f"❌ TikTok search failed: {str(e)}"


def execute_tiktok_trends(region="US", count=20):
    """Get trending TikTok content (hashtags, videos, creators) using Apify."""
    logger.info(f"TikTok trends: region={region}, count={count}")

    config = load_apify_config()
    if not config or config.get("api_token") == "YOUR_APIFY_API_TOKEN_HERE":
        return None, "❌ Apify not configured. Add your API token to ~/Agent/config/apify.json"

    api_token = config["api_token"]
    actor_id = "clockworks/tiktok-trends-scraper"

    try:
        import urllib.request
        import urllib.parse

        encoded_actor = urllib.parse.quote(actor_id, safe='')
        url = f"https://api.apify.com/v2/acts/{encoded_actor}/run-sync-get-dataset-items"

        # Input for the trends scraper
        input_data = json.dumps({
            "region": region,
            "maxHashtags": count,
            "maxCreators": count,
            "maxVideos": count,
            "maxSongs": count
        }).encode('utf-8')

        headers = {
            'Content-Type': 'application/json',
            'Authorization': f'Bearer {api_token}'
        }

        req = urllib.request.Request(url, data=input_data, headers=headers, method='POST')

        # This can take a while
        with urllib.request.urlopen(req, timeout=180) as response:
            results = json.loads(response.read().decode('utf-8'))

        if not results:
            return f"🔍 No trends data available for region: {region}", None

        # Format results - the API returns hashtag trends with rank info
        formatted = [f"📈 TikTok Trending Hashtags ({region})\n"]

        for r in results[:count]:
            name = r.get('name', 'Unknown')
            rank = r.get('rank', '?')
            rank_diff = r.get('rankDiff', 0)
            is_new = r.get('markedAsNew', False)

            # Trend indicator
            if is_new:
                trend = "🆕"
            elif rank_diff > 0:
                trend = f"📈+{rank_diff}"
            elif rank_diff < 0:
                trend = f"📉{rank_diff}"
            else:
                trend = "➡️"

            formatted.append(f"{rank}. #{name} {trend}")

        logger.info(f"TikTok trends completed: {len(results)} items")
        return '\n'.join(formatted), None

    except urllib.error.HTTPError as e:
        if e.code == 401:
            return None, "❌ Apify auth failed. Check your API token."
        elif e.code == 402:
            return None, "❌ Apify credits exhausted. Check your account."
        logger.error(f"TikTok trends HTTP error: {e}")
        return None, f"❌ TikTok trends failed: HTTP {e.code}"
    except Exception as e:
        logger.error(f"TikTok trends error: {e}")
        return None, f"❌ TikTok trends failed: {str(e)}"


# ============ Context Building ============

def load_context():
    """Load the agent's full context including memory."""
    context_parts = []

    # Basic context file
    context_file = MEMORY_DIR / "context.md"
    if context_file.exists():
        context_parts.append("## Saved Context\n" + context_file.read_text())

    # Knowledge base
    knowledge = load_knowledge()
    if knowledge["facts"]:
        facts_text = "\n".join([f"- {f['text']}" for f in knowledge["facts"][-10:]])
        context_parts.append(f"## Things I Know\n{facts_text}")

    if knowledge["preferences"]:
        prefs_text = "\n".join([f"- {p['text']}" for p in knowledge["preferences"][-5:]])
        context_parts.append(f"## User Preferences\n{prefs_text}")

    # Active reminders
    reminders = load_reminders()
    active = [r for r in reminders["reminders"] if not r["completed"]]
    if active:
        rem_lines = []
        for r in active[-10:]:
            line = f"- [{r['id']}] {r['text']}"
            if r.get('due'):
                line += f" (due: {r['due']})"
            if r.get('recurring'):
                line += f" [repeats {r['recurring']}]"
            rem_lines.append(line)
        context_parts.append(f"## Active Reminders\n" + "\n".join(rem_lines))

    # Pending actions
    pending = load_pending_actions()
    pending_list = [a for a in pending["actions"] if a["status"] == "pending"]
    if pending_list:
        pending_text = "\n".join([f"- [{a['id']}] {a['type']}: {a['description']}" for a in pending_list])
        context_parts.append(f"## Pending Actions (awaiting approval)\n{pending_text}")

    # Workspace contents
    workspace_files = list(WORKSPACE_DIR.glob("*"))
    if workspace_files:
        files_text = "\n".join([f"- {f.name}" for f in workspace_files[:10]])
        context_parts.append(f"## Workspace Files\n{files_text}")

    return "\n\n".join(context_parts)


def save_conversation(sender, user_message, agent_response):
    """Save conversation to memory and index for semantic search."""
    log_file = MEMORY_DIR / "conversation_log.jsonl"

    entry = {
        "timestamp": datetime.now().isoformat(),
        "sender": sender,
        "user_message": user_message,
        "agent_response": agent_response
    }

    with open(log_file, 'a') as f:
        f.write(json.dumps(entry) + '\n')

    # Index into vector store for semantic search
    if MEMORY_MANAGER_AVAILABLE:
        try:
            manager = get_manager()
            manager.index_conversation(entry)
        except Exception as e:
            logger.warning(f"Failed to index conversation: {e}")


def get_recent_history(n=5):
    """Get the last n conversation exchanges."""
    log_file = MEMORY_DIR / "conversation_log.jsonl"
    if not log_file.exists():
        return "No previous conversations."

    lines = log_file.read_text().strip().split('\n')
    recent = lines[-n:] if len(lines) >= n else lines

    history = []
    for line in recent:
        try:
            entry = json.loads(line)
            history.append(f"User: {entry['user_message']}")
            history.append(f"Agent: {entry['agent_response']}")
        except:
            pass

    return '\n'.join(history) if history else "No previous conversations."


# ============ System Prompt ============

def get_system_prompt():
    """Get the agent's system prompt."""
    prompt_file = AGENT_DIR / "config" / "personality.md"
    if prompt_file.exists():
        return prompt_file.read_text()

    return """You are a helpful personal AI assistant communicating via iMessage.
Keep responses concise and conversational - this is text messaging, not email.
Be friendly, helpful, and proactive. If asked to remember something, acknowledge it.
If asked about your capabilities, explain you can help with questions, tasks, reminders, building things, and more."""


def get_tools_prompt(message=""):
    """Instructions for available actions. Lazy-loads based on message content."""
    msg_lower = message.lower()

    # Core tools (always included) - compressed
    tools = """## Actions (use XML tags in response)

### Core
- <REMEMBER>fact</REMEMBER> - Save to memory
- <REMINDER due="YYYY-MM-DD HH:MM" recurring="daily|weekly|monthly">text</REMINDER>
- <COMPLETE>id</COMPLETE> - Mark reminder done
- <RUN_COMMAND>cmd</RUN_COMMAND> - Run shell command
- <CREATE_FILE path="name.txt">content</CREATE_FILE>"""

    # Build tools - if mentions build, create, script, code, project, or self-modification
    build_keywords = ['build', 'create', 'script', 'code', 'project', 'make me', 'write a',
                      'modify', 'update yourself', 'change yourself', 'add feature',
                      'improve yourself', 'upgrade yourself', 'yourself to']
    if any(kw in msg_lower for kw in build_keywords):
        tools += """

### Building
- <BUILD>description</BUILD> - Multi-file projects (needs approval)
- <MODIFY_SELF file="agent.py">changes</MODIFY_SELF> - Update own code (needs approval)"""

        # Load features manifest for self-modification context
        features_file = AGENT_DIR / "config" / "FEATURES.md"
        if features_file.exists():
            features_content = features_file.read_text()
            tools += f"""

## Current Agent Features (consult before modifying)
{features_content}"""

    # Browser tools - if mentions website, url, browse, screenshot, page
    if any(kw in msg_lower for kw in ['website', 'url', 'browse', 'screenshot', 'page', 'http', 'www', '.com', '.org']):
        tools += """

### Browser
- <BROWSER action="read">url</BROWSER> - Get page text
- <BROWSER action="screenshot">url</BROWSER> - Take screenshot
- <BROWSER action="click">url|selector</BROWSER> - Click element
- <BROWSER action="type">url|selector|text</BROWSER> - Type in field"""

    # Search tools - if mentions search, look up, find, what is, weather, news
    if any(kw in msg_lower for kw in ['search', 'look up', 'find', 'what is', 'weather', 'news', 'latest', 'current']):
        tools += """

### Web Search
- <WEB_SEARCH query="terms">context</WEB_SEARCH> - Search DuckDuckGo"""

    # TikTok tools - if mentions tiktok, trending, viral, social
    if any(kw in msg_lower for kw in ['tiktok', 'trending', 'viral', 'social media', 'buzz']):
        tools += """

### TikTok
- <TIKTOK_SEARCH query="terms" max="20">context</TIKTOK_SEARCH> - Search videos
- <TIKTOK_TRENDS region="US" count="20"></TIKTOK_TRENDS> - Get trends"""

    # Image tools - if mentions image, photo, picture, edit, generate, analyze
    if any(kw in msg_lower for kw in ['image', 'photo', 'picture', 'pic', 'edit', 'generate', 'analyze', 'brighter', 'filter', 'blur']):
        tools += """

### Images (Gemini)
- <IMAGE_ANALYZE prompt="describe">which</IMAGE_ANALYZE> - Analyze image
- <IMAGE_EDIT prompt="make brighter">which</IMAGE_EDIT> - Edit image
- <IMAGE_GENERATE prompt="description" aspect="1:1"></IMAGE_GENERATE> - Create image"""

    return tools


# ============ Claude Invocation ============

def invoke_claude(user_message, sender):
    """Call Claude Code CLI with the user's message."""
    context = load_context()
    system_prompt = get_system_prompt()
    tools_prompt = get_tools_prompt(user_message)  # Lazy-load based on message

    # Get conversation history - semantic search if available, else recent history
    if MEMORY_MANAGER_AVAILABLE:
        relevant_context = get_relevant_context(user_message, top_k=3)  # Reduced for efficiency
        history_section = f"## Relevant Context (semantic search)\n{relevant_context}" if relevant_context else ""
        # Also include last 2 recent exchanges for immediate continuity
        recent = get_recent_history(2)
        if recent and recent != "No previous conversations.":
            history_section += f"\n\n## Recent Exchanges\n{recent}"
    else:
        history_section = f"## Conversation History (last few exchanges)\n{get_recent_history(5)}"

    full_prompt = f"""{system_prompt}

{tools_prompt}

## Current Context
{context}

{history_section}

## Current Time
{datetime.now().strftime("%Y-%m-%d %H:%M %A")}

## New Message from {sender}
{user_message}

Respond naturally as if texting. Keep it brief unless detail is needed. Use action tags when appropriate."""

    # Token counting for context monitoring
    if TOKEN_COUNTER_AVAILABLE:
        token_count = estimate_tokens(full_prompt)
        status = check_limit(token_count)
        logger.info(format_token_status(token_count))

        if status["critical"]:
            logger.warning("CRITICAL: Context window nearly full! Compaction needed.")
        elif status["warning"]:
            logger.warning(f"Context at {status['usage_pct']}% - approaching limit")

        # Memory flush before compaction if approaching limit
        if MEMORY_FLUSH_AVAILABLE and should_flush(token_count):
            logger.info("Triggering memory flush before compaction...")
            flush_result = run_memory_flush(history_section)
            if flush_result:
                mark_flushed(token_count)

        # Auto-compaction if at critical threshold
        if COMPACTION_AVAILABLE and status["critical"]:
            logger.info("Triggering auto-compaction...")
            compaction_result = check_and_compact()
            if compaction_result and compaction_result.success:
                logger.info(f"Compacted {compaction_result.messages_compacted} messages")

    try:
        result = subprocess.run(
            ["claude", "-p", full_prompt],
            capture_output=True,
            text=True,
            timeout=120
        )

        response = result.stdout.strip()

        if not response:
            response = "I received your message but had trouble generating a response. Please try again."

        return response

    except subprocess.TimeoutExpired:
        logger.error("Claude CLI timed out")
        return "Sorry, I'm taking too long to think. Please try again."
    except Exception as e:
        logger.error(f"Error invoking Claude: {e}")
        return "I encountered an error. Please try again."


# ============ Action Processing ============

def process_actions(response, sender):
    """Process action tags in the response and return cleaned response."""
    cleaned = response
    additional_messages = []

    # Process REMEMBER tags
    remember_pattern = r'<REMEMBER>(.*?)</REMEMBER>'
    for match in re.finditer(remember_pattern, response, re.DOTALL):
        fact = match.group(1).strip()
        add_fact(fact)
        logger.info(f"Remembered: {fact}")
    cleaned = re.sub(remember_pattern, '', cleaned, flags=re.DOTALL)

    # Process REMINDER tags
    reminder_pattern = r'<REMINDER(?:\s+due="([^"]*)")?(?:\s+recurring="([^"]*)")?\s*>(.*?)</REMINDER>'
    for match in re.finditer(reminder_pattern, response, re.DOTALL):
        due = match.group(1)
        recurring = match.group(2)
        text = match.group(3).strip()
        add_reminder(text, due, recurring)
        logger.info(f"Added reminder: {text} (due: {due}, recurring: {recurring})")
    cleaned = re.sub(reminder_pattern, '', cleaned, flags=re.DOTALL)

    # Process COMPLETE tags
    complete_pattern = r'<COMPLETE>(\d+)</COMPLETE>'
    for match in re.finditer(complete_pattern, response):
        reminder_id = int(match.group(1))
        complete_reminder(reminder_id)
        logger.info(f"Completed reminder: {reminder_id}")
    cleaned = re.sub(complete_pattern, '', cleaned)

    # Process CREATE_FILE tags
    file_pattern = r'<CREATE_FILE\s+path="([^"]+)">(.*?)</CREATE_FILE>'
    for match in re.finditer(file_pattern, response, re.DOTALL):
        path = match.group(1)
        content = match.group(2).strip()
        result, error = execute_file_write(path, content)
        if error:
            additional_messages.append(error)
        else:
            logger.info(f"Created file: {path}")
    cleaned = re.sub(file_pattern, '', cleaned, flags=re.DOTALL)

    # Process RUN_COMMAND tags
    cmd_pattern = r'<RUN_COMMAND>(.*?)</RUN_COMMAND>'
    for match in re.finditer(cmd_pattern, response, re.DOTALL):
        command = match.group(1).strip()
        result, error = execute_command(command)
        if error:
            additional_messages.append(error)
        elif result:
            additional_messages.append(f"Command output:\n```\n{result}\n```")
        logger.info(f"Ran command: {command}")
    cleaned = re.sub(cmd_pattern, '', cleaned, flags=re.DOTALL)

    # Process BUILD tags
    build_pattern = r'<BUILD>(.*?)</BUILD>'
    for match in re.finditer(build_pattern, response, re.DOTALL):
        task = match.group(1).strip()
        result, error = execute_build_task(task, sender)
        if error:
            additional_messages.append(error)
        elif result:
            additional_messages.append(result)
        logger.info(f"Build task: {task[:50]}...")
    cleaned = re.sub(build_pattern, '', cleaned, flags=re.DOTALL)

    # Process MODIFY_SELF tags
    modify_pattern = r'<MODIFY_SELF(?:\s+file="([^"]*)")?>(.*?)</MODIFY_SELF>'
    for match in re.finditer(modify_pattern, response, re.DOTALL):
        target_file = match.group(1) or "agent.py"
        description = match.group(2).strip()
        full_description = f"[Target: {target_file}] {description}"
        result, error = execute_self_modify(full_description, sender)
        if error:
            additional_messages.append(error)
        elif result:
            additional_messages.append(result)
        logger.info(f"Self-modification: {description[:50]}...")
    cleaned = re.sub(modify_pattern, '', cleaned, flags=re.DOTALL)

    # Process BROWSER tags
    browser_pattern = r'<BROWSER\s+action="([^"]+)">(.*?)</BROWSER>'
    for match in re.finditer(browser_pattern, response, re.DOTALL):
        action = match.group(1).strip().lower()
        target = match.group(2).strip()
        result, error = execute_browser_action(action, target)
        if error:
            additional_messages.append(error)
        elif result:
            additional_messages.append(result)
        logger.info(f"Browser action: {action} - {target[:50] if target else 'N/A'}")
    cleaned = re.sub(browser_pattern, '', cleaned, flags=re.DOTALL)

    # Process WEB_SEARCH tags
    search_pattern = r'<WEB_SEARCH\s+query="([^"]+)">(.*?)</WEB_SEARCH>'
    for match in re.finditer(search_pattern, response, re.DOTALL):
        query = match.group(1).strip()
        context = match.group(2).strip()
        result, error = execute_web_search(query, context if context else None)
        if error:
            additional_messages.append(error)
        elif result:
            additional_messages.append(result)
        logger.info(f"Web search: {query}")
    cleaned = re.sub(search_pattern, '', cleaned, flags=re.DOTALL)

    # Process TIKTOK_SEARCH tags
    tiktok_pattern = r'<TIKTOK_SEARCH\s+query="([^"]+)"(?:\s+max="(\d+)")?>(.*?)</TIKTOK_SEARCH>'
    for match in re.finditer(tiktok_pattern, response, re.DOTALL):
        query = match.group(1).strip()
        max_results = int(match.group(2)) if match.group(2) else 20
        context = match.group(3).strip()
        result, error = execute_tiktok_search(query, max_results)
        if error:
            additional_messages.append(error)
        elif result:
            additional_messages.append(result)
        logger.info(f"TikTok search: {query} (max: {max_results})")
    cleaned = re.sub(tiktok_pattern, '', cleaned, flags=re.DOTALL)

    # Process TIKTOK_TRENDS tags
    trends_pattern = r'<TIKTOK_TRENDS(?:\s+region="([^"]*)")?(?:\s+count="(\d+)")?>(.*?)</TIKTOK_TRENDS>'
    for match in re.finditer(trends_pattern, response, re.DOTALL):
        region = match.group(1).strip() if match.group(1) else "US"
        count = int(match.group(2)) if match.group(2) else 20
        result, error = execute_tiktok_trends(region, count)
        if error:
            additional_messages.append(error)
        elif result:
            additional_messages.append(result)
        logger.info(f"TikTok trends: region={region}, count={count}")
    cleaned = re.sub(trends_pattern, '', cleaned, flags=re.DOTALL)

    # Process IMAGE_ANALYZE tags
    if IMAGE_PIPELINE_AVAILABLE:
        analyze_pattern = r'<IMAGE_ANALYZE\s+prompt="([^"]+)">(.*?)</IMAGE_ANALYZE>'
        for match in re.finditer(analyze_pattern, response, re.DOTALL):
            prompt = match.group(1).strip()
            which_image = match.group(2).strip() if match.group(2) else None
            # Get the most recent image from sender
            recent = get_recent_image_attachments(sender, limit=1)
            if recent:
                result, error = analyze_image_with_gemini(recent[0]['path'], prompt)
                if error:
                    additional_messages.append(error)
                elif result:
                    additional_messages.append(f"📷 {result}")
            else:
                additional_messages.append("📷 No recent image found. Please send me an image first!")
            logger.info(f"Image analyze: {prompt[:50]}")
        cleaned = re.sub(analyze_pattern, '', cleaned, flags=re.DOTALL)

        # Process IMAGE_EDIT tags
        edit_pattern = r'<IMAGE_EDIT\s+prompt="([^"]+)">(.*?)</IMAGE_EDIT>'
        for match in re.finditer(edit_pattern, response, re.DOTALL):
            prompt = match.group(1).strip()
            which_image = match.group(2).strip() if match.group(2) else None
            recent = get_recent_image_attachments(sender, limit=1)
            if recent:
                output_path, error = edit_image_with_gemini(recent[0]['path'], prompt)
                if error:
                    additional_messages.append(error)
                elif output_path:
                    # Send the edited image back
                    if send_image_via_imessage(sender, output_path, "✨ Here's your edited image!"):
                        additional_messages.append(f"✨ Edited image sent!")
                    else:
                        additional_messages.append(f"✨ Edited image saved to: {output_path}")
            else:
                additional_messages.append("📷 No recent image found. Please send me an image first!")
            logger.info(f"Image edit: {prompt[:50]}")
        cleaned = re.sub(edit_pattern, '', cleaned, flags=re.DOTALL)

        # Process IMAGE_GENERATE tags
        generate_pattern = r'<IMAGE_GENERATE\s+prompt="([^"]+)"(?:\s+aspect="([^"]*)")?\s*>(.*?)</IMAGE_GENERATE>'
        for match in re.finditer(generate_pattern, response, re.DOTALL):
            prompt = match.group(1).strip()
            aspect = match.group(2).strip() if match.group(2) else "1:1"
            output_path, error = generate_image_with_gemini(prompt, aspect)
            if error:
                additional_messages.append(error)
            elif output_path:
                # Send the generated image back
                if send_image_via_imessage(sender, output_path, "🎨 Here's your generated image!"):
                    additional_messages.append(f"🎨 Generated image sent!")
                else:
                    additional_messages.append(f"🎨 Generated image saved to: {output_path}")
            logger.info(f"Image generate: {prompt[:50]}")
        cleaned = re.sub(generate_pattern, '', cleaned, flags=re.DOTALL)

    # Clean up extra whitespace
    cleaned = re.sub(r'\n{3,}', '\n\n', cleaned).strip()

    # Append additional messages
    if additional_messages:
        cleaned = cleaned + "\n\n" + "\n\n".join(additional_messages)

    return cleaned


# ============ Approval Handling ============

def check_for_approval(text):
    """Check if a message is approving or rejecting a pending action."""
    text_lower = text.lower().strip()

    # Check for approval
    approval_words = ["yes", "approve", "approved", "ok", "okay", "go ahead", "do it", "proceed", "confirm", "y"]
    if any(text_lower == word or text_lower.startswith(word + " ") for word in approval_words):
        return "approved"

    # Check for rejection
    rejection_words = ["no", "cancel", "reject", "stop", "don't", "abort", "n"]
    if any(text_lower == word or text_lower.startswith(word + " ") for word in rejection_words):
        return "rejected"

    return None


def check_for_special_commands(text):
    """Check for special agent control commands."""
    text_lower = text.lower().strip()

    if text_lower in ["restart agent", "restart yourself", "restart"]:
        return "restart"
    if text_lower in ["status", "agent status"]:
        return "status"

    return None


def handle_special_command(command, sender):
    """Handle special agent control commands."""
    if command == "restart":
        send_imessage(sender, "🔄 Restarting... I'll be back in a moment!")
        logger.info("Restart requested via iMessage")
        # Touch a file to trigger fswatch restart, or just exit
        import sys
        sys.exit(0)  # Exit cleanly, LaunchAgent or watch.sh will restart us

    elif command == "status":
        pending = load_pending_actions()
        pending_count = len([a for a in pending["actions"] if a["status"] == "pending"])
        reminders = load_reminders()
        active_reminders = len([r for r in reminders["reminders"] if not r["completed"]])
        knowledge = load_knowledge()
        facts_count = len(knowledge.get("facts", []))

        workspace_files = list(WORKSPACE_DIR.glob("*"))

        status = f"""📊 Agent Status:
• Active reminders: {active_reminders}
• Pending approvals: {pending_count}
• Facts remembered: {facts_count}
• Workspace files: {len(workspace_files)}
• Uptime: Running"""

        return status

    return None


def handle_approval(approval_type, sender):
    """Handle approval or rejection of pending action."""
    pending_action = get_pending_action()

    if not pending_action:
        return None  # No pending action, treat as normal message

    if approval_type == "approved":
        mark_action_completed(pending_action["id"], "approved")

        if pending_action["type"] == "command":
            # Execute the approved command
            result, error = execute_command(pending_action["content"], needs_approval=False)
            if error:
                return f"❌ {error}"
            return f"✅ Command executed!\n\n```\n{result}\n```" if result else "✅ Command executed (no output)"

        elif pending_action["type"] == "file_write":
            # Write the approved file
            content = pending_action["content"]
            result, error = execute_file_write(content["path"], content["content"], needs_approval=False)
            if error:
                return f"❌ {error}"
            return f"✅ {result}"

        elif pending_action["type"] == "build":
            # Run the approved build
            return run_approved_build(pending_action["content"], sender)

        elif pending_action["type"] == "self_modify":
            # Run the approved self-modification
            return run_approved_self_modify(pending_action["content"], sender)

    elif approval_type == "rejected":
        mark_action_completed(pending_action["id"], "rejected")
        return "🚫 Action cancelled."

    return None


# ============ Reminder Checking ============

def calculate_next_occurrence(current_due, recurring):
    """Calculate the next occurrence for a recurring reminder."""
    try:
        due_dt = datetime.fromisoformat(current_due)
    except:
        due_dt = datetime.now()

    if recurring == "daily":
        next_due = due_dt + timedelta(days=1)
    elif recurring == "weekly":
        next_due = due_dt + timedelta(weeks=1)
    elif recurring == "monthly":
        next_due = due_dt + timedelta(days=30)
    elif recurring == "weekdays":
        next_due = due_dt + timedelta(days=1)
        while next_due.weekday() >= 5:
            next_due += timedelta(days=1)
    elif recurring == "weekends":
        next_due = due_dt + timedelta(days=1)
        while next_due.weekday() < 5:
            next_due += timedelta(days=1)
    else:
        return None

    return next_due.strftime("%Y-%m-%d %H:%M")


def reschedule_reminder(reminder_id, new_due):
    """Reschedule a recurring reminder."""
    reminders = load_reminders()
    for r in reminders["reminders"]:
        if r["id"] == reminder_id:
            r["due"] = new_due
            r["last_triggered"] = datetime.now().isoformat()
            break
    save_reminders(reminders)


def check_and_send_reminders(recipient):
    """Check for due reminders and send notifications."""
    due_reminders = get_due_reminders()

    if due_reminders:
        for r in due_reminders:
            message = f"⏰ Reminder: {r['text']}"
            if r.get("recurring"):
                message += f" (repeats {r['recurring']})"
            send_imessage(recipient, message)
            logger.info(f"Sent reminder {r['id']}: {r['text']}")

            if r.get("recurring"):
                next_due = calculate_next_occurrence(r["due"], r["recurring"])
                if next_due:
                    reschedule_reminder(r["id"], next_due)
                    logger.info(f"Rescheduled reminder {r['id']} to {next_due}")
            else:
                complete_reminder(r["id"])
                logger.info(f"Completed one-time reminder {r['id']}")


# ============ Message Processing ============

def process_message(rowid, text, date, sender, has_attachments=False):
    """Process a single message and respond."""
    display_text = text[:50] if text else "(image only)"
    logger.info(f"Processing message from {sender}: {display_text}...")

    if sender not in ALLOWED_SENDERS:
        logger.info(f"Ignoring message from unknown sender: {sender}")
        return

    # Handle image-only messages
    if not text and has_attachments and IMAGE_PIPELINE_AVAILABLE:
        text = "describe this image"
        logger.info("Image-only message received, defaulting to 'describe this image'")

    if not text:
        logger.info("Empty message with no actionable content, skipping")
        return

    # Check for special commands first
    special_cmd = check_for_special_commands(text)
    if special_cmd:
        response = handle_special_command(special_cmd, sender)
        if response:
            send_imessage(sender, response)
            save_conversation(sender, text, response)
        return

    # Check if this is an approval/rejection of a pending action
    approval_type = check_for_approval(text)
    if approval_type:
        approval_response = handle_approval(approval_type, sender)
        if approval_response:
            send_imessage(sender, approval_response)
            save_conversation(sender, text, approval_response)
            logger.info(f"Handled approval: {approval_type}")
            return

    # Check for direct image processing requests (with or without attachment)
    if IMAGE_PIPELINE_AVAILABLE and has_attachments:
        image_keywords = ['image', 'photo', 'picture', 'pic', 'this', 'describe', 'what', 'analyze', 'edit', 'modify', 'make', 'change']
        if any(kw in text.lower() for kw in image_keywords):
            response_text, image_path = process_image_request(sender, text, rowid)
            if image_path:
                # Send the image first, then the text
                send_image_via_imessage(sender, image_path)
            send_imessage(sender, response_text)
            save_conversation(sender, text, response_text)
            logger.info(f"Processed image request from {sender}")
            return

    # Normal message processing
    send_imessage(sender, "🔄")  # Quick "working on it" indicator
    response = invoke_claude(text, sender)
    clean_response = process_actions(response, sender)
    send_imessage(sender, clean_response)
    save_conversation(sender, text, clean_response)
    logger.info(f"Responded to {sender}")


# ============ Main Loop ============

def run_agent():
    """Main agent loop."""
    ensure_directories()

    # Ensure only one instance runs
    if not acquire_lock():
        logger.error("Another agent instance is already running! Exiting.")
        print("ERROR: Another agent instance is already running!")
        print(f"Check PID file: {PID_FILE}")
        sys.exit(1)

    # Clean up lock on exit
    import atexit
    atexit.register(release_lock)

    state = load_state()

    agent_name = _user_config.get("agent_name", "Claude")
    logger.info("=" * 50)
    logger.info(f"{agent_name} iMessage Agent starting...")
    logger.info(f"Monitoring for messages from: {ALLOWED_SENDERS}")
    logger.info(f"Poll interval: {POLL_INTERVAL} seconds")
    logger.info(f"Workspace: {WORKSPACE_DIR}")
    logger.info("=" * 50)

    last_reminder_check = None

    while True:
        try:
            messages = get_new_messages(state["last_rowid"])

            for msg in messages:
                rowid, text, date, sender = msg[0], msg[1], msg[2], msg[3]
                has_attachments = msg[4] if len(msg) > 4 else False
                process_message(rowid, text, date, sender, has_attachments)
                state["last_rowid"] = rowid
                save_state(state)

            now = datetime.now()
            if last_reminder_check is None or (now - last_reminder_check).seconds >= 60:
                for sender in ALLOWED_SENDERS:
                    check_and_send_reminders(sender)
                last_reminder_check = now

            time.sleep(POLL_INTERVAL)

        except KeyboardInterrupt:
            logger.info("Agent stopped by user")
            break
        except Exception as e:
            logger.error(f"Error in main loop: {e}")
            time.sleep(POLL_INTERVAL)


def print_memory_stats():
    """Print memory system statistics."""
    print("\n=== Memory System Status ===\n")

    # Token counter
    if TOKEN_COUNTER_AVAILABLE:
        print("Token Counter: Available")
    else:
        print("Token Counter: Not available")

    # Memory manager
    if MEMORY_MANAGER_AVAILABLE:
        try:
            manager = get_manager()
            stats = manager.get_stats()
            print(f"Memory Manager: Available")
            print(f"  - Documents indexed: {stats['total_documents']}")
            print(f"  - Collection: {stats['collection_name']}")
            print(f"  - Storage: {stats['chroma_path']}")
        except Exception as e:
            print(f"Memory Manager: Error - {e}")
    else:
        print("Memory Manager: Not available")

    # Memory flush
    if MEMORY_FLUSH_AVAILABLE:
        from memory_flush import get_flush_stats
        stats = get_flush_stats()
        print(f"Memory Flush: Available")
        print(f"  - Threshold: {stats['flush_threshold']:,} tokens")
        print(f"  - MEMORY.md exists: {stats['memory_exists']}")
    else:
        print("Memory Flush: Not available")

    # Compaction
    if COMPACTION_AVAILABLE:
        from compaction import load_conversations, HARD_LIMIT
        messages = load_conversations()
        print(f"Compaction: Available")
        print(f"  - Threshold: {HARD_LIMIT:,} tokens")
        print(f"  - Messages in log: {len(messages)}")
    else:
        print("Compaction: Not available")

    print()


def reindex_memory():
    """Rebuild the vector store from scratch."""
    if not MEMORY_MANAGER_AVAILABLE:
        print("Memory manager not available")
        return

    print("Rebuilding vector store...")

    # Clear hash cache to force re-indexing
    hash_cache_file = MEMORY_DIR / ".content_hashes.json"
    if hash_cache_file.exists():
        hash_cache_file.unlink()
        print("Cleared hash cache")

    # Clear chroma directory
    chroma_dir = MEMORY_DIR / "chroma"
    if chroma_dir.exists():
        import shutil
        shutil.rmtree(chroma_dir)
        print("Cleared vector store")

    # Re-initialize with backfill
    manager = init_memory(backfill=True)
    stats = manager.get_stats()
    print(f"Re-indexed {stats['total_documents']} documents")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        if sys.argv[1] == "--stats":
            print_memory_stats()
        elif sys.argv[1] == "--reindex":
            reindex_memory()
        elif sys.argv[1] == "--help":
            print("Usage: python agent.py [OPTIONS]")
            print("")
            print("Options:")
            print("  --stats     Show memory system statistics")
            print("  --reindex   Rebuild vector store from scratch")
            print("  --help      Show this help message")
            print("")
            print("Without options, runs the agent main loop.")
        else:
            print(f"Unknown option: {sys.argv[1]}")
            print("Use --help for usage information")
    else:
        run_agent()
