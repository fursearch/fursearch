"""AI tool command handler for running Claude CLI via Telegram."""

import asyncio
import os
import shlex
import subprocess

from telegram import Update
from telegram.ext import ContextTypes


# Configuration (from environment variables with baked-in defaults)
WORK_DIR = os.environ.get(
    "AITOOL_WORK_DIR",
    os.path.abspath(os.path.join(os.path.dirname(__file__), "."))
)
BINARY = os.environ.get("AITOOL_BINARY", "claude")
ARGS = os.environ.get(
    "AITOOL_ARGS",
    "--allowedTools 'Bash(git:*) Edit Write Read Glob Grep' -p"
)
TIMEOUT = int(os.environ.get("AITOOL_TIMEOUT", "600"))  # 10 minutes default
UPDATE_INTERVAL = float(os.environ.get("AITOOL_UPDATE_INTERVAL", "5.0"))
ALLOWED_USERS = os.environ.get("AITOOL_ALLOWED_USERS", "")


def is_user_authorized(update: Update) -> bool:
    """Check if the user is authorized to use /aitool."""
    if not ALLOWED_USERS:
        return False  # No users configured = no access

    allowed = [u.strip().lower() for u in ALLOWED_USERS.split(",") if u.strip()]
    user = update.effective_user
    if not user:
        return False

    # Check by user ID or username
    user_id_str = str(user.id)
    username = (user.username or "").lower()
    print(f"username: {username} id: {user_id_str}")

    return user_id_str in allowed or username in allowed


async def handle_aitool(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle /aitool command - run AI tool with prompt and stream output.

    Usage:
        /aitool <prompt>       - Continue previous conversation with prompt
        /aitool new <prompt>   - Start a new conversation with prompt
    """
    chat_id = update.effective_chat.id

    # Check authorization
    if not is_user_authorized(update):
        await context.bot.send_message(
            chat_id=chat_id,
            text="You are not authorized to use this command."
        )
        return

    if not update.message or not context.args:
        await context.bot.send_message(
            chat_id=chat_id,
            text="Usage:\n  /aitool <prompt> - continue conversation\n  /aitool new <prompt> - start new conversation\n\nExample: /aitool fix the bug in main.py"
        )
        return

    # Check for 'new' subcommand
    args = list(context.args)
    use_continue = True
    if args and args[0].lower() == "new":
        use_continue = False
        args = args[1:]

    if not args:
        await context.bot.send_message(
            chat_id=chat_id,
            text="Please provide a prompt after 'new'."
        )
        return

    prompt = " ".join(args)

    # Send initial status
    mode = "continuing" if use_continue else "new conversation"
    await context.bot.send_message(
        chat_id=chat_id,
        text=f"Running: {BINARY} ({mode})\nPrompt: {prompt[:100]}{'...' if len(prompt) > 100 else ''}\n\nStarting..."
    )

    # Build command: claude --allowedTools '...' -p [-c] "prompt"
    cmd_parts = [BINARY] + shlex.split(ARGS)
    if use_continue:
        cmd_parts.append("-c")
    cmd_parts.append(prompt)

    work_dir = WORK_DIR if WORK_DIR else None

    output_buffer = []
    process = None

    try:
        # Start the process
        print(f"Starting prompt: {cmd_parts}")
        process = await asyncio.create_subprocess_exec(
            *cmd_parts,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
            cwd=work_dir,
        )

        async def read_output():
            """Read output from process and update buffer."""
            while True:
                try:
                    line = await asyncio.wait_for(
                        process.stdout.readline(),
                        timeout=1.0
                    )
                    if not line:
                        break
                    decoded = line.decode('utf-8', errors='replace').rstrip()
                    if decoded:
                        output_buffer.append(decoded)
                except asyncio.TimeoutError:
                    continue

        async def send_updates():
            """Periodically send output updates to user."""
            last_sent_len = 0
            while process.returncode is None:
                await asyncio.sleep(UPDATE_INTERVAL)
                if len(output_buffer) > last_sent_len:
                    # Get new lines since last update
                    new_lines = output_buffer[last_sent_len:]
                    last_sent_len = len(output_buffer)

                    # Truncate if too long for Telegram (4096 char limit)
                    text = "\n".join(new_lines)
                    if len(text) > 3900:
                        text = text[-3900:]
                        text = "...(truncated)\n" + text

                    try:
                        await context.bot.send_message(
                            chat_id=chat_id,
                            text=f"Output:\n```\n{text}\n```",
                            parse_mode="Markdown"
                        )
                    except Exception:
                        # Fallback without markdown if it fails
                        try:
                            await context.bot.send_message(
                                chat_id=chat_id,
                                text=f"Output:\n{text}"
                            )
                        except Exception:
                            pass

        # Run both tasks with timeout
        try:
            read_task = asyncio.create_task(read_output())
            update_task = asyncio.create_task(send_updates())

            await asyncio.wait_for(read_task, timeout=TIMEOUT)
            update_task.cancel()

        except asyncio.TimeoutError:
            await context.bot.send_message(
                chat_id=chat_id,
                text=f"Timeout after {TIMEOUT}s - terminating process..."
            )
            process.terminate()
            try:
                await asyncio.wait_for(process.wait(), timeout=5)
            except asyncio.TimeoutError:
                process.kill()

        # Wait for process to complete
        await process.wait()
        return_code = process.returncode

        # Send final output
        if output_buffer:
            final_output = "\n".join(output_buffer[-50:])  # Last 50 lines
            if len(output_buffer) > 50:
                final_output = f"...(showing last 50 of {len(output_buffer)} lines)\n" + final_output
            if len(final_output) > 3900:
                final_output = final_output[-3900:]

            await context.bot.send_message(
                chat_id=chat_id,
                text=f"Completed (exit code: {return_code})\n\nFinal output:\n```\n{final_output}\n```",
                parse_mode="Markdown"
            )
        else:
            await context.bot.send_message(
                chat_id=chat_id,
                text=f"Completed (exit code: {return_code})\n\n(no output)"
            )

    except FileNotFoundError:
        await context.bot.send_message(
            chat_id=chat_id,
            text=f"Error: Binary '{BINARY}' not found. Make sure it's installed and in PATH."
        )
    except Exception as e:
        await context.bot.send_message(
            chat_id=chat_id,
            text=f"Error: {e}"
        )
    finally:
        if process and process.returncode is None:
            try:
                process.kill()
            except Exception:
                pass


async def handle_commit(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle /commit command - git add, commit and push changes."""
    chat_id = update.effective_chat.id

    # Get commit message from command arguments, or use default
    commit_msg = " ".join(context.args) if context.args else "Update from bot"

    try:
        # Git add all changes
        result = subprocess.run(
            ["git", "add", "-A"],
            cwd=WORK_DIR,
            capture_output=True,
            text=True
        )
        if result.returncode != 0:
            await context.bot.send_message(
                chat_id=chat_id,
                text=f"git add failed: {result.stderr}"
            )
            return

        # Git commit
        result = subprocess.run(
            ["git", "commit", "-m", commit_msg],
            cwd=WORK_DIR,
            capture_output=True,
            text=True
        )
        if result.returncode != 0:
            if "nothing to commit" in result.stdout or "nothing to commit" in result.stderr:
                await context.bot.send_message(
                    chat_id=chat_id,
                    text="Nothing to commit."
                )
                return
            await context.bot.send_message(
                chat_id=chat_id,
                text=f"git commit failed: {result.stderr}"
            )
            return

        # Git push
        result = subprocess.run(
            ["git", "push"],
            cwd=WORK_DIR,
            capture_output=True,
            text=True
        )
        if result.returncode != 0:
            await context.bot.send_message(
                chat_id=chat_id,
                text=f"git push failed: {result.stderr}"
            )
            return

        await context.bot.send_message(
            chat_id=chat_id,
            text=f"Changes committed and pushed.\nMessage: {commit_msg}"
        )

    except Exception as e:
        await context.bot.send_message(
            chat_id=chat_id,
            text=f"Error: {e}"
        )
