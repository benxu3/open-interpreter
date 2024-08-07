from yaspin import yaspin

spinner = yaspin()
spinner.start()

import os
import platform
import re
import time

import platformdirs
import pyperclip
import yaml
from pynput.keyboard import Controller, Key

os.environ["LITELLM_LOCAL_MODEL_COST_MAP"] = "True"
import litellm

SYSTEM_MESSAGE = f"""
You are a fast, efficient AI assistant specialized in analyzing terminal history and providing solutions. You are summoned via the wtf command. Your task is to:

1. Scan the provided terminal history.
2. Identify the most recent error or issue.
3. Take a deep breath, and thoughtfully, carefully determine the most likely solution or debugging step.
4. Respond with a VERY brief explanation followed by a markdown code block containing a shell command to address the issue.

Rules:
- Provide a single shell command in your code block, using line continuation characters (\\ for Unix-like systems, ^ for Windows) for multiline commands.
- Ensure the entire command is on one logical line, requiring the user to press enter only once to execute.
- If multiple steps are needed, explain the process briefly, then provide only the first command or a combined command using && or ;.
- Keep any explanatory text extremely brief and concise.
- Place explanatory text before the code block.
- NEVER USE COMMENTS IN YOUR CODE.
- Construct the command with proper escaping: e.g. use sed with correctly escaped quotes to ensure the shell interprets the command correctly. This involves:
	•	Using double quotes around the sed expression to handle single quotes within the command.
	•	Combining single and double quotes to properly escape characters within the shell command.
- If previous commands attempted to fix the issue and failed, learn from them by proposing a DIFFERENT command.
- Focus on the most recent error, ignoring earlier unrelated commands.
- If you need more information to confidently fix the problem, ask the user to run wtf again in a moment, then write a command like grep to learn more about the problem.
- The error may be as simple as a spelling error, or as complex as requiring tests to be run, or code to be find-and-replaced.
- Prioritize speed and conciseness in your response. Don't use markdown headings. Don't say more than a sentence or two. Be incredibly concise.

User's System: {platform.system()}
CWD: {os.getcwd()}
{"Shell: " + os.environ.get('SHELL') if os.environ.get('SHELL') else ''}

"""


def main():
    keyboard = Controller()

    # Save clipboard
    clipboard = pyperclip.paste()

    # Select all text
    shortcut_key = Key.cmd if platform.system() == "Darwin" else Key.ctrl
    with keyboard.pressed(shortcut_key):
        keyboard.press("a")
        keyboard.release("a")

    # Copy selected text
    with keyboard.pressed(shortcut_key):
        keyboard.press("c")
        keyboard.release("c")

    # Deselect
    keyboard.press(Key.backspace)
    keyboard.release(Key.backspace)

    # Wait for the clipboard to update
    time.sleep(0.1)

    # Get terminal history from clipboard
    history = pyperclip.paste()

    # Reset clipboard to stored one
    pyperclip.copy(clipboard)

    # Trim history
    history = history[-9000:].strip()

    # Remove any trailing spinner commands
    spinner_commands = [
        "⠴",
        "⠦",
        "⠇",
        "⠉",
        "⠙",
        "⠸",
        "⠼",
        "⠤",
        "⠴",
        "⠂",
        "⠄",
        "⠈",
        "⠐",
        "⠠",
    ]
    for command in spinner_commands:
        if history.endswith(command):
            history = history[: -len(command)].strip()
            break

    commands_to_remove = ["poetry run wtf", "wtf"]
    for command in commands_to_remove:
        if history.endswith(command):
            history = history[: -len(command)].strip()
            break

    # Get error context

    # Regex pattern to extract filename and line number
    pattern = r'File "([^"]+)", line (\d+)'
    matches = re.findall(pattern, history)

    # Only keep the last X matches
    matches = matches[-1:]  # Just the last match, change -1 to get more

    # Function to get specified lines from a file
    def get_lines_from_file(filename, line_number):
        lines = []
        try:
            with open(filename, "r") as file:
                all_lines = file.readlines()
                start_line = max(0, line_number - 3)  # Preceding lines
                end_line = min(len(all_lines), line_number + 2)  # Following lines
                for i in range(start_line, end_line + 1):
                    lines.append(f"Line {i+1}: " + all_lines[i].rstrip())
        except Exception as e:
            lines.append(f"Error reading file: {e}")
        return lines

    # Create the dictionary with filename, line number, and text
    result = []
    for match in matches:
        filename, line_number = match
        line_number = int(line_number)
        lines = get_lines_from_file(filename, line_number)
        result.append({"filename": filename, "text": "\n".join(lines)})

    if result != []:
        history = "Terminal: " + history

    # Add context
    for entry in result:
        history = f"""File: {entry["filename"]}\n{entry["text"]}\n\n""" + history

    # print(history)
    # print("---")
    # time.sleep(10)

    # Prepare messages for LLM
    messages = [
        {"role": "system", "content": SYSTEM_MESSAGE},
        {"role": "user", "content": history},
    ]

    # Get LLM model from profile
    default_profile_path = os.path.join(
        platformdirs.user_config_dir("open-interpreter"), "profiles", "default.yaml"
    )

    try:
        with open(default_profile_path, "r") as file:
            profile = yaml.safe_load(file)
            wtf_model = profile.get("wtf", {}).get("model")
            if wtf_model:
                model = wtf_model
            else:
                model = profile.get("llm", {}).get("model", "gpt-3.5-turbo")
    except:
        model = "gpt-3.5-turbo"

    # Process LLM response
    in_code = False
    backtick_count = 0
    language_buffer = ""

    started = False

    for chunk in litellm.completion(
        model=model, messages=messages, temperature=0, stream=True
    ):
        if not started:
            started = True
            spinner.stop()
            print("")

        content = chunk.choices[0].delta.content
        if content:
            for char in content:
                if char == "`":
                    backtick_count += 1
                    if backtick_count == 3:
                        in_code = not in_code
                        backtick_count = 0
                        language_buffer = ""
                        if not in_code:  # We've just exited a code block
                            time.sleep(0.1)
                            print("\n")
                            return  # Exit after typing the command
                        else:  # Entered code block
                            print("Press `enter` to run: ", end="", flush=True)
                elif in_code:
                    if language_buffer is not None:
                        if char.isalnum():
                            language_buffer += char
                        elif char.isspace():
                            language_buffer = None
                    elif char not in ["\n", "\\"]:
                        keyboard.type(char)
                else:
                    if backtick_count:
                        print("`" * backtick_count, end="", flush=True)
                        backtick_count = 0

                    # if "\n" in char:
                    #     char.replace("\n", "\n    ")

                    print(char, end="", flush=True)

                    backtick_count = 0


if __name__ == "__main__":
    main()
