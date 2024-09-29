import os
import json
import logging
import re
import time
from openai import OpenAI
from termcolor import colored
from prompt_toolkit import prompt
from prompt_toolkit.styles import Style
from prompt_toolkit.completion import WordCompleter
from rich import print as rprint
from rich.markdown import Markdown
from rich.console import Console
from rich.text import Text
from rich.syntax import Syntax
import difflib
from fuzzywuzzy import fuzz  # Add this import for fuzzy matching


# Initialize OpenAI client
client = OpenAI(api_key="YOUR KEY")

# System prompt to be included with edit requests
SYSTEM_PROMPT = """You are an advanced AI assistant designed to analyze and modify large text-based files based on user instructions. Your primary objective is to identify the specific sections within one or multiple provided files that need to be updated or changed, regardless of the file size. 

When given a user request and one or more files, perform the following steps:

1. Understand the User Request: Carefully interpret what the user wants to achieve with the modification.
2. Analyze the Files: Efficiently parse the provided file(s) to locate the exact parts that are relevant to the user's request.
3. Generate Modification Instructions: For each identified section that requires changes, provide clear and precise instructions using the following format:

<search>
```language
# Original code to be replaced
```
</search>
<replace>
```language
# New code to replace the original
```
</replace>

4. Ensure Clarity and Precision: Your instructions must be unambiguous to allow accurate implementation without additional clarification.

IMPORTANT: Your response must only contain the search and replace blocks as described above, with no additional text before or after. Use the appropriate language identifier in the code blocks (e.g., ```python, ```javascript, etc.).

Example of the expected format:

<search>
```python
def old_function():
    print("This is the old implementation")
```
</search>
<replace>
```python
def new_function():
    print("This is the new implementation")
    print("With an extra line")
```
</replace>

<search>
```python
# Old configuration
MAX_RETRIES = 3
```
</search>
<replace>
```python
# New configuration
MAX_RETRIES = 5
TIMEOUT = 30
```
</replace>

Ensure that your response strictly follows this structure to facilitate seamless integration with the modification script."""

# Updated CREATE_SYSTEM_PROMPT to request code blocks instead of JSON
CREATE_SYSTEM_PROMPT = """You are an advanced AI assistant designed to create files and folders based on user instructions. Your primary objective is to generate the content of the files to be created as code blocks. Each code block should specify whether it's a file or folder, along with its path.

When given a user request, perform the following steps:

1. Understand the User Request: Carefully interpret what the user wants to create.
2. Generate Creation Instructions: Provide the content for each file to be created within appropriate code blocks. Each code block should begin with a special comment line that specifies whether it's a file or folder, along with its path.

IMPORTANT: Your response must ONLY contain the code blocks with no additional text before or after. Do not use markdown formatting outside of the code blocks. Use the following format for the special comment line:

For folders:
```
### FOLDER: path/to/folder
```

For files:
```language
### FILE: path/to/file.extension
File content goes here...
```

Example of the expected format:

```
### FOLDER: new_app
```

```html
### FILE: new_app/index.html
<!DOCTYPE html>
<html>
<head>
    <title>New App</title>
</head>
<body>
    <h1>Hello, World!</h1>
</body>
</html>
```

```css
### FILE: new_app/styles.css
body {
    font-family: Arial, sans-serif;
}
```

```javascript
### FILE: new_app/script.js
console.log('Hello, World!');
```

Ensure that each file and folder is correctly specified to facilitate seamless creation by the script."""

def is_binary_file(file_path):
    """Check if a file is binary."""
    try:
        with open(file_path, 'tr') as check_file:
            check_file.read()
        return False
    except UnicodeDecodeError:
        return True

def add_file_to_context(file_path, added_files):
    if not os.path.isfile(file_path):
        print(colored(f"Error: {file_path} is not a valid file.", "red"))
        logging.error(f"{file_path} is not a valid file.")
        return

    if is_binary_file(file_path):
        print(colored(f"Error: {file_path} appears to be a binary file and cannot be added.", "red"))
        logging.error(f"{file_path} is a binary file and cannot be added.")
        return

    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            content = file.read()
            added_files[file_path] = content
            print(colored(f"Added {file_path} to the chat context.", "green"))
            logging.info(f"Added {file_path} to the chat context.")
    except Exception as e:
        print(colored(f"Error reading file {file_path}: {e}", "red"))
        logging.error(f"Error reading file {file_path}: {e}")

def chat_with_ai(user_message, is_edit_request=False, retry_count=0, added_files=None):
    try:
        # Include added file contents in the user message
        if added_files:
            file_context = "Added files:\n"
            for file_path, content in added_files.items():
                file_context += f"File: {file_path}\nContent:\n{content}\n\n"
            user_message = f"{file_context}\n{user_message}"

        messages = [
            {
                "role": "user",
                "content": (SYSTEM_PROMPT + "\n\nUser request: " + user_message) if is_edit_request else user_message
            }
        ]
        
        if is_edit_request and retry_count == 0:
            print(colored("Analyzing files and generating modifications...", "magenta"))
            logging.info("Sending edit request to AI.")
        elif not is_edit_request:
            print(colored("AI assistant is thinking...", "magenta"))
            logging.info("Sending general query to AI.")

        response = client.chat.completions.create(
            model="gpt-4",
            messages=messages,
            max_tokens=4000
        )
        logging.info("Received response from AI.")
        return response.choices[0].message.content
    except Exception as e:
        print(colored(f"Error while communicating with OpenAI: {e}", "red"))
        logging.error(f"Error while communicating with OpenAI: {e}")
        return None


def is_potential_json(text):
    """Check if the text potentially contains JSON."""
    text = text.strip()
    return text.startswith('{') and text.endswith('}')

def extract_code_blocks(text):
    """
    Extracts all code blocks from the text.
    Returns a list of tuples: (language, code_content)
    """
    # Improved regex to handle optional language and variations in formatting
    code_block_pattern = r'```(?:\s*(\w+))?\s*\n([\s\S]*?)```'
    matches = re.findall(code_block_pattern, text, re.MULTILINE)
    return matches


def apply_modifications(modifications_text, retry_count=0):
    max_retries = 3
    
    style = Style.from_dict({
        'prompt': 'yellow',
    })
    
    try:
        blocks = re.findall(r'<search>[\s\S]*?</search>[\s\S]*?<replace>[\s\S]*?</replace>', modifications_text, re.DOTALL)
        
        if not blocks:
            raise ValueError("No valid search and replace blocks found in the AI response.")

        logging.info(f"Found {len(blocks)} modification blocks.")

        successful_edits = 0
        total_edits = len(blocks)

        console = Console()  # For displaying diffs

        for block in blocks:
            search_match = re.search(r'<search>\s*(```[\s\S]*?```)\s*</search>', block, re.DOTALL)
            replace_match = re.search(r'<replace>\s*(```[\s\S]*?```)\s*</replace>', block, re.DOTALL)

            if not search_match or not replace_match:
                logging.warning("Invalid search or replace block found. Skipping.")
                continue

            search_code = extract_code_from_block(search_match.group(1))
            replace_code = extract_code_from_block(replace_match.group(1))

            if not search_code or not replace_code:
                logging.warning("Empty search or replace block found. Skipping.")
                continue

            matching_files = find_files_with_code(search_code)

            if not matching_files:
                print(colored(f"No exact matches found. Trying fuzzy search...", "yellow"))
                matching_files = find_files_with_fuzzy_match(search_code)

            if not matching_files:
                print(colored(f"No files found containing the search code.", "yellow"))
                logging.warning(f"No files found containing the search code.")
                continue

            for file_path in matching_files:
                try:
                    with open(file_path, 'r', encoding='utf-8') as file:
                        content = file.read()

                    new_content = fuzzy_replace(content, search_code, replace_code)

                    if new_content == content:
                        print(colored(f"No changes needed in {file_path}", "yellow"))
                        logging.info(f"No changes needed in {file_path}")
                        continue

                    display_diff(content, new_content, file_path, console)

                    confirm = prompt(f"Apply these changes to {file_path}? (yes/no): ", style=style).strip().lower()
                    if confirm == 'yes':
                        with open(file_path, 'w', encoding='utf-8') as file:
                            file.write(new_content)
                        print(colored(f"Modifications applied to {file_path} successfully.", "green"))
                        logging.info(f"Modifications applied to {file_path} successfully.")
                        successful_edits += 1
                    else:
                        print(colored("Changes not applied.", "yellow"))
                        logging.info(f"User chose not to apply changes to {file_path}.")

                except Exception as e:
                    print(colored(f"Error processing file {file_path}: {e}", "red"))
                    logging.error(f"Error processing file {file_path}: {e}")

        if successful_edits == total_edits:
            print(colored(f"All {successful_edits} edits were successful!", "green"))
            logging.info(f"All {successful_edits} edits were successful.")
            return True
        else:
            print(colored(f"{successful_edits} out of {total_edits} edits were successful.", "yellow"))
            logging.warning(f"{successful_edits} out of {total_edits} edits were successful.")
            return False

    except ValueError as e:
        if retry_count < max_retries:
            print(colored(f"Error: {str(e)} Retrying... (Attempt {retry_count + 1})", "yellow"))
            logging.warning(f"Modification parsing failed: {str(e)}. Retrying... (Attempt {retry_count + 1})")
            error_message = f"{str(e)} Please provide the modifications again using the specified format with <search> and <replace> blocks."
            time.sleep(2 ** retry_count)  # Exponential backoff
            new_modifications = chat_with_ai(error_message, is_edit_request=True)
            if new_modifications:
                return apply_modifications(new_modifications, retry_count + 1)
            else:
                return False
        else:
            print(colored(f"Failed to parse modifications after multiple attempts: {str(e)}", "red"))
            logging.error(f"Failed to parse modifications after multiple attempts: {str(e)}")
            print("Modifications that failed to parse:")
            print(modifications_text)
            return False
    except Exception as e:
        print(colored(f"An unexpected error occurred: {e}", "red"))
        logging.error(f"An unexpected error occurred: {e}")
        return False

def extract_code_from_block(block):
    match = re.search(r'```(?:\w+)?\s*([\s\S]*?)\s*```', block)
    if match:
        return match.group(1).strip()
    else:
        logging.warning(f"Failed to extract code from block: {block}")
        return None

def find_files_with_code(search_code):
    matching_files = []
    for root, dirs, files in os.walk('.'):
        for file in files:
            if file.endswith(('.py', '.js', '.html', '.css')):  # Add more extensions if needed
                file_path = os.path.join(root, file)
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                        if search_code in content:
                            matching_files.append(file_path)
                except Exception as e:
                    print(colored(f"Error reading file {file_path}: {e}", "red"))
                    logging.error(f"Error reading file {file_path}: {e}")
    return matching_files

def find_files_with_fuzzy_match(search_code, threshold=80):
    matching_files = []
    for root, dirs, files in os.walk('.'):
        for file in files:
            if file.endswith(('.py', '.js', '.html', '.css')):  # Add more extensions if needed
                file_path = os.path.join(root, file)
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                        if fuzz.partial_ratio(search_code, content) >= threshold:
                            matching_files.append(file_path)
                except Exception as e:
                    print(colored(f"Error reading file {file_path}: {e}", "red"))
                    logging.error(f"Error reading file {file_path}: {e}")
    return matching_files

def fuzzy_replace(content, search_code, replace_code):
    lines = content.splitlines()
    new_lines = []
    i = 0
    while i < len(lines):
        if fuzz.partial_ratio(search_code, '\n'.join(lines[i:i+len(search_code.splitlines())])) >= 80:
            new_lines.extend(replace_code.splitlines())
            i += len(search_code.splitlines())
        else:
            new_lines.append(lines[i])
            i += 1
    return '\n'.join(new_lines)

def display_diff(old_content, new_content, file_path, console):
    old_lines = old_content.splitlines()
    new_lines = new_content.splitlines()
    
    diff = list(difflib.unified_diff(old_lines, new_lines, lineterm='', n=5))
    
    if not diff:
        console.print(f"No changes detected in {file_path}")
        return

    console.print(f"\nDiff for {file_path}:")
    
    markdown_diff = "```diff\n"
    for line in diff:
        if line.startswith('+'):
            markdown_diff += line + "\n"
        elif line.startswith('-'):
            markdown_diff += line + "\n"
        elif line.startswith('^'):
            markdown_diff += line + "\n"
        else:
            markdown_diff += " " + line + "\n"
    markdown_diff += "```"

    console.print(Markdown(markdown_diff))

def apply_creation_steps(creation_response, added_files, retry_count=0):
    max_retries = 3
    try:
        code_blocks = extract_code_blocks(creation_response)
        if not code_blocks:
            raise ValueError("No code blocks found in the AI response.")

        print("Successfully extracted code blocks:")
        logging.info("Successfully extracted code blocks from creation response.")

        for language, code in code_blocks:
            # Extract file/folder information from the special comment line
            info_match = re.match(r'### (FILE|FOLDER): (.+)', code.strip())
            
            if info_match:
                item_type, path = info_match.groups()
                
                if item_type == 'FOLDER':
                    # Create the folder
                    os.makedirs(path, exist_ok=True)
                    print(colored(f"Folder created: {path}", "green"))
                    logging.info(f"Folder created: {path}")
                elif item_type == 'FILE':
                    # Extract file content (everything after the special comment line)
                    file_content = re.sub(r'### FILE: .+\n', '', code, count=1).strip()

                    # Create directories if necessary
                    directory = os.path.dirname(path)
                    if directory and not os.path.exists(directory):
                        os.makedirs(directory, exist_ok=True)
                        print(colored(f"Folder created: {directory}", "green"))
                        logging.info(f"Folder created: {directory}")

                    # Write content to the file
                    with open(path, 'w', encoding='utf-8') as f:
                        f.write(file_content)
                    print(colored(f"File created: {path}", "green"))
                    logging.info(f"File created: {path}")

                    # Remove the linting call
                    # if path.endswith(('.py', '.js', '.jsx', '.ts', '.tsx')):
                    #     lint_file(path)
            else:
                print(colored("Error: Could not determine the file or folder information from the code block.", "red"))
                logging.error("Could not determine the file or folder information from the code block.")
                continue

        return True

    except (ValueError, re.error) as e:
        if retry_count < max_retries:
            print(colored(f"Error: {str(e)} Retrying... (Attempt {retry_count + 1})", "yellow"))
            logging.warning(f"Creation parsing failed: {str(e)}. Retrying... (Attempt {retry_count + 1})")
            error_message = f"{str(e)} Please provide the creation instructions again using the specified format."
            time.sleep(2 ** retry_count)  # Exponential backoff
            new_response = chat_with_ai(error_message, is_edit_request=False, added_files=added_files)
            if new_response:
                return apply_creation_steps(new_response, added_files, retry_count + 1)
            else:
                return False
        else:
            print(colored(f"Failed to parse creation instructions after multiple attempts: {str(e)}", "red"))
            logging.error(f"Failed to parse creation instructions after multiple attempts: {str(e)}")
            print("Creation response that failed to parse:")
            print(creation_response)
            return False
    except Exception as e:
        print(colored(f"An unexpected error occurred during creation: {e}", "red"))
        logging.error(f"An unexpected error occurred during creation: {e}")
        return False

def main():
    print(colored("AI File Editor is ready to help you.", "cyan"))
    print("\nAvailable commands:")
    print(f"{colored('/edit', 'magenta'):<10} {colored('Edit files (followed by file paths)', 'dark_grey')}")
    print(f"{colored('/create', 'magenta'):<10} {colored('Create files or folders (followed by instructions)', 'dark_grey')}")
    print(f"{colored('/add', 'magenta'):<10} {colored('Add files to context', 'dark_grey')}")
    print(f"{colored('/quit', 'magenta'):<10} {colored('Exit the program', 'dark_grey')}")

    style = Style.from_dict({
        'prompt': 'yellow',
    })

    # Get the list of files in the current directory
    files = [f for f in os.listdir('.') if os.path.isfile(f)]

    # Create a WordCompleter with available commands and files
    completer = WordCompleter(['/edit', '/create', '/add', '/quit'] + files, ignore_case=True)

    added_files = {}

    while True:
        print()  # Add a newline before the prompt
        user_input = prompt("You: ", style=style, completer=completer).strip()

        if user_input.lower() == '/quit':
            print("Goodbye!")
            logging.info("User exited the program.")
            break

        elif user_input.startswith('/add'):
            file_paths = user_input.split()[1:]
            if not file_paths:
                print(colored("Please provide at least one file path.", "yellow"))
                logging.warning("User issued /add without file paths.")
                continue

            for file_path in file_paths:
                add_file_to_context(file_path, added_files)

            total_size = sum(len(content) for content in added_files.values())
            if total_size > 100000:  # Warning if total content exceeds ~100KB
                print(colored("Warning: The total size of added files is large and may affect performance.", "yellow"))
                logging.warning("Total size of added files exceeds 100KB.")

        elif user_input.startswith('/edit'):
            file_paths = user_input.split()[1:]
            if not file_paths:
                print(colored("Please provide at least one file path.", "yellow"))
                logging.warning("User issued /edit without file paths.")
                continue

            file_contents = []
            for file_path in file_paths:
                try:
                    with open(file_path, 'r', encoding='utf-8') as file:
                        content = file.read()
                        file_contents.append(f"File: {file_path}\nContent:\n{content}\n")
                except Exception as e:
                    print(colored(f"Error reading file {file_path}: {e}", "red"))
                    logging.error(f"Error reading file {file_path}: {e}")
                    continue

            edit_instruction = prompt("Edit Instruction: ", style=style).strip()

            edit_request = f"""User request: {edit_instruction}

Files to modify: {" ".join(file_contents)}

IMPORTANT: Your response must contain only <search> and <replace> blocks as described in the system prompt. Do not include any other text."""

            ai_response = chat_with_ai(edit_request, is_edit_request=True, added_files=added_files)

            if ai_response:
                success = apply_modifications(ai_response)
                if not success:
                    retry = prompt("Some modifications failed. Do you want to see the AI's response? (yes/no): ", style=style).strip().lower()
                    if retry == 'yes':
                        print("AI Response:")
                        print(ai_response)
                    retry = prompt("Do you want the AI to try again? (yes/no): ", style=style).strip().lower()
                    if retry == 'yes':
                        ai_response = chat_with_ai("The previous modifications were not fully successful. Please try again with a different approach, using only <search> and <replace> blocks.", is_edit_request=True, added_files=added_files)
                        success = apply_modifications(ai_response)

        elif user_input.startswith('/create'):
            creation_instruction = user_input[7:].strip()  # Remove '/create' and leading/trailing whitespace
            if not creation_instruction:
                print(colored("Please provide creation instructions after /create.", "yellow"))
                logging.warning("User issued /create without instructions.")
                continue

            create_request = f"{CREATE_SYSTEM_PROMPT}\n\nUser request: {creation_instruction}"
            ai_response = chat_with_ai(create_request, is_edit_request=False, added_files=added_files)
            
            if ai_response:
                while True:
                    print("AI Assistant: Here is the suggested creation structure:")
                    rprint(Markdown(ai_response))

                    confirm = prompt("Do you want to execute these creation steps? (yes/no): ", style=style).strip().lower()
                    if confirm == 'yes':
                        success = apply_creation_steps(ai_response, added_files)
                        if success:
                            break
                        else:
                            retry = prompt("Creation failed. Do you want the AI to try again? (yes/no): ", style=style).strip().lower()
                            if retry != 'yes':
                                break
                            ai_response = chat_with_ai("The previous creation attempt failed. Please try again with a different approach.", is_edit_request=False, added_files=added_files)
                    else:
                        print(colored("Creation steps not executed.", "yellow"))
                        logging.info("User chose not to execute creation steps.")
                        break

        else:
            ai_response = chat_with_ai(user_input, added_files=added_files)
            if ai_response:
                print()
                print(colored("AI Assistant:", "blue"))
                rprint(Markdown(ai_response))
                logging.info("Provided AI response to user query.")

if __name__ == "__main__":
    main()
