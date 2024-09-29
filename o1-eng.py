import os
import logging
import time
from openai import OpenAI
from termcolor import colored
from prompt_toolkit import prompt
from prompt_toolkit.styles import Style
from prompt_toolkit.completion import WordCompleter
from rich import print as rprint
from rich.markdown import Markdown
from rich.console import Console
import difflib
import re

# Initialize OpenAI client
client = OpenAI(api_key="YOUR KEY")

# System prompt to be included with edit requests
SYSTEM_PROMPT = """You are an advanced AI assistant designed to analyze and modify large text-based files based on user instructions. Your primary objective is to provide a complete, updated version of the file that incorporates the requested changes.

When given a user request and one or more files, perform the following steps:

1. Understand the User Request: Carefully interpret what the user wants to achieve with the modification.
2. Analyze the File: Review the entire content of the provided file.
3. Generate a Complete Updated File: Provide the full content of the updated file, incorporating all necessary changes to address the user's request.

IMPORTANT: Your response must contain only the complete, updated content of the file. Do not include any explanations, additional text, or code block markers (such as ```html or ```). The entire response should be valid content for the file being edited.

Example of the expected format:
```html
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Updated Snake Game</title>
    <style>
        /* CSS styles here */
    </style>
</head>
<body>
    <!-- HTML content here -->
    <script>
        // JavaScript code here
    </script>
</body>
</html>
```

Ensure that your response strictly follows this structure, providing the entire updated file content without any markdown or code block formatting."""

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

# Add this near the top of the file, with other global variables
last_ai_response = None
conversation_history = []

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
    global last_ai_response, conversation_history
    try:
        # Include added file contents and conversation history in the user message
        if added_files:
            file_context = "Added files:\n"
            for file_path, content in added_files.items():
                file_context += f"File: {file_path}\nContent:\n{content}\n\n"
            user_message = f"{file_context}\n{user_message}"

        # Include conversation history
        if not is_edit_request:
            history = "\n".join([f"User: {msg}" if i % 2 == 0 else f"AI: {msg}" for i, msg in enumerate(conversation_history)])
            if history:
                user_message = f"{history}\nUser: {user_message}"

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
            model="o1-mini",
            messages=messages,
            max_completion_tokens=60000  
        )
        logging.info("Received response from AI.")
        last_ai_response = response.choices[0].message.content

        if not is_edit_request:
            # Update conversation history
            conversation_history.append(user_message)
            conversation_history.append(last_ai_response)
            if len(conversation_history) > 20:  # 10 interactions (user + AI each)
                conversation_history = conversation_history[-20:]

        return last_ai_response
    except Exception as e:
        print(colored(f"Error while communicating with OpenAI: {e}", "red"))
        logging.error(f"Error while communicating with OpenAI: {e}")
        return None

def apply_modifications(new_content, file_path):
    if new_content is None:
        print(colored(f"No changes to apply for {file_path}", "yellow"))
        return True

    try:
        with open(file_path, 'r') as file:
            old_content = file.read()

        # Remove any leading/trailing whitespace and newlines
        new_content = new_content.strip()
        old_content = old_content.strip()

        if old_content == new_content:
            print(colored(f"No changes detected in {file_path}", "yellow"))
            return True

        display_diff(old_content, new_content, file_path)

        confirm = prompt(f"Apply these changes to {file_path}? (yes/no): ", style=Style.from_dict({'prompt': 'yellow'})).strip().lower()
        if confirm == 'yes':
            with open(file_path, 'w') as file:
                file.write(new_content)
            print(colored(f"Modifications applied to {file_path} successfully.", "green"))
            logging.info(f"Modifications applied to {file_path} successfully.")
            return True
        else:
            print(colored(f"Changes not applied to {file_path}.", "yellow"))
            logging.info(f"User chose not to apply changes to {file_path}.")
            return False

    except Exception as e:
        print(colored(f"An error occurred while applying modifications to {file_path}: {e}", "red"))
        logging.error(f"Error applying modifications to {file_path}: {e}")
        return False

def display_diff(old_content, new_content, file_path):
    diff = list(difflib.unified_diff(
        old_content.splitlines(keepends=True),
        new_content.splitlines(keepends=True),
        fromfile=f"a/{file_path}",
        tofile=f"b/{file_path}",
        lineterm='',
        n=5
    ))
    
    if not diff:
        print(f"No changes detected in {file_path}")
        return

    print(f"\nDiff for {file_path}:")

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

    console = Console()
    console.print(Markdown(markdown_diff))

def apply_creation_steps(creation_response, added_files, retry_count=0):
    max_retries = 3
    try:
        code_blocks = re.findall(r'```(?:\w+)?\s*([\s\S]*?)```', creation_response)
        if not code_blocks:
            raise ValueError("No code blocks found in the AI response.")

        print("Successfully extracted code blocks:")
        logging.info("Successfully extracted code blocks from creation response.")

        for code in code_blocks:
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
            else:
                print(colored("Error: Could not determine the file or folder information from the code block.", "red"))
                logging.error("Could not determine the file or folder information from the code block.")
                continue

        return True

    except ValueError as e:
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
    global last_ai_response, conversation_history
    print(colored("AI File Editor is ready to help you.", "cyan"))
    print("\nAvailable commands:")
    print(f"{colored('/edit', 'magenta'):<10} {colored('Edit files (followed by file paths)', 'dark_grey')}")
    print(f"{colored('/create', 'magenta'):<10} {colored('Create files or folders (followed by instructions)', 'dark_grey')}")
    print(f"{colored('/add', 'magenta'):<10} {colored('Add files to context', 'dark_grey')}")
    print(f"{colored('/debug', 'magenta'):<10} {colored('Print the last AI response', 'dark_grey')}")
    print(f"{colored('/quit', 'magenta'):<10} {colored('Exit the program', 'dark_grey')}")

    style = Style.from_dict({
        'prompt': 'yellow',
    })

    # Get the list of files in the current directory
    files = [f for f in os.listdir('.') if os.path.isfile(f)]

    # Create a WordCompleter with available commands and files
    completer = WordCompleter(['/edit', '/create', '/add', '/quit', '/debug'] + files, ignore_case=True)

    added_files = {}

    while True:
        print()  # Add a newline before the prompt
        user_input = prompt("You: ", style=style, completer=completer).strip()

        if user_input.lower() == '/quit':
            print("Goodbye!")
            logging.info("User exited the program.")
            break

        elif user_input.lower() == '/debug':
            if last_ai_response:
                print(colored("Last AI Response:", "blue"))
                print(last_ai_response)
            else:
                print(colored("No AI response available yet.", "yellow"))

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

            file_contents = {}
            for file_path in file_paths:
                try:
                    with open(file_path, 'r', encoding='utf-8') as file:
                        file_contents[file_path] = file.read()
                except Exception as e:
                    print(colored(f"Error reading file {file_path}: {e}", "red"))
                    logging.error(f"Error reading file {file_path}: {e}")
                    continue

            if not file_contents:
                print(colored("No valid files to edit.", "yellow"))
                continue

            edit_instruction = prompt(f"Edit Instruction for all files: ", style=style).strip()

            edit_request = f"""User request: {edit_instruction}

Files to modify:
"""
            for file_path, content in file_contents.items():
                edit_request += f"\nFile: {file_path}\nContent:\n{content}\n\n"

            edit_request += "\nIMPORTANT: Your response must contain only the complete, updated content of each modified file. Do not include any explanations or additional text. Provide the full content for each file you modify, even if some files remain unchanged."

            ai_response = chat_with_ai(edit_request, is_edit_request=True, added_files=added_files)

            if ai_response:
                modified_files = parse_ai_response(ai_response, file_contents.keys())
                for file_path, new_content in modified_files.items():
                    success = apply_modifications(new_content, file_path)
                    if not success:
                        retry = prompt(f"Modifications for {file_path} failed. Do you want the AI to try again for all files? (yes/no): ", style=style).strip().lower()
                        if retry == 'yes':
                            ai_response = chat_with_ai(f"The previous modifications were not fully successful. Please try again with a different approach, providing the complete updated content for all files.", is_edit_request=True, added_files=added_files)
                            modified_files = parse_ai_response(ai_response, file_contents.keys())
                            for file_path, new_content in modified_files.items():
                                apply_modifications(new_content, file_path)
                        break

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

def parse_ai_response(response, original_files):
    modified_files = {}
    current_file = None
    current_content = []

    lines = response.split('\n')
    
    # Check if the response is a single file content without "File: " prefix
    if len(original_files) == 1 and not lines[0].startswith("File: "):
        modified_files[list(original_files)[0]] = response
    else:
        for line in lines:
            if line.startswith("File: "):
                if current_file and current_content:
                    modified_files[current_file] = '\n'.join(current_content)
                current_file = line[6:].strip()
                current_content = []
            elif current_file:
                current_content.append(line)

        if current_file and current_content:
            modified_files[current_file] = '\n'.join(current_content)

    # Add any original files that weren't modified
    for file in original_files:
        if file not in modified_files:
            modified_files[file] = None

    return modified_files

if __name__ == "__main__":
    main()
