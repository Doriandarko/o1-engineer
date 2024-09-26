from openai import OpenAI
from prompt_toolkit import PromptSession
from prompt_toolkit.styles import Style as PromptStyle
from prompt_toolkit.history import InMemoryHistory
from rich.console import Console
from rich.markdown import Markdown
from pathlib import Path
import yaml  # For YAML parsing
import re
import sys
import threading
import time
import itertools
import os
from typing import List
from dotenv import load_dotenv
import logging
from logging.handlers import RotatingFileHandler
import shutil
import difflib
from fuzzywuzzy import fuzz

def load_yaml(yaml_content):
    try:
        return yaml.load(yaml_content, Loader=yaml.FullLoader)
    except yaml.YAMLError as e:
        raise e


# Initialize Rich console
console = Console()

# Load environment variables from .env file
load_dotenv()  # Ensure this is called before accessing environment variables

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Set the model as a variable at the top of the script
MainModel = "o1-mini"
EditorModel = "o1-mini"
PlanningModel = "o1-preview"

# Initialize total cost variables
total_input_cost = 0.0
total_output_cost = 0.0
total_chat_cost = 0.0

# Define the maximum number of messages to keep in conversation history, the more messages the higher the cost
MAX_CONVERSATION_HISTORY = 10

# Planning prompt
planningprompt = """You are an AI planning assistant. Your task is to create a detailed plan based on the user's request. Consider all aspects of the task, break it down into steps, and provide a comprehensive strategy for accomplishment. Your plan should be clear, actionable, and thorough. Here's the user's request:"""

# Preprompt for code generation
preprompt = """When you are asked to provide code for files, please adhere strictly to the following guidelines:

1. **Response Format**: Return your response **only** as a YAML object enclosed within `<tags>` and `</tags>`. Do not include any text before or after the YAML content.

2. **YAML Structure**:
    - The YAML object should have **filenames** as keys.
    - Each filename maps to an object containing the following keys:
        - `extension`: The file extension without the dot (e.g., `py` for Python files).
        - `code`: A string containing the actual code content.

3. **Code Content**:
    - **Escape Characters**: Use double backslashes (`\\\\`) to represent single backslashes in the code. For newlines, use `\\n`.
    - **String Literals**: Ensure all double quotes (`"`) inside the code are escaped with a backslash (e.g., `\"`).

4. **Example**:
<tags>
src/main.py:
  extension: py
  code: |
    print('Hello, World!')\\nprint('This is a new line')
src/utils/helper.py:
  extension: py
  code: |
    def helper_function():\\n    return 'I am helping!'
</tags>

5. **Folder Creation**: You can create folders by including them in the file paths. The system will automatically create any necessary directories.

6. **No Additional Text**: Do not include any explanations, comments, or any other text outside the `<tags>` block.

7. **Remember**: Only use this format when explicitly asked to provide code for files.
"""

# Editor prompt for direct file updates
def get_formatted_editor_prompt(files_to_edit):
    file_list = '\n'.join([f"- {Path(f).name}" for f in files_to_edit])
    return f"""You are an AI code editor assistant. Your task is to provide specific edits for the files listed below based on the user's request. Follow these guidelines strictly:

1. Available Files:
{file_list}


2. Response Format:
   - Provide your response as a single YAML object within <edit> and </edit> tags.
   - Do not include any text outside these tags.

3. YAML Structure:
   - Use filenames as top-level keys.
   - Each filename should map to a list of edit objects.
   - Each edit object must have 'find' and 'replace' keys.

4. Code Snippet Formatting:
   - Preserve original formatting, including indentation and line breaks.
   - For multiline strings, use the YAML block scalar indicator '|' and indent properly.
   - If code contains special YAML characters (:, -, ?, etc.), wrap the entire block in double quotes (") or use '|'.

Example YAML Structure:

<edit>
example.py:
  - find: |
      def old_function():
          # Old implementation
          pass
    replace: |
      def new_function():
          # New implementation
          return "Updated function"

another_file.py:
  - find: |
      print("Hello, World!")
    replace: |
      print("Greetings, Universe!")
</edit>

6. **No Additional Text:** Do not include any explanations, comments, or any other text outside the `<edit>` block.

7. **Remember:** Only use this format when explicitly asked to provide specific edits for the available files.

**Important:** Be precise and strictly adhere to the format.
"""

# Prompt toolkit style
prompt_style = PromptStyle.from_dict({
    'prompt': '#ansigreen bold',
})

# Function to decode the 'code' field in file information
def decode_code_field(code: str, filename: str) -> str:
    try:
        # First replace double backslashes with single backslashes
        code = code.replace('\\\\', '\\')
        # Then decode unicode escapes
        code = bytes(code, "utf-8").decode("unicode_escape")
        return code
    except UnicodeDecodeError as e:
        console.print(f"[red]Unicode decode error in file {filename}: {str(e)}[/red]")
        logging.error(f"Unicode decode error in file {filename}: {str(e)}")
        return ""

def parse_command(user_input: str):
    """
    Parses the user input to determine if it's a command.
    Returns the command and its arguments if applicable.
    """
    user_input = user_input.strip().lower()
    if user_input == 'exit':
        return ('exit', None)
    elif user_input == 'reset':
        return ('reset', None)
    elif user_input == 'save':
        return ('save', None)
    elif user_input.startswith('/add'):
        parts = user_input.split()
        if len(parts) < 2:
            return ('invalid_add', None)
        files = parts[1:]
        return ('add', files)
    elif user_input.startswith('/edit'):
        parts = user_input.split()
        if len(parts) < 2:
            return ('invalid_edit', None)
        files = parts[1:]
        return ('edit', files)
    else:
        return (None, None)

def add_files_to_context(files: List[str], conversation_history: list):
    """
    Adds the content of the specified files to the conversation context.

    Args:
        files (List[str]): A list of file paths.
        conversation_history (list): The current conversation history.
    """
    for file_path in files:
        path = Path(file_path)
        if not path.is_absolute():
            path = Path.cwd() / path

        if not path.is_file():
            console.print(f"[red]File not found: {path}[/red]")
            continue

        try:
            with open(path, 'r', encoding='utf-8') as f:
                content = f.read()

            file_name = path.name
            file_type = path.suffix.lower()[1:]  # Remove the dot from extension
            content_message = f"Content from {file_name} ({file_type}):\n{content}\n"
            # Add to conversation history as assistant message to provide context
            conversation_history.append({"role": "assistant", "content": content_message})
            console.print(f"[green]Added {file_name} to the conversation context.[/green]")
        except Exception as e:
            console.print(f"[red]Error reading file {path}: {str(e)}[/red]")

    prompt_next_request()

def prompt_next_request():
    """
    Prompts the user for the next request after adding files.
    """
    console.print("\n[bold yellow]Files have been added to the context.[/bold yellow]")

def parse_and_create_files(ai_response):
    created_files = []
    script_dir = Path(__file__).parent.resolve()
    # Pass the correct tags for file creation
    yaml_content = extract_yaml_blocks(ai_response, "<tags>", "</tags>")

    if not yaml_content:
        # logging.warning("No YAML content found in AI response.")
        return created_files

    try:
        parsed_yaml = load_yaml(yaml_content)
    except yaml.YAMLError as e:
        console.print(f"[red]YAML parsing error: {str(e)}[/red]")
        logging.error(f"YAML parsing error: {str(e)}")
        return created_files

    if not isinstance(parsed_yaml, dict):
        logging.error("Parsed YAML is not a dictionary.")
        return created_files

    for filename, file_info in parsed_yaml.items():
        if not isinstance(file_info, dict):
            logging.warning(f"File info for {filename} is not a dictionary. Skipping.")
            continue

        if not all(k in file_info for k in ("extension", "code")):
            logging.warning(f"Missing keys in file info for {filename}. Skipping.")
            continue

        try:
            code = decode_code_field(file_info['code'], filename)
            file_path = script_dir / filename
            file_path.parent.mkdir(parents=True, exist_ok=True)
            file_path.write_text(code, encoding='utf-8')
            created_files.append(str(file_path))
            logging.info(f"Created file: {file_path}")
        except Exception as e:
            console.print(f"[red]Failed to create file {filename}: {str(e)}[/red]")
            logging.error(f"Failed to create file {filename}: {str(e)}")

    return created_files

def extract_yaml_blocks(response_text, start_tag, end_tag):
    """
    Extracts YAML content enclosed within specified start and end tags from the response text.

    Args:
        response_text (str): The AI's response containing YAML.
        start_tag (str): The opening tag to look for.
        end_tag (str): The closing tag to look for.

    Returns:
        str: The extracted YAML content or None if not found.
    """
    start_index = response_text.find(start_tag)
    end_index = response_text.find(end_tag)

    if start_index != -1 and end_index != -1:
        start_index += len(start_tag)
        yaml_content = response_text[start_index:end_index].strip()
        return yaml_content
    return None

def animate_thinking(stop_event):
    spinner = itertools.cycle(['⠋', '⠙', '⠹', '⠸', '⠼', '⠴', '⠦', '⠧', '⠇', '⠏'])
    while not stop_event.is_set():
        sys.stdout.write(f"\r\033[36mThinking {next(spinner)}\033[0m")
        sys.stdout.flush()
        time.sleep(0.1)

def calculate_cost(prompt_tokens, completion_tokens, model):
    model_costs = {
        "o1-mini": {"input": 3.0, "output": 12.0},
        "o1-preview": {"input": 15.0, "output": 60.0},
        # Add more models and their costs here if needed
    }
    if model not in model_costs:
        input_cost_per_million = 0
        output_cost_per_million = 0
    else:
        input_cost_per_million = model_costs[model]["input"]
        output_cost_per_million = model_costs[model]["output"]

    input_cost = (prompt_tokens / 1_000_000) * input_cost_per_million
    output_cost = (completion_tokens / 1_000_000) * output_cost_per_million
    total_cost = input_cost + output_cost
    return input_cost, output_cost, total_cost


def compute_diff(original_content: str, updated_content: str, filename: str) -> str:
    """
    Computes the unified diff between the original and updated file contents.

    Args:
        original_content (str): The original file content.
        updated_content (str): The updated file content.
        filename (str): The name of the file being compared.

    Returns:
        str: The formatted unified diff string.
    """
    original_lines = original_content.splitlines(keepends=True)
    updated_lines = updated_content.splitlines(keepends=True)
    diff = difflib.unified_diff(
        original_lines,
        updated_lines,
        fromfile=f"Original {filename}",
        tofile=f"Updated {filename}",
        lineterm=''
    )
    return ''.join(diff)

def generate_diffs(ai_response: str, files_to_edit: List[str]) -> str:
    # Pass the correct tags for edit operations
    yaml_content = extract_yaml_blocks(ai_response, "<edit>", "</edit>")
    if not yaml_content:
        console.print(f"[red]No edit instructions found in AI response.[/red]")
        logging.error("No edit instructions found in AI response.")
        return ""

    edit_instructions = load_yaml(yaml_content)
    diffs = ""

    for file_path in files_to_edit:
        filename = Path(file_path).name
        if filename not in edit_instructions:
            console.print(f"[red]No edit instructions found for {filename}.[/red]")
            logging.error(f"No edit instructions found for {filename}.")
            continue

        edits = edit_instructions[filename]
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                original_content = f.read()
        except Exception as e:
            console.print(f"[red]Failed to read original content of {file_path}: {str(e)}[/red]")
            logging.error(f"Failed to read original content of {file_path}: {str(e)}")
            continue

        # Apply edits to get the updated content
        updated_content, failed_edits = apply_edits_to_content(original_content, edits)

        if failed_edits:
            console.print(f"[yellow]Warning: Some edits failed for {file_path}.[/yellow]")
            for failed_edit in failed_edits:
                console.print(f"[yellow]Failed edit: {failed_edit}[/yellow]")

        # Compute the diff between original and updated content
        diff = compute_diff(original_content, updated_content, filename)
        diffs += diff + "\n"

    return diffs

def display_ai_response(response, block_type="tags"):
    start_tag = "<tags>" if block_type == "tags" else "<edit>"
    end_tag = "</tags>" if block_type == "tags" else "</edit>"

    yaml_content = extract_yaml_blocks(response, start_tag, end_tag)

    if yaml_content:
        try:
            parsed_yaml = load_yaml(yaml_content)
            if block_type == "tags":
                # File creation response
                for filename, file_info in parsed_yaml.items():
                    console.print(f"\n[bold blue]Content for {filename}:[/bold blue]")
                    code = decode_code_field(file_info['code'], filename)
                    code_block = f"```{file_info['extension']}\n{code}\n```"
                    console.print(Markdown(code_block))
            elif block_type == "edit":
                # File editing response
                for filename, edits in parsed_yaml.items():
                    console.print(f"\n[bold blue]Edits for {filename}:[/bold blue]")
                    for edit in edits:
                        find_code = edit.get('find', '').strip()
                        replace_code = edit.get('replace', '').strip()

                        console.print("[bold green]<find>[/bold green]")
                        console.print(Markdown(f"```python\n{find_code}\n```"))
                        console.print("[bold green]</find>[/bold green]")
                        console.print("[bold red]<replace>[/bold red]")
                        console.print(Markdown(f"```python\n{replace_code}\n```"))
                        console.print("[bold red]</replace>[/bold red]")
        except yaml.YAMLError as e:
            console.print(f"[red]Error parsing YAML content: {str(e)}[/red]")
            logging.error(f"Error parsing YAML content: {str(e)}")
    else:
        console.print(Markdown(response))

def get_planning_response(planning_request):
    global total_input_cost, total_output_cost, total_chat_cost

    planning_prompt = f"{planningprompt} {planning_request}"
    planning_messages = [{"role": "user", "content": planning_prompt}]

    stop_event = threading.Event()
    animation_thread = threading.Thread(target=animate_thinking, args=(stop_event,))
    animation_thread.start()

    try:
        response = client.chat.completions.create(
            model=PlanningModel,
            messages=planning_messages,
            max_completion_tokens=32768
        )

        stop_event.set()
        animation_thread.join()
        sys.stdout.write("\r" + " " * 50 + "\r")  # Clear the "Thinking" line
        sys.stdout.flush()

        planning_response = response.choices[0].message.content
        console.print("\nPlanning Assistant:")
        display_ai_response(planning_response)

        usage = response.usage
        console.print(f"[cyan]Planning - Prompt: {usage.prompt_tokens}, "
                      f"Completion: {usage.completion_tokens}, "
                      f"Total: {usage.total_tokens}[/cyan]")

        input_cost, output_cost, message_cost = calculate_cost(usage.prompt_tokens, usage.completion_tokens, PlanningModel)

        # Accumulate costs
        total_input_cost += input_cost
        total_output_cost += output_cost
        total_chat_cost += message_cost

        console.print(f"[cyan]Planning Cost: Input: ${input_cost:.4f}, "
                      f"Output: ${output_cost:.4f}, "
                      f"Total: ${message_cost:.4f}[/cyan]")
        console.print(f"[cyan]Total Chat Cost So Far: ${total_chat_cost:.4f}[/cyan]")

        return planning_response

    except Exception as e:
        stop_event.set()
        console.print(f"[red]An error occurred during planning: {str(e)}[/red]")
        logging.error(f"An error occurred during planning: {str(e)}")
        return ""

def save_conversation(conversation_history: list):
    from datetime import datetime
    # Generate filename based on current time
    now = datetime.now()
    filename = f"o1-{now.hour}-{now.minute}.md"
    file_path = Path.cwd() / filename

    try:
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write("# Chat Conversation History\n\n")
            for message in conversation_history:
                role = message.get('role', '').capitalize()
                content = message.get('content', '')
                if role == "User" and content.startswith(preprompt):
                    # Skip the preprompt and only save the actual user request
                    content = content.split("User:", 1)[-1].strip()
                f.write(f"**{role}:** {content}\n\n")
        console.print(f"[green]Conversation history saved to {file_path}[/green]")
    except Exception as e:
        console.print(f"[red]Failed to save conversation: {str(e)}[/red]")
        logging.error(f"Failed to save conversation: {str(e)}")

def parse_add_or_edit_command(command_type, args, conversation_history):
    """
    Handles both add and edit commands by adding files to context.

    Args:
        command_type (str): 'add' or 'edit'
        args (List[str]): List of file paths
        conversation_history (list): The current conversation history
    """
    global add_edit_flag
    add_edit_flag = command_type  # Keep track of whether it's add or edit
    add_files_to_context(args, conversation_history)

def prompt_edit_instructions():
    """
    Prompts the user to enter edit instructions.
    """
    console.print("\n[bold yellow]Please enter your edit instructions for the added files:[/bold yellow]")

def get_edit_response(edit_request, conversation_history, files_to_edit, max_retries=3):
    global total_input_cost, total_output_cost, total_chat_cost

    formatted_editor_prompt = get_formatted_editor_prompt(files_to_edit)

    edit_messages = []
    edit_messages.append({"role": "user", "content": formatted_editor_prompt})

    for message in conversation_history:
        if message.get('content', '').startswith('Content from'):
            edit_messages.append(message)

    edit_messages.append({"role": "user", "content": edit_request})

    retry_count = 0

    while retry_count < max_retries:
        stop_event = threading.Event()
        animation_thread = threading.Thread(target=animate_thinking, args=(stop_event,))
        animation_thread.start()

        try:
            response = client.chat.completions.create(
                model=EditorModel,
                messages=edit_messages,
                max_completion_tokens=32768
            )

            raw_ai_response = response.choices[0].message.content
            logging.debug(f"AI Edit Response: {raw_ai_response}")

            stop_event.set()
            animation_thread.join()
            sys.stdout.write("\r" + " " * 50 + "\r")
            sys.stdout.flush()

            edit_response = raw_ai_response

            valid, message = is_valid_yaml_content_response(edit_response, block_type="edit")
            if valid:
                console.print("\nEdit Instructions:")
                display_ai_response(edit_response, block_type="edit")

                usage = response.usage
                console.print(f"[cyan]Edit - Prompt: {usage.prompt_tokens}, "
                              f"Completion: {usage.completion_tokens}, "
                              f"Total: {usage.total_tokens}[/cyan]")

                input_cost, output_cost, message_cost = calculate_cost(usage.prompt_tokens, usage.completion_tokens, EditorModel)

                total_input_cost += input_cost
                total_output_cost += output_cost
                total_chat_cost += message_cost

                console.print(f"[cyan]Edit Cost: Input: ${input_cost:.4f}, "
                              f"Output: ${output_cost:.4f}, "
                              f"Total: ${message_cost:.4f}[/cyan]")
                console.print(f"[cyan]Total Chat Cost So Far: ${total_chat_cost:.4f}[/cyan]")

                # Apply edits and collect failed edits
                failed_edits = apply_file_updates(edit_response, conversation_history, files_to_edit, block_type="edit")

                if failed_edits:
                    # Report failed edits
                    console.print(f"[yellow]Some edits could not be applied:[/yellow]")
                    for filename, edits in failed_edits.items():
                        console.print(f"[yellow]In file {filename}:[/yellow]")
                        for edit in edits:
                            console.print(f"[yellow]- {edit}[/yellow]")

                    # Prepare error message for the assistant
                    error_message = "The following edits could not be applied because the 'find' patterns did not match any content in the files:\n"
                    for filename, edits in failed_edits.items():
                        error_message += f"In file {filename}:\n"
                        for edit in edits:
                            error_message += f"- {edit}\n"
                    error_message += "Please correct the 'find' patterns to match the content exactly."

                    # Update the messages for the next retry
                    edit_messages.append({"role": "assistant", "content": edit_response})
                    edit_messages.append({"role": "user", "content": error_message})

                    console.print(f"[yellow]Retrying with corrected edit instructions from the assistant.[/yellow]")
                    retry_count += 1
                else:
                    # All edits applied successfully
                    console.print("[green]All edits applied successfully.[/green]")
                    break  # Exit the retry loop
            else:
                console.print(f"[red]Validation Error: {message}[/red]")
                error_message = f"The previous response did not meet the required format because: {message}. Please provide the updated edit instructions again, strictly adhering to the format specified."
                edit_messages.append({"role": "assistant", "content": edit_response})
                edit_messages.append({"role": "user", "content": error_message})
                console.print(f"[yellow]Invalid response format. Requesting correction from the assistant.[/yellow]")
                retry_count += 1

            if retry_count >= max_retries:
                console.print("[red]Maximum number of retries reached. Some edits could not be applied.[/red]")
                break  # Ensure the loop exits after maximum retries

        except Exception as e:
            stop_event.set()
            console.print(f"[red]An error occurred during editing: {str(e)}[/red]")
            logging.error(f"An error occurred during editing: {str(e)}")
            retry_count += 1
            if retry_count >= max_retries:
                console.print("[red]Maximum number of retries reached due to errors. Exiting edit process.[/red]")
                break

    return  # Ensure the function exits properly

def is_valid_yaml_content_response(response_text, block_type="tags"):
    start_tag = "<tags>" if block_type == "tags" else "<edit>"
    end_tag = "</tags>" if block_type == "tags" else "</edit>"

    yaml_content = extract_yaml_blocks(response_text, start_tag, end_tag)
    if not yaml_content:
        return False, "No YAML content found."

    try:
        parsed_yaml = load_yaml(yaml_content)
    except yaml.YAMLError as e:
        error_detail = str(e)
        return False, f"YAML parsing error: {error_detail}"

    if not isinstance(parsed_yaml, dict):
        return False, "Parsed YAML is not a dictionary."

    if block_type == "tags":
        # Validation for file creation
        for filename, file_info in parsed_yaml.items():
            if not isinstance(file_info, dict):
                return False, f"File info for '{filename}' is not a dictionary."
            if 'extension' not in file_info or 'code' not in file_info:
                return False, f"Missing 'extension' or 'code' in file info for '{filename}'."
    elif block_type == "edit":
        # Validation for file editing
        for filename, edits in parsed_yaml.items():
            if not isinstance(edits, list):
                return False, f"Edits for '{filename}' are not a list."
            for index, edit in enumerate(edits):
                if not isinstance(edit, dict):
                    return False, f"Edit #{index + 1} in '{filename}' is not a dictionary."
                if 'find' not in edit or 'replace' not in edit:
                    return False, f"Missing 'find' or 'replace' in edit #{index + 1} for '{filename}'."
    return True, "Valid YAML content response."

def clean_diff(diff: str) -> str:
    """
    Removes empty '+' or '-' lines from the diff to prevent patch errors.

    Args:
        diff (str): The raw diff string.

    Returns:
        str: The cleaned diff string.
    """
    cleaned_lines = []
    for line in diff.split('\n'):
        if (line.startswith('+') and line.strip() == '+') or (line.startswith('-') and line.strip() == '-'):
            continue  # Skip empty '+' or '-' lines
        cleaned_lines.append(line)
    return '\n'.join(cleaned_lines)

def extract_between_tags(text, start_tag, end_tag):
    start_index = text.find(start_tag)
    end_index = text.find(end_tag)
    if start_index != -1 and end_index != -1:
        start_index += len(start_tag)
        return text[start_index:end_index]
    else:
        return None

def apply_edits_to_content(content: str, edits: list, threshold=60) -> tuple:
    """
    Applies a list of edits to the given content using exact and fuzzy matching.

    Args:
        content (str): The original file content.
        edits (list): A list of edits, each containing 'find' and 'replace' strings.
        threshold (int): Similarity threshold (0-100) for fuzzy matching.

    Returns:
        tuple: The updated content and a list of failed edits.
    """
    failed_edits = []
    for edit in edits:
        find_str = edit.get('find', '').strip()
        replace_str = edit.get('replace', '').strip()

        if not find_str or not replace_str:
            console.print(f"[red]Invalid edit: 'find' or 'replace' is empty. Skipping this edit.[/red]")
            failed_edits.append(edit)
            continue

        # Decode and normalize whitespace
        find_str = decode_code_field(find_str, 'find').strip()
        replace_str = decode_code_field(replace_str, 'replace').strip()

        # Debugging: Log the find and replace strings
        logging.debug(f"Applying edit:\nFind:\n{find_str}\nReplace:\n{replace_str}\n")

        # First attempt exact matching
        if find_str in content:
            content = content.replace(find_str, replace_str, 1)
            console.print(f"[green]Successfully applied exact match edit for '{find_str}'.[/green]")
            logging.info(f"Successfully replaced exact match '{find_str}' with '{replace_str}'.")
            continue

        # If exact match fails, attempt fuzzy matching
        lines = content.splitlines()
        match_found = False
        for i, line in enumerate(lines):
            similarity = fuzz.ratio(find_str, line.strip())
            if similarity >= threshold:
                lines[i] = replace_str
                match_found = True
                console.print(f"[green]Successfully applied fuzzy match edit for line {i+1} (Similarity: {similarity}%).[/green]")
                logging.info(f"Successfully replaced line {i+1}: '{line.strip()}' with '{replace_str}' (Similarity: {similarity}%).")
                break

        if not match_found:
            console.print(f"[yellow]Warning: 'find' string '{find_str}' not found in content.[/yellow]")
            logging.warning(f"'find' string '{find_str}' not found in content.")
            failed_edits.append(edit)
            continue

        # Reconstruct the content
        content = '\n'.join(lines)

    return content, failed_edits

def apply_edit_instructions(file_path: Path, edits: list):
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()

        # Apply edits to get the updated content and failed edits
        updated_content, failed_edits = apply_edits_to_content(content, edits)

        if failed_edits:
            console.print(f"[yellow]Warning: Some edits failed for {file_path}.[/yellow]")
            for failed_edit in failed_edits:
                console.print(f"[yellow]Failed edit: {failed_edit}[/yellow]")

        # Backup the original file
        backup_file(file_path)

        # Write the updated content to the file
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(updated_content)

        console.print(f"[green]Successfully applied edits to {file_path}.[/green]")
        logging.info(f"Successfully applied edits to {file_path}.")

    except Exception as e:
        console.print(f"[red]Failed to apply edits to {file_path}: {str(e)}[/red]")
        logging.error(f"Failed to apply edits to {file_path}: {str(e)}")

def apply_file_updates(ai_response, conversation_history, files_to_edit, block_type="edit"):
    """
    Parses the AI's edit instructions and applies them to the specified files.
    """
    failed_edits = {}

    try:
        start_tag = "<tags>" if block_type == "tags" else "<edit>"
        end_tag = "</tags>" if block_type == "tags" else "</edit>"
        yaml_content = extract_yaml_blocks(ai_response, start_tag, end_tag)

        if not yaml_content:
            console.print(f"[red]No YAML content found in AI response.[/red]")
            return failed_edits

        edit_instructions = yaml.safe_load(yaml_content)

        # Map filenames to their full paths
        filename_to_path = {Path(f).name: Path(f) for f in files_to_edit}

        # Now apply the edits after user confirmation
        if prompt_user_confirmation():
            for filename, edits in edit_instructions.items():
                if filename not in filename_to_path:
                    console.print(f"[red]Filename '{filename}' is not in the list of files to edit. Skipping.[/red]")
                    continue

                file_path = filename_to_path[filename]

                if not file_path.is_file():
                    console.print(f"[red]File not found for editing: {file_path}[/red]")
                    continue

                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()

                    # Apply edits to get the updated content and failed edits
                    updated_content, file_failed_edits = apply_edits_to_content(content, edits)

                    if file_failed_edits:
                        failed_edits[filename] = file_failed_edits

                    # Backup the original file
                    backup_file(file_path)

                    # Write the updated content to the file
                    with open(file_path, 'w', encoding='utf-8') as f:
                        f.write(updated_content)

                    console.print(f"[green]Successfully applied edits to {file_path}.[/green]")

                except Exception as e:
                    console.print(f"[red]Failed to apply edits to {file_path}: {str(e)}[/red]")

        else:
            console.print("[yellow]Edits were not applied as per user's decision.[/yellow]")

    except Exception as e:
        console.print(f"[red]An unexpected error occurred: {str(e)}[/red]")

    return failed_edits

def backup_file(file_path: Path):
    """
    Creates a backup of the specified file.
    """
    backup_path = file_path.with_suffix(file_path.suffix + '.bak')
    shutil.copy(file_path, backup_path)
    # console.print(f"[yellow]Backup created for {file_path} at {backup_path}.[/yellow]")
    return backup_path

def prompt_user_confirmation():
    """
    Prompts the user to confirm whether to apply the edits.
    """
    while True:
        confirmation = input("\nDo you want to apply these edits? (yes/no): ").strip().lower()
        if confirmation in ['yes', 'y']:
            return True
        elif confirmation in ['no', 'n']:
            return False
        else:
            console.print("[red]Please respond with 'yes' or 'no'.[/red]")

def chat_with_ai():
    global total_input_cost, total_output_cost, total_chat_cost

    from rich.table import Table

    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("Command", style="cyan", no_wrap=True)
    table.add_column("Description", style="dim")

    table.add_row("exit", "Quit the program")
    table.add_row("reset", "Clear conversation history")
    table.add_row("/add file1 [file2 ...]", "Add files to the conversation")
    table.add_row("/edit file1 [file2 ...]", "Edit specified files")
    table.add_row("save", "Save the conversation")
    table.add_row("planning", "Use the planning feature")

    console.print(table)
    console.print("o1-engineer is ready. Please enter a command or start your conversation.", style="bold green")
    conversation_history = []
    
    # Initialize state variables
    awaiting_confirmation = False
    current_plan = ""

    session = PromptSession(
        style=prompt_style,
        history=InMemoryHistory()
    )

    while True:
        try:
            user_input = session.prompt("You: ")
        except KeyboardInterrupt:
            continue
        except EOFError:
            break

        # If awaiting confirmation, handle 'yes/no' input
        if awaiting_confirmation:
            use_plan = user_input.strip().lower()
            if use_plan in ['yes', 'y']:
                # Prepend the YAML preprompt to the current plan
                full_input = f"{preprompt}\n\nUser: {current_plan}"
                conversation_history = conversation_history[-MAX_CONVERSATION_HISTORY:]

                # Send the updated conversation history to the AI
                stop_event = threading.Event()
                animation_thread = threading.Thread(target=animate_thinking, args=(stop_event,))
                animation_thread.start()

                try:
                    response = client.chat.completions.create(
                        model=MainModel,
                        messages=conversation_history,
                        max_completion_tokens=32768  # Adjust as needed
                    )

                    # Log the raw AI response
                    raw_ai_response = response.choices[0].message.content
                    logging.debug(f"AI Response (Plan Execution): {raw_ai_response}")

                    stop_event.set()
                    animation_thread.join()
                    sys.stdout.write("\r" + " " * 50 + "\r")  # Clear the "Thinking" line
                    sys.stdout.flush()

                    ai_response = raw_ai_response
                    console.print("\no1-mini:")
                    display_ai_response(ai_response)

                    created_files = parse_and_create_files(ai_response)
                    if created_files:
                        console.print(f"[green]Created files: {', '.join(created_files)}[/green]")

                    conversation_history = conversation_history[-MAX_CONVERSATION_HISTORY:]

                    usage = response.usage
                    reasoning_tokens = usage.completion_tokens_details.reasoning_tokens if hasattr(usage, 'completion_tokens_details') else "N/A"
                    console.print(f"[cyan]Prompt: {usage.prompt_tokens}, "
                                  f"Completion: {usage.completion_tokens}, "
                                  f"[magenta]Reasoning: {reasoning_tokens}[/magenta], "
                                  f"Total: {usage.total_tokens}[/cyan]")

                    input_cost, output_cost, message_cost = calculate_cost(usage.prompt_tokens, usage.completion_tokens, MainModel)

                    # Accumulate costs
                    total_input_cost += input_cost
                    total_output_cost += output_cost
                    total_chat_cost += message_cost

                    console.print(f"[cyan]Cost: Input: ${input_cost:.4f}, "
                                  f"Output: ${output_cost:.4f}, "
                                  f"Total: ${message_cost:.4f}[/cyan]")
                    console.print(f"[cyan]Total Chat Cost So Far: ${total_chat_cost:.4f}[/cyan]")

                    if response.choices[0].finish_reason == 'length':
                        console.print("[yellow]Warning: The response was cut off due to token limit. "
                                      "Consider increasing max_completion_tokens if this is an issue.[/yellow]")

                except Exception as e:
                    stop_event.set()
                    console.print(f"[red]An error occurred: {str(e)}[/red]")

            elif use_plan in ['no', 'n']:
                console.print("[yellow]Plan was not used. Continuing without applying the plan.[/yellow]")
            else:
                console.print("[red]Please respond with 'yes' or 'no'.[/red]")

            # Reset the confirmation state
            awaiting_confirmation = False
            current_plan = ""
            continue

        # Parse the user input for commands
        command, args = parse_command(user_input)

        if command == 'exit':
            console.print(f"\n[bold green]Total Chat Cost: Input: ${total_input_cost:.4f}, "
                          f"Output: ${total_output_cost:.4f}, "
                          f"Total: ${total_chat_cost:.4f}[/bold green]")
            break
        elif command == 'reset':
            conversation_history.clear()
            console.print("[yellow]Conversation history has been reset.[/yellow]")
            continue
        elif command == 'save':
            save_conversation(conversation_history)
            continue
        elif command in ['add', 'edit']:
            parse_add_or_edit_command(command, args, conversation_history)
            if command == 'edit':
                prompt_edit_instructions()
                edit_instructions = session.prompt()
                get_edit_response(edit_instructions, conversation_history, args)
            continue
        elif command in ['invalid_add', 'invalid_edit']:
            console.print(f"[red]Invalid /{command.split('_')[1]} command format. Use: /{command.split('_')[1]} file1 [file2 ...][/red]")
            continue
        elif command == 'planning' or user_input.lower() == 'planning':
            planning_request = session.prompt("Enter your planning request: ")
            planning_response = get_planning_response(planning_request)
            console.print("\nDo you want to use this plan? (yes/no): ", style="bold yellow")
            # Set the state to awaiting confirmation
            awaiting_confirmation = True
            current_plan = planning_response
            continue

        # Handle normal user input (file creation)
        if command is None:
            full_input = f"{preprompt}\n\nUser: {user_input}"
            conversation_history.append({"role": "user", "content": full_input})

            stop_event = threading.Event()
            animation_thread = threading.Thread(target=animate_thinking, args=(stop_event,))
            animation_thread.start()

            try:
                response = client.chat.completions.create(
                    model=MainModel,
                    messages=conversation_history,
                    max_completion_tokens=32768  # Default max for o1-mini
                )

                # Log the raw AI response
                raw_ai_response = response.choices[0].message.content
                logging.debug(f"AI Response: {raw_ai_response}")

                stop_event.set()
                animation_thread.join()
                sys.stdout.write("\r" + " " * 50 + "\r")  # Clear the "Thinking" line
                sys.stdout.flush()

                ai_response = raw_ai_response
                console.print("\no1-mini:")
                display_ai_response(ai_response)

                created_files = parse_and_create_files(ai_response)
                if created_files:
                    console.print(f"[green]Created files: {', '.join(created_files)}[/green]")

                conversation_history.append({"role": "assistant", "content": ai_response})

                usage = response.usage
                reasoning_tokens = usage.completion_tokens_details.reasoning_tokens if hasattr(usage, 'completion_tokens_details') else "N/A"
                console.print(f"[cyan]Prompt: {usage.prompt_tokens}, "
                              f"Completion: {usage.completion_tokens}, "
                              f"[magenta]Reasoning: {reasoning_tokens}[/magenta], "
                              f"Total: {usage.total_tokens}[/cyan]")

                input_cost, output_cost, message_cost = calculate_cost(usage.prompt_tokens, usage.completion_tokens, MainModel)

                # Accumulate costs
                total_input_cost += input_cost
                total_output_cost += output_cost
                total_chat_cost += message_cost

                console.print(f"[cyan]Cost: Input: ${input_cost:.4f}, "
                              f"Output: ${output_cost:.4f}, "
                              f"Total: ${message_cost:.4f}[/cyan]")
                console.print(f"[cyan]Total Chat Cost So Far: ${total_chat_cost:.4f}[/cyan]")

                if response.choices[0].finish_reason == 'length':
                    console.print("[yellow]Warning: The response was cut off due to token limit. "
                                  "Consider increasing max_completion_tokens if this is an issue.[/yellow]")

            except Exception as e:
                stop_event.set()
                console.print(f"[red]An error occurred: {str(e)}[/red]")

    # After the while loop ends
    console.print(f"\n[bold green]Total Chat Cost: Input: ${total_input_cost:.4f}, "
                  f"Output: ${total_output_cost:.4f}, "
                  f"Total: ${total_chat_cost:.4f}[/bold green]")

if __name__ == "__main__":
    chat_with_ai()
