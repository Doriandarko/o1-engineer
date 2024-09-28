import os
import json
import subprocess
from dotenv import load_dotenv
from openai import OpenAI
from termcolor import colored
from prompt_toolkit import prompt
from prompt_toolkit.styles import Style
from rich import print as rprint
from rich.markdown import Markdown
from prompt_toolkit.completion import WordCompleter

load_dotenv()

# Initialize OpenAI client
client = OpenAI(api_key="YOUR KEY")

# System prompt to be included with edit requests
SYSTEM_PROMPT = """You are an advanced AI assistant designed to analyze and modify large text-based files based on user instructions. Your primary objective is to identify the specific sections within one or multiple provided files that need to be updated or changed, regardless of the file size. 

When given a user request and one or more files, perform the following steps:

1. Understand the User Request: Carefully interpret what the user wants to achieve with the modification.
2. Analyze the Files: Efficiently parse the provided file(s) to locate the exact parts that are relevant to the user's request. Given that files can be large, ensure that your analysis is optimized for performance and does not require loading the entire file into memory if not necessary.
3. Generate Modification Instructions: For each identified section that requires changes, provide clear and precise instructions in a structured JSON format. The JSON should include:
   - file_path: The path to the file to be modified.
   - modifications: A list of modifications, each containing:
     - start_line: The line number where the modification starts.
     - end_line: The line number where the modification ends.
     - original_text: The original text that will be replaced.
     - new_text: The text that will replace the original.
4. Ensure Clarity and Precision: Your instructions must be unambiguous to allow accurate implementation without additional clarification.

IMPORTANT: Your response must be ONLY valid JSON, with no additional text before or after. Do not use markdown formatting or code blocks. Ensure proper JSON formatting, including correct use of quotes, commas, and brackets. The output should start with '{' and end with '}' with no other characters outside of this.

Example of the expected JSON format:
{
  "modifications": [
    {
      "file_path": "path/to/file1.txt",
      "modifications": [
        {
          "start_line": 150,
          "end_line": 152,
          "original_text": "Old configuration setting.",
          "new_text": "New configuration setting."
        },
        {
          "start_line": 300,
          "end_line": 300,
          "original_text": "Deprecated function call.",
          "new_text": "Updated function call."
        }
      ]
    },
    {
      "file_path": "path/to/file2.txt",
      "modifications": [
        {
          "start_line": 45,
          "end_line": 47,
          "original_text": "Obsolete code block.",
          "new_text": "Refactored code block."
        }
      ]
    }
  ]
}

Ensure that the JSON output strictly follows this structure to facilitate seamless integration with the modification script."""

CREATE_SYSTEM_PROMPT = """You are an advanced AI assistant designed to create files and folders based on user instructions. Your primary objective is to generate a JSON structure that describes the files and folders to be created.

When given a user request, perform the following steps:

1. Understand the User Request: Carefully interpret what the user wants to create.
2. Generate Creation Instructions: Provide clear and precise instructions in a structured JSON format. The JSON should include:
   - creations: A list of items to create, each containing:
     - type: Either "file" or "folder"
     - name: The name of the file or folder
     - path: The path where the file or folder should be created (use "." for current directory)
     - content: For files, the content to be written (leave empty for folders)

IMPORTANT: Your response must be ONLY valid JSON, with no additional text before or after. Do not use markdown formatting or code blocks. Ensure proper JSON formatting, including correct use of quotes, commas, and brackets. The output should start with '{' and end with '}' with no other characters outside of this.

Example of the expected JSON format:
{
  "creations": [
    {
      "type": "folder",
      "name": "new_folder",
      "path": "."
    },
    {
      "type": "file",
      "name": "new_file.txt",
      "path": "new_folder",
      "content": "This is the content of the new file."
    }
  ]
}

Ensure that the JSON output strictly follows this structure to facilitate seamless creation of files and folders."""

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
        return

    if is_binary_file(file_path):
        print(colored(f"Error: {file_path} appears to be a binary file and cannot be added.", "red"))
        return

    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            content = file.read()
            added_files[file_path] = content
            print(colored(f"Added {file_path} to the chat context.", "green"))
    except Exception as e:
        print(colored(f"Error reading file {file_path}: {e}", "red"))

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
                "content": SYSTEM_PROMPT + "\n\nUser request: " + user_message if is_edit_request else user_message
            }
        ]
        
        if is_edit_request and retry_count == 0:
            print(colored("Analyzing files and generating modifications...", "magenta"))
        elif not is_edit_request:
            print(colored("o1 engineer is thinking...", "magenta"))
        response = client.chat.completions.create(
            model="o1-mini",
            messages=messages,
            max_completion_tokens=60000  
        )
        return response.choices[0].message.content
    except Exception as e:
        print(colored(f"Error while communicating with OpenAI: {e}", "red"))
        return None

def lint_file(file_path):
    _, file_extension = os.path.splitext(file_path)
    
    if file_extension == '.py':
        try:
            subprocess.run(['pylint', file_path], check=True)
            print(colored(f"Linting completed for {file_path}", "green"))
        except subprocess.CalledProcessError:
            print(colored(f"Linting found issues in {file_path}", "yellow"))
        except FileNotFoundError:
            print(colored("pylint is not installed. Please install it to enable Python linting.", "yellow"))
    
    elif file_extension in ['.js', '.jsx', '.ts', '.tsx']:
        try:
            subprocess.run(['eslint', file_path], check=True)
            print(colored(f"Linting completed for {file_path}", "green"))
        except subprocess.CalledProcessError:
            print(colored(f"Linting found issues in {file_path}", "yellow"))
        except FileNotFoundError:
            print(colored("eslint is not installed. Please install it to enable JavaScript/TypeScript linting.", "yellow"))
    
    else:
        print(colored(f"No linter configured for files with extension {file_extension}", "yellow"))

def apply_modifications(modifications_json, retry_count=0):
    try:
        json_start = modifications_json.find('{')
        json_end = modifications_json.rfind('}') + 1
        if json_start != -1 and json_end != -1:
            modifications_json = modifications_json[json_start:json_end]
        
        try:
            modifications_data = json.loads(modifications_json)
        except json.JSONDecodeError as e:
            if retry_count < 3:
                print(colored(f"Failed to parse JSON. Retrying... (Attempt {retry_count + 1})", "yellow"))
                error_message = f"The JSON parsing failed. Please generate a valid JSON structure. Error: {str(e)}"
                new_json = chat_with_ai(error_message, is_edit_request=True)
                return apply_modifications(new_json, retry_count + 1)
            else:
                raise ValueError("Unable to extract valid JSON from the response after multiple attempts")

        print("Successfully parsed JSON:")

        if "modifications" not in modifications_data:
            if retry_count < 3:
                print(colored(f"Invalid JSON structure. Retrying... (Attempt {retry_count + 1})", "yellow"))
                error_message = "The JSON structure is invalid. Please ensure it contains a 'modifications' key at the top level."
                new_json = chat_with_ai(error_message, is_edit_request=True)
                return apply_modifications(new_json, retry_count + 1)
            else:
                print(colored("Error: Invalid JSON structure. 'modifications' key not found after multiple attempts.", "red"))
                return False

        successful_edits = 0
        total_edits = sum(len(file_mod.get("modifications", [])) for file_mod in modifications_data["modifications"])

        for file_mod in modifications_data["modifications"]:
            file_path = file_mod.get("file_path")
            mods = file_mod.get("modifications", [])

            if not file_path:
                print(colored("Error: file_path not specified for a modification.", "red"))
                continue

            if not os.path.isfile(file_path):
                print(colored(f"File not found: {file_path}", "red"))
                continue

            temp_file_path = f"{file_path}.temp"

            try:
                with open(file_path, 'r', encoding='utf-8') as original_file, \
                     open(temp_file_path, 'w', encoding='utf-8') as temp_file:

                    lines = original_file.readlines()
                    for mod in mods:
                        start_line = mod["start_line"]
                        end_line = mod["end_line"]
                        original_text = ''.join(lines[start_line-1:end_line]).strip()
                        expected_original = mod["original_text"].strip()
                        
                        if original_text != expected_original:
                            print(colored(f"Original text mismatch in {file_path} at lines {start_line}-{end_line}.", "yellow"))
                            print(colored(f"Expected: {expected_original}", "yellow"))
                            print(colored(f"Found: {original_text}", "yellow"))
                            print(colored("Attempting to apply modification anyway...", "yellow"))
                        
                        lines[start_line-1:end_line] = [mod["new_text"] + '\n']
                        successful_edits += 1

                    temp_file.writelines(lines)

                os.replace(temp_file_path, file_path)
                print(colored(f"Modifications applied to {file_path} successfully.", "green"))
                
                # Linting step after successful modification
                lint_file(file_path)

            except Exception as e:
                print(colored(f"Error processing file {file_path}: {e}", "red"))
                if os.path.exists(temp_file_path):
                    os.remove(temp_file_path)

        if successful_edits == total_edits:
            print(colored(f"All {successful_edits} edits were successful!", "green"))
            return True
        else:
            print(colored(f"{successful_edits} out of {total_edits} edits were successful.", "yellow"))
            return False

    except json.JSONDecodeError as e:
        if retry_count < 3:
            print(colored(f"Invalid JSON format. Retrying... (Attempt {retry_count + 1})", "yellow"))
            error_message = f"The JSON parsing failed. Please generate a valid JSON structure. Error: {str(e)}"
            new_json = chat_with_ai(error_message, is_edit_request=True)
            return apply_modifications(new_json, retry_count + 1)
        else:
            print(colored(f"Invalid JSON format after multiple attempts: {e}", "red"))
            print("JSON that failed to parse:")
            print(modifications_json)
            return False
    except Exception as e:
        print(colored(f"An error occurred: {e}", "red"))
        return False

def apply_creation_steps(creation_json, retry_count=0):
    try:
        json_start = creation_json.find('{')
        json_end = creation_json.rfind('}') + 1
        if json_start != -1 and json_end != -1:
            creation_json = creation_json[json_start:json_end]
        
        try:
            creation_data = json.loads(creation_json)
        except json.JSONDecodeError as e:
            if retry_count < 3:
                print(colored(f"Failed to parse JSON. Retrying... (Attempt {retry_count + 1})", "yellow"))
                error_message = f"The JSON parsing failed. Please generate a valid JSON structure. Error: {str(e)}"
                new_json = chat_with_ai(error_message, is_edit_request=True)
                return apply_creation_steps(new_json, retry_count + 1)
            else:
                raise ValueError("Unable to extract valid JSON from the response after multiple attempts")
    
        print("Successfully parsed JSON:")
    
        for item in creation_data.get("creations", []):
            creation_type = item.get("type")
            name = item.get("name")
            path = item.get("path", ".")
            full_path = os.path.join(path, name)
            
            if creation_type == "folder":
                os.makedirs(full_path, exist_ok=True)
                print(colored(f"Folder created: {full_path}", "green"))
            elif creation_type == "file":
                with open(full_path, 'w', encoding='utf-8') as f:
                    f.write(item.get("content", ""))
                print(colored(f"File created: {full_path}", "green"))
                
                # Linting step after file creation
                lint_file(full_path)
            else:
                print(colored(f"Unknown creation type: {creation_type} for {name}", "red"))
    
        return True

    except json.JSONDecodeError as e:
        if retry_count < 3:
            print(colored(f"Invalid JSON format. Retrying... (Attempt {retry_count + 1})", "yellow"))
            error_message = f"The JSON parsing failed. Please generate a valid JSON structure. Error: {str(e)}"
            new_json = chat_with_ai(error_message, is_edit_request=True)
            return apply_creation_steps(new_json, retry_count + 1)
        else:
            print(colored(f"Invalid JSON format after multiple attempts: {e}", "red"))
            print("JSON that failed to parse:")
            print(creation_json)
            return False
    except Exception as e:
        print(colored(f"An error occurred: {e}", "red"))
        return False

def main():
    print("Welcome to the AI File Editor. Type '/edit' followed by file paths to edit files, '/create' followed by creation instructions, '/add' to add files to context, or '/quit' to exit.")

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
            break

        elif user_input.startswith('/add'):
            file_paths = user_input.split()[1:]
            if not file_paths:
                print(colored("Please provide at least one file path.", "yellow"))
                continue

            for file_path in file_paths:
                add_file_to_context(file_path, added_files)

            total_size = sum(len(content) for content in added_files.values())
            if total_size > 100000:  # Warning if total content exceeds ~100KB
                print(colored("Warning: The total size of added files is large and may affect performance.", "yellow"))

        elif user_input.startswith('/edit'):
            file_paths = user_input.split()[1:]
            if not file_paths:
                print(colored("Please provide at least one file path.", "yellow"))
                continue

            file_contents = []
            for file_path in file_paths:
                try:
                    with open(file_path, 'r', encoding='utf-8') as file:
                        content = file.read()
                        file_contents.append(f"File: {file_path}\nContent:\n{content}\n")
                except Exception as e:
                    print(colored(f"Error reading file {file_path}: {e}", "red"))
                    continue

            print("Enter your edit instruction:")
            edit_instruction = prompt("Edit Instruction: ", style=style).strip()

            edit_request = f"""Please analyze and suggest modifications for the following files based on this instruction: {edit_instruction}

Remember to output ONLY valid JSON in your response, with no additional text before or after.

Files to modify:
{" ".join(file_contents)}"""

            ai_response = chat_with_ai(edit_request, is_edit_request=True, added_files=added_files)

            if ai_response:
                while True:
                    print("AI Assistant: Here are the suggested modifications:")
                    rprint(Markdown(ai_response))

                    confirm = prompt("Do you want to apply these modifications? (yes/no): ", style=style).strip().lower()
                    if confirm == 'yes':
                        success = apply_modifications(ai_response)
                        if success:
                            break
                        else:
                            retry = prompt("Modifications failed. Do you want the AI to try again? (yes/no): ", style=style).strip().lower()
                            if retry != 'yes':
                                break
                            ai_response = chat_with_ai("The previous modifications failed. Please try again with a different approach.", is_edit_request=True, added_files=added_files)
                    else:
                        print(colored("Modifications not applied.", "yellow"))
                        break

        elif user_input.startswith('/create'):
            creation_instruction = user_input[7:].strip()  # Remove '/create' and leading/trailing whitespace
            if not creation_instruction:
                print(colored("Please provide creation instructions after /create.", "yellow"))
                continue

            create_request = f"{CREATE_SYSTEM_PROMPT}\n\nUser request: {creation_instruction}"
            ai_response = chat_with_ai(create_request, is_edit_request=True, added_files=added_files)
            
            if ai_response:
                while True:
                    print("AI Assistant: Here is the suggested creation structure:")
                    rprint(Markdown(ai_response))

                    confirm = prompt("Do you want to execute these creation steps? (yes/no): ", style=style).strip().lower()
                    if confirm == 'yes':
                        success = apply_creation_steps(ai_response)
                        if success:
                            break
                        else:
                            retry = prompt("Creation failed. Do you want the AI to try again? (yes/no): ", style=style).strip().lower()
                            if retry != 'yes':
                                break
                            ai_response = chat_with_ai("The previous creation attempt failed. Please try again with a different approach.", is_edit_request=True, added_files=added_files)
                    else:
                        print(colored("Creation steps not executed.", "yellow"))
                        break

        else:
            ai_response = chat_with_ai(user_input, added_files=added_files)
            if ai_response:
                print()
                print(colored("o1 engineer:", "blue"))
                rprint(Markdown(ai_response))

if __name__ == "__main__":
    main()
