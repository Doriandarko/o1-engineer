# 🛠️ o1-engineer

![Updated Features](https://img.shields.io/badge/Features-Updated-brightgreen)

A command-line tool designed to assist developers in managing and interacting with their projects efficiently. Leveraging the power of OpenAI's API, o1-engineer provides functionalities such as code generation, file editing, project planning, and code review to streamline your development workflow.

## 🛠️ Updated Features

- **Enhanced File and Folder Management**: The `/add` and `/edit` commands now support adding and modifying both files and folders, providing greater flexibility in managing your project structure.
- **Project Planning**: Introducing the `/planning` command, which allows users to create comprehensive project plans that can be used to generate files and directories systematically.
- **Advanced Workflows**: New examples demonstrate how to integrate planning and creation commands for efficient project setup.

## ✨ Features

- **Automated Code Generation**: Generate code for your projects effortlessly.

- **File Management**: Add, edit, and manage project files directly from the command line.

- **Interactive Console**: User-friendly interface with rich text support for enhanced readability.

- **Conversation History**: Save and reset conversation histories as needed.

- **Code Review**: Analyze and review code files for quality and suggestions.

- **Enhanced File and Folder Management**: The `/add` and `/edit` commands now support adding and modifying both files and folders, providing greater flexibility in managing your project structure.

- **Project Planning**: Introducing the `/planning` command, which allows users to create comprehensive project plans that can be used to generate files and directories systematically.

## 💡 How the Script Works

1. **Initialization**: The script initializes global variables and sets up the OpenAI client using the provided API key.

2. **Handling User Commands**: It listens for user commands such as `/edit`, `/create`, `/add`, `/review`, and the new `/planning` command, processing them accordingly.

3. **Processing File and Folder Modifications**: Based on the user's instructions, the script modifies files and folders, adds new content, or creates new files and folders as needed. The `/add` and `/edit` commands have been enhanced to support both files and folders, providing greater flexibility in project management.

4. **Project Planning**: The newly introduced `/planning` command allows users to create comprehensive project plans, which the script can use to generate files and directories systematically using the `/create` command.

5. **AI-Generated Instructions**: The tool interacts with OpenAI's API to generate instructions and suggestions for code generation, editing, project planning, and reviewing.

6. **Applying Changes**: Changes are applied to the project files and folders based on the AI-generated instructions, ensuring that the project stays up-to-date and well-maintained.

7. **Managing Conversation History and Added Files**: The script manages the conversation history and keeps track of files and folders added to the context, allowing users to reset or modify the history as needed.

## 📥 Installation

### Prerequisites

- **Python**: Ensure you have Python 3.7 or higher installed. [Download Python](https://www.python.org/downloads/)

- **OpenAI API Key**: Obtain an API key from [OpenAI](https://platform.openai.com/).

### 🔧 Steps

1. **Clone the Repository**:

   ```bash
   git clone https://github.com/doriandarko/o1-engineer.git
   ```

2. **Navigate to the Project Directory**:

   ```bash
   cd o1-engineer
   ```

3. **Install Dependencies**:

   ```bash
   pip install -r requirements.txt
   ```

4. **Configure APIs**:

   Add your API key at the top of the script

   ```
   client = OpenAI(api_key="YOUR API")
   ```

## 📚 Usage

Launch the application using the following command:

```bash
python o1-eng.py
```

### 🎮 Available Commands

- `/edit`: Edit files or folders (followed by file or folder paths)

- `/create`: Create files or folders (followed by instructions)

- `/add`: Add files or folders to context (followed by file or folder paths)

- `/planning`: Plan project structure and tasks (followed by instructions)

- `/debug`: Print the last AI response

- `/reset`: Reset chat context and clear added files

- `/review`: Review and analyze code files for quality and potential improvements (followed by file or folder paths)

- `/quit`: Exit the program

### 🚀 Advanced Workflows

Here's an example workflow that demonstrates using `/planning` followed by `/create` to generate files based on the created plan:

1. **Planning the Project**:

   ```bash
   You: /planning Create a basic web application with the following structure:
   
   - A frontend folder containing HTML, CSS, and JavaScript files.
   
   - A backend folder with server-side scripts.
   
   - A README.md file with project documentation.
   ```

2. **Creating the Project Structure based on the Plan**:

   ```bash
   You: /create Generate the project structure based on the above plan.
   ```

This demonstrates how to use the new `/planning` command to define a project structure, and then `/create` to generate the files and folders accordingly.

### 📝 Examples

```bash
You: /add src/main.py src/utils/helper.py src/models/

You: /planning Outline a RESTful API project with separate folders for models, views, and controllers.

You: /create Set up the basic structure for a RESTful API project with models, views, and controllers folders, including initial files.

You: /edit src/main.py src/models/user.py src/views/user_view.py
```

## 🤝 Contributing

We welcome contributions! Please follow these steps:

1. Fork the repository.

2. Create a new branch (`git checkout -b feature/YourFeature`).

3. Commit your changes (`git commit -m 'Add some feature'`).

4. Push to the branch (`git push origin feature/YourFeature`).

5. Open a pull request.

## 🙏 Acknowledgments

- OpenAI for providing the powerful API.

## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=Doriandarko/o1-engineer&type=Date)](https://star-history.com/#Doriandarko/o1-engineer&Date)
