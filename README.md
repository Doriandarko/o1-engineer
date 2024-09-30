# 🛠️ o1-engineer

A command-line tool designed to assist developers in managing and interacting with their projects efficiently. Leveraging the power of OpenAI's API, o1-engineer provides functionalities such as code generation, file editing, project planning, and code review to streamline your development workflow.

## ✨ Features

- **Automated Code Generation**: Generate code for your projects effortlessly.

- **File Management**: Add, edit, and manage project files directly from the command line.

- **Interactive Console**: User-friendly interface with rich text support for enhanced readability.

- **Conversation History**: Save and reset conversation histories as needed.

- **Code Review**: Analyze and review code files for quality and suggestions.

## 💡 How the Script Works

1. **Initialization**: The script initializes global variables and sets up the OpenAI client using the provided API key.

2. **Handling User Commands**: It listens for user commands such as `/edit`, `/create`, `/add`, and `/review`, and processes them accordingly.

3. **Processing File Modifications**: Based on the user's instructions, the script modifies files, adds new content, or creates new files and folders as needed.

4. **AI-Generated Instructions**: The tool interacts with OpenAI's API to generate instructions and suggestions for code generation, editing, and reviewing.

5. **Applying Changes**: Changes are applied to the project files based on the AI-generated instructions, ensuring that the project stays up-to-date and well-maintained.

6. **Managing Conversation History and Added Files**: The script manages the conversation history and keeps track of files added to the context, allowing users to reset or modify the history as needed.

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

- `/edit`: Edit files (followed by file paths)

- `/create`: Create files or folders (followed by instructions)

- `/add`: Add files to context (followed by file paths)

- `/debug`: Print the last AI response

- `/reset`: Reset chat context and clear added files

- `/review`: Review and analyze code files for quality and potential improvements (followed by file paths)

- `/quit`: Exit the program


### 📝 Example

```bash
You: /add src/main.py src/utils/helper.py

You: /create a game of snake with 3 files, css, js and htlm all in one folder
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
