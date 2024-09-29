# 🛠️ o1-engineer

A command-line tool designed to assist developers in managing and interacting with their projects efficiently. Leveraging the power of OpenAI's API, o1-engineer provides functionalities such as code generation, file editing, and project planning to streamline your development workflow.

## ✨ Features

- **Automated Code Generation**: Generate code for your projects effortlessly.

- **File Management**: Add, edit, and manage project files directly from the command line.

- **Project Planning**: Create detailed plans based on your project requirements.

- **Interactive Console**: User-friendly interface with rich text support for enhanced readability.

- **Conversation History**: Save and reset conversation histories as needed.

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

   Add your API at the top of the script

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
- `/review`: Review code files (followed by file paths)
- `/quit`: Exit the program


### 📝 Example

```bash
You: /add src/main.py src/utils/helper.py

You: /create write a professional README.md for this project
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
