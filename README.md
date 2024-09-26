# 🛠️ o1-engineer

A command-line tool designed to assist developers in managing and interacting with their projects efficiently. Leveraging the power of OpenAI's API, o1-engineer provides functionalities such as code generation, file editing, and project planning to streamline your development workflow.

## ✨ Features

- **Automated Code Generation**: Generate code for your projects effortlessly.

- **File Management**: Add, edit, and manage project files directly from the command line.

- **Project Planning**: Create detailed plans based on your project requirements.

- **Interactive Console**: User-friendly interface with rich text support for enhanced readability.

- **Cost Tracking**: Monitor the usage and costs associated with API calls.

- **Conversation History**: Save and reset conversation histories as needed.

## 📥 Installation

### Prerequisites

- **Python**: Ensure you have Python 3.7 or higher installed. [Download Python](https://www.python.org/downloads/)

- **OpenAI API Key**: Obtain an API key from [OpenAI](https://platform.openai.com/).

### 🔧 Steps

1. **Clone the Repository**:

   ```bash
   git clone https://github.com/Doriandarko/o1-engineer.git
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

- `exit`: Quit the application.

- `reset`: Clear the conversation history.

- `save`: Save the current conversation history to a Markdown file.

- `/add file1 [file2 ...]`: Add specified files to the conversation context.

- `/edit file1 [file2 ...]`: Edit specified files based on AI suggestions.

- `planning`: Enter the planning mode to create detailed project plans.

### 📝 Example

```bash
You: /add src/main.py src/utils/helper.py

You: write a professional README.md for this project
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

## ✨ Star History

[![Star History Chart](https://api.star-history.com/svg?repos=o1-engineer/o1-engineer&type=Date)](https://star-history.com/#o1-engineer/o1-engineer&Date)

