# o1-mini-engineer

## Overview
o1-mini-engineer is a command-line tool designed to assist developers in managing and interacting with their projects efficiently. Leveraging the power of OpenAI's API, this tool provides functionalities such as code generation, file editing, and project planning to streamline your development workflow.

## Features
- **Automated Code Generation**: Generate boilerplate code for your projects effortlessly.
- **File Management**: Add, edit, and manage project files directly from the command line.
- **Project Planning**: Create detailed plans based on your project requirements.
- **Interactive Console**: User-friendly interface with rich text support for enhanced readability.

## Installation

### Prerequisites
- **Python**: Ensure you have Python 3.7 or higher installed. [Download Python](https://www.python.org/downloads/)
- **OpenAI API Key**: Obtain an API key from [OpenAI](https://platform.openai.com/).

### Steps
1. **Clone the Repository**:
   ```bash
   git clone https://github.com/yourusername/o1-mini-engineer.git
   ```

2. **Navigate to the Project Directory**:
   ```bash
   cd o1-mini-engineer
   ```

3. **Create a Virtual Environment** (optional but recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scriptsctivate
   ```

4. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

5. **Configure Environment Variables**:
   Create a `.env` file in the root directory and add your OpenAI API key:
   ```
   OPENAI_API_KEY=your-api-key-here
   ```

## Usage
Launch the application using the following command:
```bash
python o1-eng.py
```

### Available Commands
- `exit`: Quit the application.
- `reset`: Clear the conversation history.
- `save`: Save the current conversation history to a Markdown file.
- `/add file1 [file2 ...]`: Add specified files to the conversation context.
- `/edit file1 [file2 ...]`: Edit specified files based on AI suggestions.
- `planning`: Enter the planning mode to create detailed project plans.

### Example
```bash
You: /add src/main.py src/utils/helper.py
You: write a professional README.md for this project
```

## Contributing
We welcome contributions! Please follow these steps:
1. Fork the repository.
2. Create a new branch (`git checkout -b feature/YourFeature`).
3. Commit your changes (`git commit -m 'Add some feature'`).
4. Push to the branch (`git push origin feature/YourFeature`).
5. Open a pull request.


## Acknowledgments
- OpenAI for providing the powerful API.
- Contributors and the open-source community for their valuable resources.
