# CLAUDE.md - Information for AI Assistants

## Code Style Guidelines

### Python Conventions

- **Formatting**: Black (line length 88)
- **Imports**: isort with black profile
- **Type Hints**: Required for public functions
- **Docstrings**: Google style for complex functions only
- **Error Handling**: Simple try/except with logging

### Simplification Guidelines

- Prefer clarity over cleverness
- Use pandas/numpy operations over custom loops
- Minimize dependencies (each adds complexity)
- Document "why" not just "what"

## Using Gemini CLI for Large Codebase Analysis

When analyzing large codebases or multiple files that might exceed context limits, use the Gemini CLI with its massive context window. Use `gemini -p` to leverage Google Gemini's large context capacity.

### File and Directory Inclusion Syntax for Gemini

Use the `@` syntax to include files and directories in your Gemini prompts. The paths should be relative to WHERE you run the gemini command:

### Gemini Usage Examples:

Single file:
gemini -p "@src/main.py Explain this file's purpose and structure"

Multiple files:
gemini -p "@package.json @src/index.js Analyze the dependencies used in the code"

Entire directory:
gemini -p "@src/ Summarize the architecture of this codebase"

Multiple directories:
gemini -p "@src/ @tests/ Analyze test coverage for the source code"

Current directory and subdirectories:
gemini -p "@./ Give me an overview of this entire project"

All files:
gemini --all_files -p "Analyze the project structure and dependencies"

### When to Use Gemini CLI

Use gemini -p when:

- Analyzing entire codebases or large directories
- Comparing multiple large files
- Need to understand project-wide patterns or architecture
- Current context window is insufficient for the task
- Working with files totaling more than 100KB
- Verifying if specific features, patterns, or security measures are implemented
- Checking for the presence of certain coding patterns across the entire codebase

### Important Notes regarding Gemini

- Paths in @ syntax are relative to your current working directory when invoking gemini
- The CLI will include file contents directly in the context
- No need for --yolo flag for read-only analysis
- Gemini's context window can handle entire codebases that would overflow Claude's context
- When checking implementations, be specific about what you're looking for to get accurate results

## Working Principles

### Be Practical

- Focus on what will actually be implemented
- Skip academic perfection for working solutions

### Be Transparent

- Flag when you don't understand the code but are making an informed inference
- Flag when something seems over-engineered
- Suggest simpler alternatives if it makes sense to do so
- Highlight key insights over implementation details

### Be Proactive (but only when asked)

- Suggest next steps after analysis
- Propose simplifications
- Create actionable implementation plans
- Strike a balance between:
  - Doing the right thing when asked, including taking actions and follow-up actions.
  - Not surprising the user with actions you take without asking.
  - For example, if the user asks you how to approach something, you should do your best to answer their question first, and not immediately jump into taking actions.