# Contributing to FaceSwap Application

We love your input! We want to make contributing to FaceSwap Application as easy and transparent as possible, whether it's:

- Reporting a bug
- Discussing the current state of the code
- Submitting a fix
- Proposing new features
- Becoming a maintainer

## Development Process

We use GitHub to host code, to track issues and feature requests, as well as accept pull requests.

## Pull Requests

Pull requests are the best way to propose changes to the codebase. We actively welcome your pull requests:

1. Fork the repo and create your branch from `main`.
2. If you've added code that should be tested, add tests.
3. If you've changed APIs, update the documentation.
4. Ensure the test suite passes.
5. Make sure your code lints.
6. Issue that pull request!

## Any contributions you make will be under the MIT Software License

In short, when you submit code changes, your submissions are understood to be under the same [MIT License](http://choosealicense.com/licenses/mit/) that covers the project. Feel free to contact the maintainers if that's a concern.

## Report bugs using GitHub's [issue tracker](https://github.com/yourusername/faceswap-application/issues)

We use GitHub issues to track public bugs. Report a bug by [opening a new issue](https://github.com/yourusername/faceswap-application/issues/new); it's that easy!

## Write bug reports with detail, background, and sample code

**Great Bug Reports** tend to have:

- A quick summary and/or background
- Steps to reproduce
  - Be specific!
  - Give sample code if you can
- What you expected would happen
- What actually happens
- Notes (possibly including why you think this might be happening, or stuff you tried that didn't work)

People *love* thorough bug reports. I'm not even kidding.

## Development Setup

1. Fork and clone the repository
2. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   pip install -e .
   ```
4. Run tests:
   ```bash
   python -m pytest tests/
   ```

## Code Style

We use Python's PEP 8 style guide. Please ensure your code follows these conventions:

- Use 4 spaces for indentation
- Line length should not exceed 88 characters
- Use meaningful variable and function names
- Add docstrings to all functions and classes
- Use type hints where appropriate

## Testing

Please add tests for any new functionality. We use pytest for testing:

```bash
# Run all tests
python -m pytest

# Run specific test file
python -m pytest tests/test_face_detector.py

# Run with coverage
python -m pytest --cov=faceswap
```

## Documentation

If you're adding new features or changing existing functionality:

1. Update the README.md if needed
2. Add docstrings to new functions/classes
3. Update the CHANGELOG.md
4. Consider adding examples to the documentation

## Commit Messages

Please use clear and meaningful commit messages:

- Use the present tense ("Add feature" not "Added feature")
- Use the imperative mood ("Move cursor to..." not "Moves cursor to...")
- Limit the first line to 72 characters or less
- Reference issues and pull requests liberally after the first line

## Feature Requests

We welcome feature requests! Please:

1. Check if the feature has already been requested
2. Provide a clear description of the feature
3. Explain why this feature would be useful
4. Consider providing a basic implementation plan

## Code of Conduct

By participating in this project, you agree to abide by our Code of Conduct:

- Be respectful and inclusive
- Welcome newcomers and help them get started
- Focus on constructive feedback
- Respect different viewpoints and experiences

## Questions?

Don't hesitate to ask questions by opening an issue or reaching out to the maintainers.

Thanks for contributing! 🚀