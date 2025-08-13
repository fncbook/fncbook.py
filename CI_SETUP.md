# CI/CD Setup for PyPI Publishing

This document explains how to set up continuous integration and deployment for automatically publishing your package to PyPI when you push to the main branch.

## Prerequisites

1. **PyPI Account**: You need an account on [PyPI](https://pypi.org/)
2. **GitHub Repository**: Your code should be in a GitHub repository
3. **PyPI API Token**: You'll need to generate an API token for authentication

## Setup Instructions

### 1. Generate PyPI API Token

1. Log in to your PyPI account
2. Go to Account Settings → API tokens
3. Click "Add API token"
4. Give it a name (e.g., "GitHub Actions for fncbook")
5. Set the scope to "Entire account" or limit it to your specific project
6. Copy the generated token (it starts with `pypi-`)

### 2. Add GitHub Secret

1. Go to your GitHub repository
2. Navigate to Settings → Secrets and variables → Actions
3. Click "New repository secret"
4. Name: `PYPI_API_TOKEN`
5. Value: Paste your PyPI API token
6. Click "Add secret"

### 3. Version Management Strategy

The current workflow publishes every push to main. For better version control, consider these strategies:

- Update the version in `pyproject.toml` manually before pushing
- Current version: `0.1.2`


### 4. Workflow Features

The GitHub Actions workflow (`publish-to-pypi.yml`) includes:

- **Testing**: Runs tests across Python 3.8-3.12
- **Building**: Creates source and wheel distributions
- **Validation**: Checks package integrity with twine
- **Publishing**: Uploads to PyPI only on main branch pushes

### 5. Testing the Setup

1. Make a small change to your code
2. Update the version in `pyproject.toml` (e.g., from `0.1.2` to `0.1.3`)
3. Commit and push to main branch
4. Check the Actions tab in GitHub to see the workflow run
5. Verify the new version appears on PyPI

## Troubleshooting

### Common Issues

1. **Authentication Error**: Verify your PyPI API token is correctly set in GitHub secrets
2. **Version Conflict**: Ensure you're incrementing the version number in `pyproject.toml`
3. **Test Failures**: Fix any failing tests before the publish step will run
4. **Package Name Conflict**: Ensure your package name is unique on PyPI

### Workflow Status

You can monitor the workflow execution in the "Actions" tab of your GitHub repository.

## Security Notes

- Never commit API tokens to your repository
- Use GitHub secrets for sensitive information
- Consider using environment-specific tokens for production vs. testing

## Next Steps

1. Set up the PyPI API token in GitHub secrets
2. Test the workflow with a version bump
3. Consider implementing semantic versioning
4. Add release notes automation