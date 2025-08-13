#!/usr/bin/env python3
"""
Simple version bumping script for fncbook package.
Usage: python bump_version.py [major|minor|patch]
"""

import re
import sys
from pathlib import Path


def get_current_version():
    """Read current version from pyproject.toml"""
    pyproject_path = Path("pyproject.toml")
    if not pyproject_path.exists():
        raise FileNotFoundError("pyproject.toml not found")
    
    content = pyproject_path.read_text()
    version_match = re.search(r'version = "([^"]+)"', content)
    if not version_match:
        raise ValueError("Version not found in pyproject.toml")
    
    return version_match.group(1)


def bump_version(current_version, bump_type):
    """Bump version based on type (major, minor, patch)"""
    major, minor, patch = map(int, current_version.split('.'))
    
    if bump_type == 'major':
        major += 1
        minor = 0
        patch = 0
    elif bump_type == 'minor':
        minor += 1
        patch = 0
    elif bump_type == 'patch':
        patch += 1
    else:
        raise ValueError("bump_type must be 'major', 'minor', or 'patch'")
    
    return f"{major}.{minor}.{patch}"


def update_version_in_file(new_version):
    """Update version in pyproject.toml"""
    pyproject_path = Path("pyproject.toml")
    content = pyproject_path.read_text()
    
    # Replace version line
    new_content = re.sub(
        r'version = "[^"]+"',
        f'version = "{new_version}"',
        content
    )
    
    pyproject_path.write_text(new_content)


def main():
    if len(sys.argv) != 2:
        print("Usage: python bump_version.py [major|minor|patch]")
        sys.exit(1)
    
    bump_type = sys.argv[1].lower()
    if bump_type not in ['major', 'minor', 'patch']:
        print("Error: bump_type must be 'major', 'minor', or 'patch'")
        sys.exit(1)
    
    try:
        current_version = get_current_version()
        new_version = bump_version(current_version, bump_type)
        
        print(f"Current version: {current_version}")
        print(f"New version: {new_version}")
        
        # Confirm before updating
        response = input("Update version? (y/N): ")
        if response.lower() in ['y', 'yes']:
            update_version_in_file(new_version)
            print(f"Version updated to {new_version}")
            print("Don't forget to commit and push your changes!")
        else:
            print("Version update cancelled")
    
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()