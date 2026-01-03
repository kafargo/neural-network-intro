#!/usr/bin/env python
"""
run_tests.py
~~~~~~~~~~~~

Simple test runner script that provides common test execution patterns.
"""

import sys
import subprocess


def run_command(cmd):
    """Run a command and return the exit code."""
    print(f"\n{'='*60}")
    print(f"Running: {' '.join(cmd)}")
    print('='*60)
    result = subprocess.run(cmd)
    return result.returncode


def main():
    """Main test runner."""
    if len(sys.argv) > 1:
        command = sys.argv[1]
    else:
        command = "all"

    exit_code = 0

    if command == "all":
        # Run all tests
        exit_code = run_command(["pytest", "tests/", "-v"])

    elif command == "unit":
        # Run only unit tests
        exit_code = run_command(["pytest", "tests/", "-v", "-m", "unit"])

    elif command == "integration":
        # Run only integration tests
        exit_code = run_command(["pytest", "tests/", "-v", "-m", "integration"])

    elif command == "api":
        # Run only API tests
        exit_code = run_command(["pytest", "tests/", "-v", "-m", "api"])

    elif command == "fast":
        # Run all tests except slow ones
        exit_code = run_command(["pytest", "tests/", "-v", "-m", "not slow"])

    elif command == "coverage":
        # Run tests with coverage report
        exit_code = run_command([
            "pytest", "tests/", "-v",
            "--cov=src",
            "--cov-report=html",
            "--cov-report=term"
        ])
        if exit_code == 0:
            print("\n" + "="*60)
            print("Coverage report generated in: htmlcov/index.html")
            print("="*60)

    elif command == "help":
        print("\nTest Runner Commands:")
        print("  all         - Run all tests (default)")
        print("  unit        - Run only unit tests")
        print("  integration - Run only integration tests")
        print("  api         - Run only API tests")
        print("  fast        - Run all tests except slow ones")
        print("  coverage    - Run tests with coverage report")
        print("  help        - Show this help message")
        print("\nExamples:")
        print("  python run_tests.py")
        print("  python run_tests.py unit")
        print("  python run_tests.py coverage")

    else:
        print(f"Unknown command: {command}")
        print("Run 'python run_tests.py help' for available commands")
        exit_code = 1

    sys.exit(exit_code)


if __name__ == "__main__":
    main()

