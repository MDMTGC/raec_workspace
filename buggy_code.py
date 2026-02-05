"""
A simple calculator module with a bug.
"""

def add(a, b):
    """Add two numbers."""
    return a + b

def subtract(a, b):
    """Subtract b from a."""
    return a - b

def multiply(a, b):
    """Multiply two numbers."""
    return a * b

def divide(a, b):
    """Divide a by b."""
    return a / b  # Bug: no zero division check

def calculate_average(numbers):
    """Calculate the average of a list of numbers."""
    total = 0
    for num in numbers:
        total = total + num
    return total / len(numbers)  # Bug: crashes on empty list

def factorial(n):
    """Calculate factorial of n."""
    if n == 0:
        return 1
    result = 1
    for i in range(1, n):  # Bug: should be range(1, n+1)
        result *= i
    return result


if __name__ == "__main__":
    # Test the functions
    print(f"5 + 3 = {add(5, 3)}")
    print(f"5 - 3 = {subtract(5, 3)}")
    print(f"5 * 3 = {multiply(5, 3)}")
    print(f"6 / 2 = {divide(6, 2)}")
    print(f"Average of [1,2,3,4,5] = {calculate_average([1,2,3,4,5])}")
    print(f"5! = {factorial(5)}")
