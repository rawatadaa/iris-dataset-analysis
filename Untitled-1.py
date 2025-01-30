def calculate_hcf(a, b):
    while b:
        a, b = b, a % b
    return a

# Input two numbers
num1 = int(input("Enter the first number: "))


# Print the HCF
print(calculate_hcf(num1))
2
