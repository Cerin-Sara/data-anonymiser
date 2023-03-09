import random

def perturb_string(text, epsilon):
    # Define a dictionary of character replacements
    replacements = {
        'a': ['e', 'o'],
        'i': ['e', 'o'],
        'o': ['u', 'a'],
        'u': ['o', 'e'],
        'e': ['i', 'a']
    }
    # Split the text into characters
    chars = list(text)
    # Perturb each character with probability epsilon
    for i in range(len(chars)):
        if random.random() < epsilon:
            char = chars[i]
            if char in replacements:
                replacements_list = replacements[char]
                new_char = random.choice(replacements_list)
                chars[i] = new_char
    # Join the characters back into a string
    perturbed_text = ''.join(chars)
    # Return the perturbed string
    return perturbed_text

name = "John Smith"
perturbed_name = perturb_string(name, epsilon=0.1)
print("Perturbed name:", perturbed_name)
