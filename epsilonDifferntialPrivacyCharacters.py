import random
import wordmap
def perturb_string(text, epsilon):
    chars = list(text)
    # print("chars:", chars, "len(chars):", len(chars))
    new_char_list=[]
    for i in range(len(chars)):
        if random.random() > epsilon:
            # print("\n\n\nentered if")
            char = chars[i]
            # print("char:", char)
            if char in chars:
                new_char = wordmap.mapper[char]
                # print("new_char:", new_char)
                new_char_list.append(new_char)
                # print("new_char_list: ", new_char_list)
    perturbed_text = ''.join(new_char_list)
    return perturbed_text

name = "Hana"
perturbed_name = perturb_string(name, epsilon=0.1)
print("Perturbed name:", perturbed_name)