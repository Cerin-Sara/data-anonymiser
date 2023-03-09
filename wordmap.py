import random
def generateNumber(n):
    num = random.randint(0, 10)
    if(num != n):
        return num
    else:
        return generateNumber(n)
        
def generateSmallLetter(l):
    letter = chr(random.randint(97, 122))
    if(letter != l):
        return letter
    else:
        return generateSmallLetter(l)
    
def generateCapitalLetter(l):
    letter = chr(random.randint(65, 90))
    if(letter != l):
        return letter
    else:
        return generateCapitalLetter(l)
    
mapper={
    '0': str(generateNumber(0)),
    '1': str(generateNumber(1)),
    '2': str(generateNumber(2)),
    '3': str(generateNumber(3)),
    '4': str(generateNumber(4)),
    '5': str(generateNumber(5)),
    '6': str(generateNumber(6)),
    '7': str(generateNumber(7)),
    '8': str(generateNumber(8)),
    '9': str(generateNumber(9)),
    'a': str(generateSmallLetter('a')),
    'b': str(generateSmallLetter('b')),
    'c': str(generateSmallLetter('c')),
    'd': str(generateSmallLetter('d')),
    'e': str(generateSmallLetter('e')),
    'f': str(generateSmallLetter('f')),
    'g': str(generateSmallLetter('g')),
    'h': str(generateSmallLetter('h')),
    'i': str(generateSmallLetter('i')),
    'j': str(generateSmallLetter('j')),
    'k': str(generateSmallLetter('k')),
    'l': str(generateSmallLetter('l')),
    'm': str(generateSmallLetter('m')),
    'n': str(generateSmallLetter('n')),
    'o': str(generateSmallLetter('o')),
    'p': str(generateSmallLetter('p')),
    'q': str(generateSmallLetter('q')),
    'r': str(generateSmallLetter('r')),
    's': str(generateSmallLetter('s')),
    't': str(generateSmallLetter('t')),
    'u': str(generateSmallLetter('u')),
    'v': str(generateSmallLetter('v')),
    'w': str(generateSmallLetter('w')),
    'x': str(generateSmallLetter('x')),
    'y': str(generateSmallLetter('y')),
    'z': str(generateSmallLetter('z')),
    'A': str(generateCapitalLetter('A')),
    'B': str(generateCapitalLetter('B')),
    'C': str(generateCapitalLetter('C')),
    'D': str(generateCapitalLetter('D')),
    'E': str(generateCapitalLetter('E')),
    'F': str(generateCapitalLetter('F')),
    'G': str(generateCapitalLetter('G')),
    'H': str(generateCapitalLetter('H')),
    'I': str(generateCapitalLetter('I')),
    'J': str(generateCapitalLetter('J')),
    'K': str(generateCapitalLetter('K')),
    'L': str(generateCapitalLetter('L')),
    'M': str(generateCapitalLetter('M')),
    'N': str(generateCapitalLetter('N')),
    'O': str(generateCapitalLetter('O')),
    'P': str(generateCapitalLetter('P')),
    'Q': str(generateCapitalLetter('Q')),
    'R': str(generateCapitalLetter('R')),
    'S': str(generateCapitalLetter('S')),
    'T': str(generateCapitalLetter('T')),
    'U': str(generateCapitalLetter('U')),
    'V': str(generateCapitalLetter('V')),
    'W': str(generateCapitalLetter('W')),
    'X': str(generateCapitalLetter('X')),
    'Y': str(generateCapitalLetter('Y')),
    'Z': str(generateCapitalLetter('Z')),
    '@': '@',

}

# print(mapper)
# for x in mapper.values():
#   print(x) 

word="123Hello"

# for letter in word:
    # print(mapper[letter], end="")