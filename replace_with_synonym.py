import nltk
from nltk.corpus import wordnet

def replace_adjectives_with_synonyms(sentence):
    tokens = nltk.word_tokenize(sentence)  # Tokenize the sentence into words
    # print("tokens:", tokens)
    tagged_tokens = nltk.pos_tag(tokens)  # Perform part-of-speech tagging
    # print("tagged tokens: ", tagged_tokens)
    replaced_sentence = []
    for word, pos in tagged_tokens:
        if pos.startswith('JJ'):  # Check if the word is an adjective
            synonyms = []
            for syn in wordnet.synsets(word):
                # print(word, syn)
                for lemma in syn.lemmas():
                    # print(lemma)
                    synonyms.append(lemma.name())  # Add synonyms to the list
            print(synonyms)

            if synonyms:
                if (len(synonyms)>1):
                    l1 = [item for item in synonyms if item != word]
                    if(len(l1)>1):
                        replaced_sentence.append(l1[0])  # Replace the adjective with the first synonym
            else:
                replaced_sentence.append(word)  # Keep the original word if no synonyms found
        else:
            replaced_sentence.append(word)  # Keep other words as they are

    return ' '.join(replaced_sentence)

# Example usage
sentence = "The weather is happy and the flowers are sad."
replaced_sentence = replace_adjectives_with_synonyms(sentence)
print(replaced_sentence)
