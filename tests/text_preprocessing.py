import string

import nltk
from nltk import word_tokenize
from nltk.corpus import stopwords
import re
from nltk.stem import PorterStemmer, WordNetLemmatizer

nltk.download("stopwords")
cachedStopWords = stopwords.words("english")


def text_cleaner(text, keep_punctuation=False):
    text = to_unicode(text)
    tokenized_text = list()
    for token in text.split(' '):
        word = u''
        code_token = u''
        for char in token:
            if char.isupper() and all(map(lambda x: x.isupper(), word)):
                # keep building if word is currently all uppercase
                word += char
                code_token += char
            elif char.islower() and all(map(lambda x: x.isupper(), word)):
                # stop building if word is currently all uppercase,
                # but be sure to take the first letter back
                # new version: emit splitted camel case but also preserve code token
                if len(word) > 1:
                    tokenized_text.append(word[:-1])
                    word = word[-1]
                word += char
                code_token += char
            elif char.islower() and any(map(lambda x: x.islower(), word)):
                # keep building if the word is has any lowercase
                # (word came from above case)
                word += char
                code_token += char
            elif char.isdigit() and all(map(lambda x: x.isdigit(), word)):
                # keep building if all of the word is a digit so far
                word += char
                code_token += char
            elif char in string.punctuation:
                if len(word) > 0 and not all(map(lambda x: x.isdigit(), word)):
                    tokenized_text.append(word)
                    if code_token != word:
                        tokenized_text.append(code_token)
                    code_token = u''
                    word = u''

                if keep_punctuation is True:
                    tokenized_text.append(char)

                # dont yield punctuation
                # yield char
            elif char == ' ':
                if len(word) > 0 and not all(map(lambda x: x.isdigit(), word)):
                    tokenized_text.append(word)
                    if code_token != word:
                        tokenized_text.append(code_token)

                word = u''
                code_token = u''
            else:
                if len(word) > 0 and not all(map(lambda x: x.isdigit(), word)):
                    tokenized_text.append(word)

                # to make sure we have only unicode characters
                word = u''
                word += char
                code_token += char

        tokenized_text.append(word)
        if code_token != word:
            tokenized_text.append(code_token)

    text = ' '.join(tokenized_text).replace('  ', ' ').lower()
    # text = PorterStemmer().stem(text)
    text = WordNetLemmatizer().lemmatize(text)

    # we preserve spaces as tokens so we need to remove double-spaces after joining
    return text


def to_unicode(text):
    printable = set(string.printable)
    text = ''.join(filter(lambda x: x in printable, text))
    text = text.replace('\x00', ' ')  # remove nulls
    text = text.replace('\r', ' ')
    text = text.replace('\n', ' ')
    text = re.sub('[^0-9a-zA-Z]+', ' ', text)
    text = text.strip()
    return text


if __name__ == "__main__":
    query_string = "java.lang.UnsupportedOperationException"
    print(text_cleaner(query_string))
    query_string = "32-bit processor version windows#10."
    print(text_cleaner(query_string))
