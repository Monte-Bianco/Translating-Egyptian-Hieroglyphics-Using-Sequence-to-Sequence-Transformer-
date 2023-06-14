import re
import unicodedata
from typing import List

class CustomMosesTokenizer:
    def __init__(self, lang='en'):
        self.lang = lang
        
        self.DASHES = re.compile(r'[-\u2010\u2011\u2012\u2013\u2014\u2015]')
        self.FINAL_PERIOD = re.compile(r'(?<![0-9])\.')
        self.NUMBER = re.compile(r'([0-9]+(?:[.,][0-9]+)*)')

        self.protect_patterns = [
            (self.FINAL_PERIOD, r' . '),
            (self.DASHES, r' - '),
            (self.NUMBER, r' \1 '),
        ]

        self.special_chars = ['[', ']', ',', '.', '§', '(', ')', '⸢', '⸣', '⸮', '?', '{', '}', '〈', '〉', '⸮', '?', '{{', '}}', '(())', '[[]]', '〈〈', '〉〉']

        self.token_to_idx = {}
        self.idx_to_token = {}

    def _apply_patterns(self, text):
        for (pattern, replace) in self.protect_patterns:
            text = pattern.sub(replace, text)
        return text

    def remove_special_chars(self, text):
        for char in self.special_chars:
            text = text.replace(char, '')
        return text

    def tokenize(self, text: str) -> List[str]:
        text = text.strip()
        text = self._apply_patterns(text)
        text = self.remove_special_chars(text)
        text = unicodedata.normalize('NFKC', text)
        text = re.sub(r'\s+', ' ', text)
        return text.split()

    def build_vocab(self, texts, min_freq=2):
        token_freq = {}
        for text in texts:
            tokens = self.tokenize(text)
            for token in tokens:
                if token not in token_freq:
                    token_freq[token] = 0
                token_freq[token] += 1

        tokens = [token for token, freq in token_freq.items() if freq >= min_freq]

        self.token_to_idx = {'<PAD>': 0, '<UNK>': 1, '<SOS>': 2, '<EOS>': 3}
        self.idx_to_token = {0: '<PAD>', 1: '<UNK>', 2: '<SOS>', 3: '<EOS>'}

        for token in tokens:
            idx = len(self.token_to_idx)
            self.token_to_idx[token] = idx
            self.idx_to_token[idx] = token

    def convert_tokens_to_ids(self, tokens):
        return [self.token_to_idx.get(token, self.token_to_idx['<UNK>']) for token in tokens]

    def convert_ids_to_tokens(self, ids):
        return [self.idx_to_token.get(id, '<UNK>') for id in ids]
    
    def vocab_size(self):
        return len(self.token_to_idx)

# Example usage
custom_tokenizer = CustomMosesTokenizer()
text = "jri̯ [n] =[n] [m] w[ḏ].n =k nb"
tokens = custom_tokenizer.tokenize(text)
print(tokens)


