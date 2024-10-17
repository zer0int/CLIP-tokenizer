import os
import argparse
import gzip
import html
from functools import lru_cache
import ftfy
import regex as re

@lru_cache()
def default_bpe():
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), "bpe_simple_vocab_16e6.txt.gz")

@lru_cache()
def bytes_to_unicode():
    bs = list(range(ord("!"), ord("~")+1)) + list(range(ord("¡"), ord("¬")+1)) + list(range(ord("®"), ord("ÿ")+1))
    cs = bs[:]
    n = 0
    for b in range(2**8):
        if b not in bs:
            bs.append(b)
            cs.append(2**8+n)
            n += 1
    cs = [chr(n) for n in cs]
    return dict(zip(bs, cs))

def get_pairs(word):
    pairs = set()
    prev_char = word[0]
    for char in word[1:]:
        pairs.add((prev_char, char))
        prev_char = char
    return pairs

def basic_clean(text):
    text = ftfy.fix_text(text)
    text = html.unescape(html.unescape(text))
    return text.strip()

def whitespace_clean(text):
    text = re.sub(r'\s+', ' ', text)
    text = text.strip()
    return text

class SimpleTokenizer(object):
    def __init__(self, bpe_path: str = default_bpe()):
        self.byte_encoder = bytes_to_unicode()
        self.byte_decoder = {v: k for k, v in self.byte_encoder.items()}
        merges = gzip.open(bpe_path).read().decode("utf-8").split('\n')
        merges = merges[1:49152-256-2+1]
        merges = [tuple(merge.split()) for merge in merges]
        vocab = list(bytes_to_unicode().values())
        vocab = vocab + [v+'</w>' for v in vocab]
        for merge in merges:
            vocab.append(''.join(merge))
        vocab.extend(['', ''])
        self.encoder = dict(zip(vocab, range(len(vocab))))
        self.decoder = {v: k for k, v in self.encoder.items()}
        self.bpe_ranks = dict(zip(merges, range(len(merges))))
        self.cache = {}
        self.pat = re.compile(r"""<\|startoftext\|>|<\|endoftext\|>|'s|'t|'re|'ve|'m|'ll|'d|[\p{L}]+|[\p{N}]|[^\s\p{L}\p{N}]+""", re.IGNORECASE)

    def bpe(self, token):
        if token in self.cache:
            return self.cache[token]
        word = tuple(token[:-1]) + (token[-1] + '</w>',)
        pairs = get_pairs(word)

        if not pairs:
            return token + '</w>'

        while True:
            bigram = min(pairs, key=lambda pair: self.bpe_ranks.get(pair, float('inf')))
            if bigram not in self.bpe_ranks:
                break
            first, second = bigram
            new_word = []
            i = 0
            while i < len(word):
                try:
                    j = word.index(first, i)
                    new_word.extend(word[i:j])
                    i = j
                except ValueError:
                    new_word.extend(word[i:])
                    break

                if word[i] == first and i < len(word) - 1 and word[i + 1] == second:
                    new_word.append(first + second)
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1
            word = tuple(new_word)
            if len(word) == 1:
                break
            else:
                pairs = get_pairs(word)
        word = ' '.join(word)
        self.cache[token] = word
        return word

    def encode(self, text):
        bpe_tokens = []
        text = whitespace_clean(basic_clean(text)).lower()
        for token in re.findall(self.pat, text):
            token = ''.join(self.byte_encoder[b] for b in token.encode('utf-8'))
            bpe_tokens.extend(bpe_token for bpe_token in self.bpe(token).split(' '))
        return [self.encoder[token] for token in bpe_tokens if token in self.encoder]

    def decode(self, tokens):
        text = ''.join([self.decoder[token] for token in tokens])
        text = bytearray([self.byte_decoder[c] for c in text if c in self.byte_decoder]).decode('utf-8', errors='replace').replace('</w>', ' ')
        return whitespace_clean(text)
        
    def ids_to_tokens(self, token_ids):
        return [self.decoder.get(token_id, '<UNK>') for token_id in token_ids]

def process_file(tokenizer, filename, reverse=False):
    with open(filename, 'r') as file:
        lines = file.readlines()

    output_filename = f"{os.path.splitext(filename)[0]}_tokenizer{'-rev' if reverse else ''}.txt"
    with open(output_filename, 'w') as output_file:
        for line in lines:
            line = line.strip()
            if reverse:
                token_ids = list(map(int, line.split(',')))
                decoded_text = tokenizer.decode(token_ids)
                tokens = tokenizer.ids_to_tokens(token_ids)
                output_line = f"{decoded_text}\t{tokens}\t{','.join(map(str, token_ids))}\n"
            else:
                token_ids = tokenizer.encode(line)
                tokens = tokenizer.ids_to_tokens(token_ids)
                output_line = f"{line}\t{tokens}\t{','.join(map(str, token_ids))}\n"
            
            output_file.write(output_line)
    
    print(f"Processing completed. Results saved to '{output_filename}'")

def main():
    parser = argparse.ArgumentParser(description="CLIP Tokenizer Tool")
    parser.add_argument('--text', type=str, help="Text input for tokenization")
    parser.add_argument('--file', type=str, help="Path to a file with text or token IDs on each line")
    parser.add_argument('--reverse', action='store_true', help="Set to True for decoding from token IDs to text")
    args = parser.parse_args()

    tokenizer = SimpleTokenizer()

    if args.file:
        process_file(tokenizer, args.file, args.reverse)
    elif args.text:
        if args.reverse:
            token_ids = list(map(int, args.text.split(',')))
            decoded_text = tokenizer.decode(token_ids)
            tokens = tokenizer.ids_to_tokens(token_ids)
            print(f"{decoded_text}\t{tokens}\t{','.join(map(str, token_ids))}")
        else:
            token_ids = tokenizer.encode(args.text)
            tokens = tokenizer.ids_to_tokens(token_ids)
            print(f"{args.text}\t{tokens}\t{','.join(map(str, token_ids))}")
    else:
        print("Please provide either --text or --file input.")

if __name__ == "__main__":
    main()