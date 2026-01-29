import re
from typing import Optional, Set

from functools import lru_cache
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer


class Tokenizer:
    '''
        Class that handles all stuff regarding the tokenization of the text.
        _url_regex - pre-compiled regex for finding URLs in the given text.
        _email_regex - pre-compiled regex for finding emails in the given text. Pre-compilation should work faster than compilation in-place for each text.
    '''
    _url_regex = re.compile(r"https?://\S+|www\.\S+", flags=re.UNICODE)
    _email_regex = re.compile(r"\S+@\S+", flags=re.UNICODE)


    def __init__(
        self,
        separate_alphanumeric: bool | int = False,
        remove_numbers: bool | int = False,
        remove_URLs: bool | int = False,
        remove_emails: bool | int = False,
        min_token_length: int = 1,
        lowercase: bool | int = False,
        stemmer: bool | int = False,
        use_stopwords: bool | int = False,
    ):
        self.separate_alphanumeric: bool | int = separate_alphanumeric
        self.remove_numbers: bool | int = remove_numbers
        self.remove_URLs: bool | int = remove_URLs
        self.remove_emails: bool | int = remove_emails
        self.min_token_length: int = min_token_length
        self.lowercase: bool | int = lowercase
        self.stemmer: bool | int = stemmer
        self.use_stopwords: bool | int = use_stopwords
        self.stemmer_pt: Optional[SnowballStemmer] = None
        self.stopwords_pt: Set[str] = set()

        # defining stemmer
        if self.stemmer:
            self.stemmer_pt = SnowballStemmer("portuguese")
            @lru_cache(maxsize=100000) #using cache significantly reduced text-processing time, from 1hr 10 min to 45min, since it stores most recent words
            def cached_stem(word: str) -> str:
                return self.stemmer_pt.stem(word)

            self._cached_stem = cached_stem
            

        # defining stopwords
        if self.use_stopwords:
            self.stopwords_pt = set(stopwords.words("portuguese"))

        return

    def output_configuration(self) -> str:
        configuration = (
            f"  · Tokenizer configuration:\n"
            f"     · Separate alphanumeric:"
            f"{'enabled' if self.separate_alphanumeric else 'disabled'}\n"
            f"     · Remove numbers: {'enabled' if self.remove_numbers else 'disabled'}\n"
            f"     · Remove URLs: {'enabled' if self.remove_URLs else 'disabled'}\n"
            f"     · Remove emails: {'enabled' if self.remove_emails else 'disabled'}\n"
            f"     · Min token length: {self.min_token_length}\n"
            f"     · Lowercase: {'enabled' if self.lowercase else 'disabled'}\n"
            f"     · Stemmer: {'enabled' if self.stemmer else 'disabled'}\n"
            f"     · Stopwords: {'enabled' if self.use_stopwords else 'disabled'}\n"
        )
        return configuration

    def tokenize(self, text: str) -> list[str]:
        '''
            Method that tokenizes the given text.
            input: text as string
            output: list of strings(tokens)
        '''
        # lowercase
        if self.lowercase:
            text = text.lower()

        # remove URL's
        if self.remove_URLs:
            text = self._url_regex.sub("", text)

        # remove email's 
        if self.remove_emails:
            text = self._email_regex.sub("", text)

        # basic tokenization
        # keep every word that is made of letters and numbers
        # re.UNICIDE - letters from other languages are included
        tokens: list[str] = re.findall(r"\w+", text, flags=re.UNICODE)

        # separate alphanumeric
        if self.separate_alphanumeric:
            new_tokens: list[str] = []
            for t in tokens:
                splitted = re.findall(r"\D+|\d+", t)  # abc123 → ["abc", "123"]
                new_tokens.extend(splitted)
            tokens = new_tokens

        # remove numbers
        if self.remove_numbers:
            tokens = [t for t in tokens if not any(c.isdigit() for c in t)]

        # check token length (so expensive operations like stemming arent performed if they shouldnt)
        tokens = [t for t in tokens if len(t) >= self.min_token_length]

        # remove stopwords
        if self.use_stopwords:
            tokens = [t for t in tokens if t not in self.stopwords_pt]

        # stemming
        if self.stemmer and self.stemmer_pt is not None:
            tokens = [self._cached_stem(t) for t in tokens]

        return tokens
