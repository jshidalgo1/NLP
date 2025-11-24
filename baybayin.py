"""
Baybayin Transliteration Engine
Converts Tagalog text (Latin script) to Baybayin script (Unicode U+1700 to U+171F).

Baybayin is an abugida (alphasyllabary), where each character represents a consonant 
with an inherent 'a' vowel that can be modified using diacritical marks (kudlit).
"""

import re


class BaybayinTransliterator:
    def __init__(self):
        # Baybayin Unicode characters (U+1700 to U+171F)
        # Standalone vowels
        self.vowels = {
            'a': 'ᜀ',  # U+1700
            'e': 'ᜁ',  # U+1701
            'i': 'ᜁ',  # U+1701
            'o': 'ᜂ',  # U+1702
            'u': 'ᜂ',  # U+1702
        }
        
        # Base consonants (inherent 'a' vowel)
        # These represent ka, ga, nga, ta, da, na, pa, ba, ma, ya, ra/la, wa, sa, ha
        self.consonants = {
            'k': 'ᜃ',  # U+1703
            'g': 'ᜄ',  # U+1704
            'ng': 'ᜅ', # U+1705
            't': 'ᜆ',  # U+1706
            'd': 'ᜇ',  # U+1707
            'n': 'ᜈ',  # U+1708
            'p': 'ᜉ',  # U+1709
            'b': 'ᜊ',  # U+170A
            'm': 'ᜋ',  # U+170B
            'y': 'ᜌ',  # U+170C
            'r': 'ᜍ',  # U+170D
            'l': 'ᜍ',  # U+170D (r and l share the same character)
            'w': 'ᜏ',  # U+170F
            's': 'ᜐ',  # U+1710
            'h': 'ᜑ',  # U+1711
        }
        
        # Kudlit (diacritical marks)
        self.kudlit_i = 'ᜒ'  # U+1712 - for i/e sound
        self.kudlit_o = 'ᜓ'  # U+1713 - for o/u sound
        self.virama = '᜔'    # U+1714 - pamudpod/virama (cancels inherent vowel)
        
    def normalize_text(self, text: str) -> str:
        """
        Normalize text by converting to lowercase and replacing foreign letters
        with Tagalog equivalents.
        """
        text = text.lower()
        
        # Replace foreign letters with Tagalog equivalents
        replacements = {
            'c': 'k',
            'f': 'p',
            'j': 'dy',
            'q': 'k',
            'v': 'b',
            'x': 'ks',
            'z': 's',
            'ñ': 'ny',
        }
        
        for foreign, tagalog in replacements.items():
            text = text.replace(foreign, tagalog)
            
        return text
    
    def tokenize_syllables(self, text: str) -> list:
        """
        Tokenize text into syllables using regex patterns.
        Tagalog syllables follow these patterns:
        - V (vowel)
        - CV (consonant + vowel)
        - CCV (consonant cluster + vowel)
        - VC (vowel + consonant) - for word-final positions
        - CVC (consonant + vowel + consonant)
        """
        # Define consonant and vowel patterns
        consonants = r'(?:ng|[kgtndbpmyrlwsh])'
        vowels = r'[aeiou]'
        
        # Syllable patterns (order matters - match longer patterns first)
        # We only use CV, V, and C.
        # Complex patterns like CCV, CVC, VC are handled by combining these basic units.
        # This ensures correct Baybayin rendering where clusters are split (e.g. 'sta' -> 's' + 'ta').
        patterns = [
            f'{consonants}{vowels}',              # CV - e.g., "ma", "ka", "nga"
            f'{vowels}',                          # V - e.g., "a", "i"
            f'{consonants}',                      # C - standalone consonant
        ]
        
        # Combine patterns
        pattern = '|'.join(f'({p})' for p in patterns)
        
        # Tokenize
        tokens = []
        for match in re.finditer(pattern, text):
            token = match.group(0)
            if token:  # Skip empty matches
                tokens.append(token)
        
        return tokens
    
    def syllable_to_baybayin(self, syllable: str) -> str:
        """
        Convert a single syllable to Baybayin script.
        """
        if not syllable:
            return ''
        
        # Check if it's just a vowel
        if syllable in self.vowels:
            return self.vowels[syllable]
        
        # Check if it's just a consonant (use virama)
        if syllable in self.consonants:
            return self.consonants[syllable] + self.virama
        
        # Handle 'ng' digraph first
        if syllable.startswith('ng'):
            base = self.consonants['ng']
            rest = syllable[2:]
        else:
            # Get the first consonant
            base = self.consonants.get(syllable[0], '')
            rest = syllable[1:]
        
        if not base:
            # Should not happen with our new tokenizer, but safe fallback
            return syllable
        
        # If rest is empty, return base with 'a' sound (inherent)
        if not rest:
            return base
        
        # Check the vowel and apply kudlit
        if rest[0] in ['i', 'e']:
            result = base + self.kudlit_i
        elif rest[0] in ['o', 'u']:
            result = base + self.kudlit_o
        elif rest[0] == 'a':
            result = base  # Inherent 'a' vowel
        else:
            result = base
            
        return result
    
    def transliterate(self, text: str) -> str:
        """
        Main transliteration function.
        Converts Tagalog text to Baybayin script.
        """
        if not text:
            return ''
        
        # Normalize the text
        normalized = self.normalize_text(text)
        
        # Split into words to preserve spaces
        words = normalized.split()
        
        baybayin_words = []
        for word in words:
            # Tokenize into syllables
            syllables = self.tokenize_syllables(word)
            
            # Convert each syllable
            baybayin_syllables = [self.syllable_to_baybayin(syl) for syl in syllables]
            
            # Join syllables
            baybayin_word = ''.join(baybayin_syllables)
            baybayin_words.append(baybayin_word)
        
        # Join words with spaces
        return ' '.join(baybayin_words)


# Convenience function
def to_baybayin(text: str) -> str:
    """
    Convert Tagalog text to Baybayin script.
    
    Args:
        text: Tagalog text in Latin script
        
    Returns:
        Text in Baybayin script (Unicode)
    """
    transliterator = BaybayinTransliterator()
    return transliterator.transliterate(text)


# Test the transliterator if run directly
if __name__ == "__main__":
    test_words = [
        "kamusta",
        "maganda",
        "salamat",
        "ako",
        "ikaw",
        "pilipinas",
        "mahal kita",
    ]
    
    transliterator = BaybayinTransliterator()
    
    print("Baybayin Transliteration Test\n" + "="*50)
    for word in test_words:
        baybayin = transliterator.transliterate(word)
        print(f"{word:20} → {baybayin}")
