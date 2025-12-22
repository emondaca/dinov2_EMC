import re

def clean_sentence_label(row):
    sentence = row['sentence_en']
    if not isinstance(sentence, str):
        return sentence

    words_to_remove = []
    for label in row['label_group']:
        words_to_remove.extend(re.findall(r"[A-Za-z']+", label.lower()))

    if words_to_remove:
        pattern = r'\b(?:' + '|'.join(set(words_to_remove)) + r')\b'
        sentence = re.sub(pattern, '', sentence, flags=re.IGNORECASE)


    sentence = re.sub(r'[.,!?;:]', '', sentence)
    sentence = re.sub(r'\s+', ' ', sentence).strip()

    return sentence

import pandas as pd
import re

# 1) Load terms from an Excel that may have multiple sheets
def load_terms_from_excel(path, column='term', normalize=True):
    sheets = pd.read_excel(path, sheet_name=None)  # dict: sheet_name -> DataFrame
    terms = []
    for _, df in sheets.items():
        if column in df:
            t = df[column].dropna().astype(str).str.strip().tolist()
            terms.extend(t)

    # Optional normalize: lowercase and turn underscores (e.g., "no_pneumonia") into spaces
    if normalize:
        terms = [x.lower().replace('_', ' ') for x in terms]

    # Deduplicate and drop empties
    terms = sorted(set(x for x in terms if x))
    return terms

# 2) Build a regex that matches whole phrases
def build_phrase_regex(terms):
    if not terms:
        return None

    # Convert "right pleural effusion" -> r'\bright\b\s+\bpleural\b\s+\beffusion\b'
    parts = []
    for t in terms:
        toks = [re.escape(tok) for tok in t.split()]
        if toks:
            parts.append(r'\b' + r'\s+'.join(toks) + r'\b')

    # Sort by length (desc) so longer phrases are removed before their subparts
    parts.sort(key=len, reverse=True)

    # If the list is huge, you can chunk; for most cases one big regex is fine
    pattern = re.compile(r'(?:' + '|'.join(parts) + r')', flags=re.IGNORECASE)
    return pattern

# === Use the functions ===
def clean_suspects_terms(xlsx_path=""):
    terms_to_remove = load_terms_from_excel(xlsx_path, column='term', normalize=True)
    print(terms_to_remove)
    phrase_re = build_phrase_regex(terms_to_remove)

    return phrase_re

def remove_exclusive_terms(text, regex):
    if not isinstance(text, str) or regex is None:
        return text
    cleaned = regex.sub('', text)
    cleaned = re.sub(r'\s+', ' ', cleaned).strip()
    return cleaned