"""Clean and preprocess data for emotion classification."""

import re
import pandas as pd
from tqdm import tqdm

CONTRACTIONS = {
    "im": "i'm",
    "ive": "i've",
    "id": "i'd",
    "ill": "i'll",
    "youre": "you're",
    "youve": "you've",
    "youd": "you'd",
    "youll": "you'll",
    "hes": "he's",
    "shes": "she's",
    "its": "it's",
    "weve": "we've",
    "were": "we're",
    "wed": "we'd",
    "theyre": "they're",
    "theyve": "they've",
    "theyd": "they'd",
    "theyll": "they'll",
    "dont": "don't",
    "doesnt": "doesn't",
    "didnt": "didn't",
    "cant": "can't",
    "couldnt": "couldn't",
    "shouldnt": "shouldn't",
    "wouldnt": "wouldn't",
    "wont": "won't",
    "wasnt": "wasn't",
    "werent": "weren't",
    "isnt": "isn't",
    "arent": "aren't",
    "aint": "ain't",
    "hasnt": "hasn't",
    "havent": "haven't",
    "hadnt": "hadn't",
    "mustnt": "mustn't",
    "lets": "let's",
    "thats": "that's",
    "whats": "what's",
    "whos": "who's",
    "wheres": "where's",
    "heres": "here's",
    "theres": "there's",
    "whens": "when's",
    "hows": "how's",
    "whys": "why's",
    "i'm": "i'm",
    "i've": "i've",
    "i'd": "i'd",
    "i'll": "i'll",
    "you're": "you're",
    "you've": "you've",
    "you'd": "you'd",
    "you'll": "you'll",
    "he's": "he's",
    "she's": "she's",
    "it's": "it's",
    "we've": "we've",
    "we're": "we're",
    "we'd": "we'd",
    "we'll": "we'll",
    "they're": "they're",
    "they've": "they've",
    "they'd": "they'd",
    "they'll": "they'll",
    "don't": "don't",
    "doesn't": "doesn't",
    "didn't": "didn't",
    "can't": "can't",
    "couldn't": "couldn't",
    "shouldn't": "shouldn't",
    "wouldn't": "wouldn't",
    "won't": "won't",
    "wasn't": "wasn't",
    "weren't": "weren't",
    "isn't": "isn't",
    "aren't": "aren't",
    "ain't": "ain't",
    "hasn't": "hasn't",
    "haven't": "haven't",
    "hadn't": "hadn't",
    "mustn't": "mustn't",
    "let's": "let's",
    "that's": "that's",
    "what's": "what's",
    "who's": "who's",
    "where's": "where's",
    "here's": "here's",
    "there's": "there's",
    "when's": "when's",
    "how's": "how's",
    "why's": "why's",
    "gonna": "going to",
    "gotta": "got to",
    "wanna": "want to",
    "kinda": "kind of",
    "sorta": "sort of",
    "coulda": "could have",
    "shoulda": "should have",
    "woulda": "would have",
    "musta": "must have",
    "lemme": "let me",
    "gimme": "give me",
    "gotcha": "got you",
    "whatcha": "what are you",
    "ya": "you",
    "yall": "you all",
    "y'all": "you all",
    "cmon": "come on",
    "c'mon": "come on",
    "dunno": "don't know",
    "tryna": "trying to",
    "finna": "fixing to",
    "outta": "out of",
    "lotta": "lot of",
    "lotsa": "lots of",
}


def remove_urls(text):
    """Remove URLs from text."""
    url_pattern = re.compile(r'https?://\S+|www\.\S+')
    return url_pattern.sub('', text)


def remove_mentions(text):
    """Remove @mentions from text."""
    return re.sub(r'@\w+', '', text)


def handle_hashtags(text, keep_text=True):
    """Handle hashtags by removing # but keeping the word."""
    if keep_text:
        return re.sub(r'#(\w+)', r'\1', text)
    else:
        return re.sub(r'#\w+', '', text)


def expand_contractions(text):
    """Expand contractions for better tokenization."""
    words = text.split()
    expanded_words = []

    for word in words:
        word_lower = word.lower()

        if word_lower in CONTRACTIONS:
            expanded = CONTRACTIONS[word_lower]
            if word[0].isupper():
                expanded = expanded.capitalize()
            expanded_words.append(expanded)
        else:
            expanded_words.append(word)

    return ' '.join(expanded_words)


def remove_special_characters(text, keep_punctuation=True):
    """Remove special characters, optionally keeping basic punctuation."""
    if keep_punctuation:
        return re.sub(r"[^a-zA-Z0-9\s.,!?'\"-]", '', text)
    else:
        return re.sub(r'[^a-zA-Z0-9\s]', '', text)


def remove_extra_whitespace(text):
    """Remove extra whitespace and strip."""
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


def handle_emojis(text, mode='remove'):
    """Handle emojis in text."""
    if mode == 'remove':
        emoji_pattern = re.compile(
            "["
            "\U0001F600-\U0001F64F"
            "\U0001F300-\U0001F5FF"
            "\U0001F680-\U0001F6FF"
            "\U0001F1E0-\U0001F1FF"
            "\U00002702-\U000027B0"
            "\U000024C2-\U0001F251"
            "]+",
            flags=re.UNICODE
        )
        return emoji_pattern.sub('', text)
    else:
        return text


def remove_repeated_characters(text, max_repeat=2):
    """Reduce repeated characters."""
    pattern = re.compile(r'(.)\1{2,}')
    return pattern.sub(r'\1\1', text)


def clean_text(text, lowercase=False):
    """Apply all cleaning steps to text."""
    text = str(text)
    text = remove_urls(text)
    text = remove_mentions(text)
    text = handle_hashtags(text, keep_text=True)
    text = expand_contractions(text)
    text = handle_emojis(text, mode='remove')
    text = remove_special_characters(text, keep_punctuation=True)
    text = remove_repeated_characters(text)
    text = remove_extra_whitespace(text)

    if lowercase:
        text = text.lower()

    return text


def preprocess_dataset(input_path, output_path, text_column='text'):
    """Preprocess an entire dataset and save to CSV."""
    print(f"Loading data from {input_path}...")
    df = pd.read_csv(input_path)

    print(f"Found {len(df)} samples")
    print(f"Columns: {list(df.columns)}")

    df['text_original'] = df[text_column].copy()

    print("\nCleaning texts...")
    tqdm.pandas(desc="Processing")
    df[text_column] = df[text_column].progress_apply(clean_text)

    changed_mask = df[text_column] != df['text_original']
    num_changed = changed_mask.sum()
    print(f"\nTexts modified: {num_changed} ({100*num_changed/len(df):.1f}%)")

    df = df.drop(columns=['text_original'])

    df.to_csv(output_path, index=False)
    print(f"Saved cleaned data to {output_path}")

    return df


def main():
    print("DATA PREPROCESSING FOR EMOTION CLASSIFICATION")

    train_input = "data/train.csv"
    train_output = "data/train_cleaned.csv"
    val_input = "data/validation.csv"
    val_output = "data/validation_cleaned.csv"

    print("Preprocessing Data...")
    preprocess_dataset(train_input, train_output)
    preprocess_dataset(val_input, val_output)

    print(f"\nCleaned files saved to:")
    print(f"  - {train_output}")
    print(f"  - {val_output}")

if __name__ == "__main__":
    main()
