import os
import pandas as pd
import json
import numpy as np
from sklearn.model_selection import train_test_split

print("=" * 60)
print("SCRIPT 01 — NER Preparation (Dedicated NER Pipeline)")
print("=" * 60)

# 1. File paths and directory setup
RAW_DIR = "data_raw"
PROCESSED_DIR = "data_processed"
NER_OUT_DIR = os.path.join(PROCESSED_DIR, "ner")
os.makedirs(NER_OUT_DIR, exist_ok=True)

chit_ner_path = os.path.join(RAW_DIR, "Chittagong_NER.csv")
syl_ner_path  = os.path.join(RAW_DIR, "Sylhet_NER.csv")

# 2. Data loading and cleaning
def load_and_clean_ner(filepath, dialect_name, word_col):
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"NER file missing: {filepath}")

    print(f"\n[08] Loading {os.path.basename(filepath)} (dialect={dialect_name}) ...")
    df = pd.read_csv(filepath)
    print(f"     Raw row count: {df.shape[0]}")

    if 'Sentence #' in df.columns:
        df['Sentence #'] = df['Sentence #'].astype(str).str.strip()
        df['Sentence #'] = df['Sentence #'].replace(['nan', '', 'None', '<NA>', 'NaN'], np.nan)
        df['Sentence #'] = df['Sentence #'].ffill()
    else:
        raise ValueError(f"Column 'Sentence #' missing in {filepath}")

    df['bio_tag'] = df['bio_tag'].astype(str).str.strip()
    df[word_col]  = df[word_col].astype(str).str.strip()
    df = df[(df[word_col] != "") & (df[word_col] != "nan") & (df[word_col] != "None")]
    return df

# BUG FIX 3: BIO sequence validator — orphan I-X → B-X
def fix_bio(tags):
    fixed = list(tags)
    for i, tag in enumerate(fixed):
        if tag.startswith('I-'):
            entity = tag[2:]
            if i == 0:
                fixed[i] = 'B-' + entity
            else:
                prev = fixed[i - 1]
                if prev == 'O':
                    fixed[i] = 'B-' + entity
                elif (prev.startswith('B-') or prev.startswith('I-')) and prev[2:] != entity:
                    fixed[i] = 'B-' + entity
    return fixed

# 3. Sentence reconstruction
def rebuild_sentences(df, word_col, dialect_name):
    print(f"[*] Reconstructing sentences for {dialect_name} using groupby...")
    grouped = df.groupby('Sentence #', sort=False).agg({
        word_col:  list,
        'bio_tag': list
    }).reset_index()

    sentences = []
    for _, row in grouped.iterrows():
        if len(row[word_col]) > 0:
            sentences.append({
                "dialect": dialect_name,
                "tokens":  [str(t) for t in row[word_col]],
                "tags":    fix_bio(row['bio_tag'])
            })
    return sentences

# BUG FIX 1+2: GLOBAL dedup across BOTH dialects combined
# Keys on (tokens, tags) only — dialect field excluded intentionally.
# This catches cross-dialect identical sentences that per-dialect dedup misses.
def global_dedup(sentences):
    seen = set()
    clean = []
    for s in sentences:
        key = (tuple(s['tokens']), tuple(s['tags']))
        if key not in seen:
            seen.add(key)
            clean.append(s)
    return clean

# 4. Load and reconstruct
try:
    df_chit = load_and_clean_ner(chit_ner_path, "Chittagonian", "chittagong_word")
    chit_sentences = rebuild_sentences(df_chit, "chittagong_word", "Chittagonian")

    df_syl = load_and_clean_ner(syl_ner_path, "Sylheti", "sylhet_word")
    syl_sentences = rebuild_sentences(df_syl, "sylhet_word", "Sylheti")
except Exception as e:
    print(f"[08] ERROR: {e}")
    exit(1)

# Combine FIRST, then global dedup — catches cross-dialect identical sentences
all_sentences_raw = chit_sentences + syl_sentences
all_sentences = global_dedup(all_sentences_raw)

print(f"\n[08] Global dedup: {len(all_sentences_raw)} → {len(all_sentences)} sentences")

# Re-split by dialect for stratified splitting
chit_clean = [s for s in all_sentences if s['dialect'] == 'Chittagonian']
syl_clean  = [s for s in all_sentences if s['dialect'] == 'Sylheti']
print(f"     Chittagonian: {len(chit_clean)} | Sylheti: {len(syl_clean)}")

# 5. Stratified split per dialect (prevents cross-dialect leakage)
def split_dialect_data(sentences, random_state=42):
    if len(sentences) == 0:
        return [], [], []
    train, temp = train_test_split(sentences, test_size=0.20, random_state=random_state)
    val,  test  = train_test_split(temp,      test_size=0.50, random_state=random_state)
    return train, val, test

chit_train, chit_val, chit_test = split_dialect_data(chit_clean)
syl_train,  syl_val,  syl_test  = split_dialect_data(syl_clean)

train_data = chit_train + syl_train
val_data   = chit_val   + syl_val
test_data  = chit_test  + syl_test

# BUG FIX 2: Post-split leakage guard (safety net)
train_keys = set((tuple(d['tokens']), tuple(d['tags'])) for d in train_data)
val_data   = [d for d in val_data  if (tuple(d['tokens']), tuple(d['tags'])) not in train_keys]
test_data  = [d for d in test_data if (tuple(d['tokens']), tuple(d['tags'])) not in train_keys]

# Also guard val∩test
val_keys  = set((tuple(d['tokens']), tuple(d['tags'])) for d in val_data)
test_data = [d for d in test_data if (tuple(d['tokens']), tuple(d['tags'])) not in val_keys]

# 6. Save master combined (from clean global-deduped pool)
master_ner_path = os.path.join(NER_OUT_DIR, "ner_combined.jsonl")
with open(master_ner_path, 'w', encoding='utf-8') as f:
    for item in all_sentences:
        f.write(json.dumps(item, ensure_ascii=False) + "\n")
print(f"\n[08] Saved combined master → {master_ner_path} ({len(all_sentences)} sentences)")

# 7. Save splits
def save_subset(data_list, filename):
    out_path = os.path.join(NER_OUT_DIR, filename)
    with open(out_path, 'w', encoding='utf-8') as f:
        for item in data_list:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    return len(data_list)

train_len = save_subset(train_data, "train_ner.jsonl")
val_len   = save_subset(val_data,   "val_ner.jsonl")
test_len  = save_subset(test_data,  "test_ner.jsonl")

print(f"\n[08] Final Split Sizes:")
print(f"     train_ner.jsonl : {train_len} sentences")
print(f"     val_ner.jsonl   : {val_len} sentences")
print(f"     test_ner.jsonl  : {test_len} sentences")
print(f"     combined total  : {train_len+val_len+test_len} (matches combined: {train_len+val_len+test_len==len(all_sentences)})")
print("[08] ✓ NER pipeline complete — zero duplicates, zero leakage, zero BIO violations.")