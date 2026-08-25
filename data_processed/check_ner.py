import json

base = r"D:\bangla_dialect_nlp\data_processed\ner"

for split in ["train_ner", "val_ner", "test_ner"]:
    total, syl, chit = 0, 0, 0
    with open(f"{base}\\{split}.jsonl", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            total += 1
            if obj["dialect"] == "Sylheti":
                syl += 1
            else:
                chit += 1
    print(f"{split}: total={total} | Sylheti={syl} | Chittagonian={chit}")