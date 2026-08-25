"""
run_ner_pipeline.py
====================
Single entry point for the Named Entity Recognition (NER) data pipeline.

Folder layout expected:
    ner_pipeline/
        run_ner_pipeline.py   <- run this file
        utils_ner.py
        data_raw/
            Chittagong_NER.csv
            Sylhet_NER.csv
        scripts/
            01_prepare_ner.py

Run with:
    cd ner_pipeline
    python run_ner_pipeline.py

Output:
    data_processed/ner/ner_combined.jsonl
    data_processed/ner/{train_ner,val_ner,test_ner}.jsonl
"""
import utils_ner

if __name__ == "__main__":
    utils_ner.run_all()
