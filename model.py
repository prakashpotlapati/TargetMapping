import pandas as pd
from sentence_transformers import SentenceTransformer, util
from transformers import BertTokenizer, BertModel
from fuzzywuzzy import fuzz
import spacy
import torch
import os
import logging
import warnings
from concurrent.futures import ThreadPoolExecutor
import time
import math

# Set up logging to filter out warnings and only show errors and info
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("sentence_transformers").setLevel(logging.ERROR)
logging.getLogger("tqdm").setLevel(logging.ERROR)
logger = logging.getLogger()

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

# Global variables for model caching
sentence_model = None
bert_tokenizer = None
bert_model = None
nlp = None
embeddings_cache = {}

def load_dataframe(file_path):
    if file_path.endswith('.csv'):
        return pd.read_csv(file_path)
    elif file_path.endswith(('.xls', '.xlsx')):
        return pd.read_excel(file_path)
    else:
        raise ValueError("Unsupported file format. Only CSV, XLS, and XLSX are allowed.")

# Load models with error handling and logging
def load_models():
    global sentence_model, bert_tokenizer, bert_model, nlp
    
    if sentence_model is None or bert_tokenizer is None or bert_model is None or nlp is None:
        try:
            logger.info("Loading models...")
            sentence_model = SentenceTransformer('all-mpnet-base-v2')
            bert_tokenizer = BertTokenizer.from_pretrained("bert-large-uncased")
            bert_model = BertModel.from_pretrained("bert-large-uncased")
            nlp = spacy.load("en_core_web_md")
            logger.info("Models loaded successfully.")
        except Exception as e:
            logger.error(f"Error loading models: {e}")
            exit(1)
    else:
        logger.info("Models already loaded. Skipping loading.")

# Function to check if the dataset is transposed
def is_transposed(df):
    def check_column_headers():
        numeric_headers = all(pd.to_numeric(df.columns, errors="coerce").notnull())
        logger.info(f"Numeric headers check: {numeric_headers}")
        return numeric_headers

    def check_data_types_variability():
        col_data_types = df.dtypes.nunique()
        row_data_types = df.T.dtypes.nunique()
        logger.info(f"Data type variability: Columns={col_data_types}, Rows={row_data_types}")
        return col_data_types < row_data_types

    def check_missing_values_pattern():
        col_missing_pattern = df.isna().sum(axis=0).tolist()
        row_missing_pattern = df.isna().sum(axis=1).tolist()
        logger.info(f"Missing values pattern: Columns={col_missing_pattern}, Rows={row_missing_pattern}")
        return sum(row_missing_pattern) > sum(col_missing_pattern)

    def inspect_sample_values():
        first_row = df.iloc[0, :].apply(str).str.isdigit().sum()
        first_column = df.iloc[:, 0].apply(str).str.isdigit().sum()
        logger.info(f"Sample value inspection: Row={first_row}, Column={first_column}")
        return first_row > first_column

    checks = [
        check_column_headers(),
        check_data_types_variability(),
        check_missing_values_pattern(),
        inspect_sample_values(),
    ]
    logger.info(f"Is dataset transposed? {all(checks)}")
    return all(checks)

# Precompute and cache embeddings with optimizations
def compute_embeddings(df, model, tokenizer=None, use_sentence_transformer=False, batch_size=32, sample_size=50):
    def batch_encode(text_batch):
        if use_sentence_transformer:
            return model.encode(text_batch, convert_to_tensor=True)
        else:
            inputs = tokenizer(text_batch, return_tensors="pt", truncation=True, padding=True, max_length=512)
            with torch.no_grad():
                outputs = model(**inputs)
            return outputs.last_hidden_state.mean(dim=1)

    embeddings_cache = {}
    text_batches = []
    for column in df.columns:
        available_rows = df[column].dropna().astype(str)
        sample_count = min(len(available_rows), sample_size)
        text_sample = " ".join(available_rows.sample(sample_count, random_state=42))
        text_batches.append(text_sample)

    logger.info("Processing embeddings in batches...")
    with ThreadPoolExecutor(max_workers=8) as executor:
        # Process the batches concurrently
        results = list(executor.map(batch_encode, text_batches))

    # Store embeddings in the cache
    for col, emb in zip(df.columns, results):
        embeddings_cache[col] = emb.squeeze()

    return embeddings_cache

# Similarity calculation functions
def calculate_similarity_fuzzy(col1, col2):
    try:
        return fuzz.ratio(col1.lower(), col2.lower())
    except Exception as e:
        logger.error(f"Error calculating Fuzzy similarity: {e}")
        return 0

def calculate_similarity_spacy(target_col, source_col):
    try:
        target_col = target_col.lower().strip()
        source_col = source_col.lower().strip()
        if not target_col or not source_col:
            return 0
        target_doc = nlp(target_col)
        source_doc = nlp(source_col)
        return target_doc.similarity(source_doc)
    except Exception as e:
        logger.error(f"Error calculating SpaCy similarity: {e}")
        return 0

def calculate_similarity_bert(emb1, emb2):
    try:
        return torch.cosine_similarity(emb1.unsqueeze(0), emb2.unsqueeze(0)).item()
    except Exception as e:
        logger.error(f"Error calculating BERT similarity: {e}")
        return 0

def calculate_similarity_sentence_transformer(emb1, emb2):
    try:
        return util.cos_sim(emb1, emb2).item()
    except Exception as e:
        logger.error(f"Error calculating SentenceTransformer similarity: {e}")
        return 0

# Match computation with cached embeddings
def get_top_matches_for_each_column(target_df, source_df, target_embeddings, source_embeddings):
    all_matches = []

    for target_col in target_df.columns:
        column_matches = []

        for source_col in source_df.columns:
            fuzzy_score = calculate_similarity_fuzzy(target_col, source_col) / 100
            spacy_score = calculate_similarity_spacy(target_col, source_col)
            bert_score = calculate_similarity_bert(target_embeddings[target_col], source_embeddings[source_col])
            sentence_score = calculate_similarity_sentence_transformer(target_embeddings[target_col], source_embeddings[source_col])

            weighted_score = (0.2 * max(spacy_score, fuzzy_score) + 0.4 * bert_score + 0.4 * sentence_score) * 100

            column_matches.append({
                "Target Column": target_col,
                "Source Column": source_col,
                "Fuzzy Score": round(fuzzy_score, 2),
                "Spacy Score": round(spacy_score, 2),
                "BERT Score": round(bert_score, 2),
                "SentenceTransformer Score": round(sentence_score, 2),
                "Weighted Average Score": round(weighted_score, 2)
            })

        column_matches = sorted(column_matches, key=lambda x: x["Weighted Average Score"], reverse=True)
        all_matches.extend(column_matches)

    return all_matches

# Main function to run the model
def run_model(source_file_path, target_file_path):
    try:
        logger.info("Checking if input files exist...")
        if not os.path.exists(target_file_path):
            raise FileNotFoundError(f"Target file '{target_file_path}' not found.")
        if not os.path.exists(source_file_path):
            raise FileNotFoundError(f"Source file '{source_file_path}' not found.")
        
        logger.info("Loading models...")
        load_models()

        logger.info("Loading data from target and source files...")
        target_df = load_dataframe(target_file_path)
        source_df = load_dataframe(source_file_path)

        if target_df.empty or source_df.empty:
            raise ValueError("One or both input datasets are empty.")
        
        # Check if datasets are transposed
        if is_transposed(target_df):
            logger.warning("Target dataset appears to be transposed. Please correct it before proceeding.")
            return
        if is_transposed(source_df):
            logger.warning("Source dataset appears to be transposed. Please correct it before proceeding.")
            return

        logger.info("Computing embeddings for target and source datasets...")
        target_embeddings_bert = compute_embeddings(target_df, bert_model, bert_tokenizer)
        source_embeddings_bert = compute_embeddings(source_df, bert_model, bert_tokenizer)
        target_embeddings_sentence = compute_embeddings(target_df, sentence_model, None, use_sentence_transformer=True)
        source_embeddings_sentence = compute_embeddings(source_df, sentence_model, None, use_sentence_transformer=True)

        logger.info("Calculating top matches for each column...")
        matches = get_top_matches_for_each_column(
            target_df,
            source_df,
            target_embeddings={**target_embeddings_bert, **target_embeddings_sentence},
            source_embeddings={**source_embeddings_bert, **source_embeddings_sentence}
        )

        output_dir = "Results"
        os.makedirs(output_dir, exist_ok=True)

        if matches:
            results_df = pd.DataFrame(matches)
            logger.info("Top matches calculated successfully.")
            results_df.to_csv(os.path.join(output_dir, "column_matching_results.csv"), index=False)
            logger.info("Results saved successfully.")
        else:
            logger.warning("No matches found.")
    except Exception as e:
        logger.error(f"Error in model processing: {e}")
        raise