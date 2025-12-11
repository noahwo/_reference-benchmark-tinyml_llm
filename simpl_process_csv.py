 
import math
import uuid
import ast
from typing import Dict, Any, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from IPython.display import display
from tabulate import tabulate


df = pd.read_csv("combined_tinyml_benchmark_data.csv")

def clean_tags(tags_str):
    """
    NOTE: Intentionally preserves original behavior (including the original
    '_generator' / '_batch' unconditional replace branch) so outputs are IDENTICAL.
    """
    if pd.isna(tags_str) or tags_str == '[]':
        return []
    try:
        tags_list = ast.literal_eval(tags_str)
        cleaned_list = []
        for tag in tags_list:
            if tag == 'benchmark':
                continue
            if ':latest' in tag:
                tag = tag.replace(':latest', '')
            # Preserve original logic exactly:
            if '_generator':
                tag = tag.replace('_sketch_generator', '')
            if '_batch':
                tag = tag.replace('_batch', '')
            cleaned_list.append(tag)
        return cleaned_list
    except (ValueError, SyntaxError):
        return []

def normalize_dates_inplace(df: pd.DataFrame) -> None:
    """
    Apply the same date normalization rules in-place so output matches 1:1.
    """
    df["test_date"] = df["test_date"].str.replace("07.28", "07.27")
    df["test_date"] = df["test_date"].str.replace("07.29", "07.28")
    df["test_date"] = df["test_date"].str.replace("07.30_a", "07.29")
    df["test_date"] = df["test_date"].str.replace("07.30_b", "07.30")

def initial_filtering_inplace(df: pd.DataFrame) -> None:
    """
    Drop *_dp_ / *_sg_ / *_mc_ batches; leave other behavior identical.
    """
    df.drop(df[df['batch_id'].str.contains('_dp_', na=False)].index, inplace=True)
    df.drop(df[df['batch_id'].str.contains('_sg_', na=False)].index, inplace=True)
    df.drop(df[df['batch_id'].str.contains('_mc_', na=False)].index, inplace=True)


def add_parameters_status_inplace(df: pd.DataFrame) -> None:
    df['parameters_status'] = df['parameters'].apply(lambda x: 'P' if pd.notnull(x) and str(x).strip() != '' else 'NP')

def build_category_inplace(df: pd.DataFrame) -> None:
    split_vals = df['batch_id'].str.split('_')
    part_2_3 = split_vals.str[2] + '_' + split_vals.str[3]
    df['category'] = df['model_config'].astype(str) + '_' + part_2_3 + '_' + df['parameters_status'].astype(str)
    df['category'] = df['category'].apply(lambda x: x.replace('_batch', '') if '_batch' in x else x)

def drop_and_rename_inplace(df: pd.DataFrame) -> None:
    
    df['parameters']= df['parameters_status'].apply(lambda x: 'True' if x == 'P' else 'False')  
    df['processor'] = df['tags'].apply(lambda x: 'tpusg' if 'tpu' in x else 'psg' if 'py' in x else 'unknown')
    df['name']= df['name'].str.replace('_sketch_generator', 'sg', regex=False)
    df['batch_id']= df['batch_id'].str.replace('_batch', '', regex=False)
    # df.drop(columns=[
    #     'num_run', 'source_file', 'source_path','completion_cost','prompt_cost',
    #     'total_cost', 'tags'
    # ], inplace=True)
    df['model_config']= df['model_config'].str.replace('gemma3:27b', 'gemma3', regex=False)
    df['model_config']= df['model_config'].str.replace('qwen2.5-coder:14b', 'qwen14', regex=False)
    df['model_config']= df['model_config'].str.replace('qwen2.5-coder:32b', 'qwen32', regex=False)
    df.rename(columns={"model_config": "model"}, inplace=True)



def filter_from_july_inplace(df: pd.DataFrame) -> None:
    df.drop(df[df['test_date'].str.split('.').str[0].astype(int) < 7].index, inplace=True)
 
def split_batch_id_by_test_date(df: pd.DataFrame) -> pd.DataFrame:
    """
    Preserves original randomness via uuid4; keeps identical logic/structure.
    """
    bad = df.groupby("batch_id")["test_date"].nunique()
    bad = bad[bad > 1].index
    if len(bad) == 0:
        return df

    mask = df["batch_id"].isin(bad)
    sub = df.loc[mask].copy()
    sub["new_token"] = (
        sub.groupby(["batch_id", "test_date"]).ngroup()
        .map(lambda _: uuid.uuid4().hex[:4])
    )
    parts = sub["batch_id"].str.split("_", n=2, expand=True)
    sub["batch_id"] = parts[0] + "_" + sub["new_token"] + "_" + parts[2]
    df.loc[mask, "batch_id"] = sub["batch_id"]
    return df

df = split_batch_id_by_test_date(df)
# Apply 'tags' cleaning exactly as original
df['tags'] = df['tags'].apply(clean_tags)
df.drop(df[df['test_date'].str.split('.').str[0].astype(int) < 7].index, inplace=True)
normalize_dates_inplace(df)
# Filter batch_ids exactly as original
initial_filtering_inplace(df)

# Same date normalization rules
normalize_dates_inplace(df)

# Apply transformations in the same sequence as before
add_parameters_status_inplace(df)
build_category_inplace(df)

drop_and_rename_inplace(df)
filter_from_july_inplace(df)

# put name,trace_id,batch_id,status,latency,total_tokens,prompt_tokens,completion_tokens,parameters,generation_count,timestamp,test_date,model,parameters_status,category,processor in order
df.rename(columns={"name": "run_name"}, inplace=True)
orders=['run_name','processor', 'model','parameters','status','generation_count','latency','total_tokens','prompt_tokens','completion_tokens','parameters','timestamp','test_date','category','trace_id','batch_id' ]
df = df[orders]

df.to_csv("cleaned_run_level.csv", index=False)