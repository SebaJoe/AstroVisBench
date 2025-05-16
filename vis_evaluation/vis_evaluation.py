from google import genai
import base64
import pandas as pd
import os
import csv
import json
import tkinter as tk
from tkinter import ttk
import sys
import html
import pprint
import anthropic
from PIL import Image
import re
from glob import glob
from tqdm import tqdm
import copy
import argparse

prompts_csv = 'viseval_prompts.csv'
model = "claude-3-5-sonnet-20240620"
client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

def safe_parse_json_from_claude(text):
    match = re.search(r'(\{\s*"Rationale"\s*:\s*".*?"\s*,\s*"Errors"\s*:\s*".*?"\s*\})', text, re.DOTALL)
    if not match:
        return "", ":x: No JSON block found"
    json_block = match.group(1)
    try:
        clean_block = re.sub(r'(?<!\\)[\x00-\x1F]', '', json_block)
        parsed = json.loads(clean_block)
        rationale = parsed.get("Rationale", "").replace("\n", " ").strip()
        errors = parsed.get("Errors", "").replace("\n", " ").strip()
        return rationale, errors
    except Exception as e:
        return "", f":x: JSON parse error: {e}"
    

def query_model(prompt, gold_img, gen_img, gold_code, gen_code, vis_query, max_tokens=2024):
    print("Querying...")
    # with open(gold_img, "rb") as f:
    #     gold_base64 = base64.b64encode(f.read()).decode("utf-8")
    # with open(gen_img, "rb") as f:
    #     gen_base64 = base64.b64encode(f.read()).decode("utf-8")
    gold_base64 = gold_img
    gen_base64 = gen_img
    completion = client.messages.create(
        model=model,
        max_tokens=4096,
        system="You are a helpful assistant evaluating visualization.",
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "text", "text": "This is the Visualization Query: "},
                    {"type": "text", "text": vis_query},
                    {"type": "text", "text": "This is the **ground-truth (reference)** image: "},
                    {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": gold_base64}},
                    {"type": "text", "text": "This is the code for the correct ground-truth image: "},
                    {"type": "text", "text": gold_code},
                    {"type": "text", "text": "This is the **under-test (assessed)** image:"},
                    {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": gen_base64}},
                    {"type": "text", "text": "This is the code for the under-test generated image: "},
                    {"type": "text", "text": gen_code},
                ]
            }
        ],
    )
    return completion


def extract_prompts(filename):
    prompts = []
    try:
        with open(filename, mode="r", encoding="utf-8") as file:
            reader = csv.DictReader(file)
            for row in reader:
                if "viseval new prompt" in row:
                    prompts.append(row["viseval new prompt"])
    except Exception as e:
        print(f":x: Error reading prompts: {e}")
    return prompts


def do_vis_eval(all_queries):
    prompt_template = extract_prompts(prompts_csv)[0]
    new_queries = []
    for query in tqdm(all_queries):

        if not query['visualization_test']['vis_success'] or len(query['visualization_test']['vis_gen_list']) != 1:
            
            copy_query = copy.deepcopy(query)
            copy_query['visualization_llm_eval'] = {
                "rationale": "",
                "errors": "Crash" if not query['visualization_test']['vis_success'] else "VisFail",
            }
            new_queries.append(copy_query)
            continue

        try:
            response = query_model(
                prompt_template,
                query["gt_visualization"].split('base64,')[1],
                query['visualization_test']['gen_vis_list'][0].split('base64,')[1],
                query["visualization_gt_code"],
                query["visualization_gen_code"],
                query["visualization_query"]
            )
            raw_text = response.content[0].text
            rationale, errors = safe_parse_json_from_claude(raw_text)
        except Exception as e:
            rationale = ""
            errors = f":x: Exception: {e}"
        copy_query = copy.deepcopy(query)
        copy_query['visualization_llm_eval'] = {
            "rationale": rationale,
            "errors": errors
        }
        new_queries.append(copy_query)
    return new_queries


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('filename')
    parser.add_argument('outfile')
    args = parser.parse_args()
    in_queries = json.load(open(args.filename))
    out_queries = do_vis_eval(in_queries)
    with open(args.outfile, 'w') as f:
        json.dump(out_queries, f, indent=4)
if __name__ == "__main__":
    main()

