import argparse
import json
import os
import time
from pprint import pprint
from typing import (
    Dict,
    List,
)

import numpy as np
import pandas as pd
import ray
from tqdm import tqdm

from ray_bench_utils import (
    eval_source_query_d_batch,
    get_result,
    iter_batches,
    read_file,
    run_parallel_ray,
    source_query_env_path,
)

'''
## To debug the env, test if this works. 
## The output should not contain any errors.

import nbformat
import tempfile
x = nbformat.v4.new_notebook()
x['cells'].append(nbformat.v4.new_code_cell("""
import sys
import os

print("--- KERNEL INTROSPECTION ---")
print(f"KERNEL EXECUTABLE: {sys.executable}")
print(f"KERNEL CWD: {os.getcwd()}")
print(f"KERNEL SYS.ARGV: {sys.argv}")
print("--- END KERNEL INTROSPECTION ---")

import tensorflow as tf
"""))

from nbclient import NotebookClient
with tempfile.TemporaryDirectory() as temp_dir:
    client = NotebookClient(
        x, 
        timeout=600, 
        kernel_name='python', 
        resources={'metadata': {'path': temp_dir}}, 
        allow_errors=True
    )
    client.execute()
for cell in x['cells']:
    print(cell['source'])
    print(); print('-'*80, end='\n')
    for output in cell.get('outputs', []):
        if output['output_type'] == 'stream':
            print(output['text'], end='')
        elif output['output_type'] == 'error':
            print(f"ERROR: {output['ename']}: {output['evalue']}")
        else:
            print(output)
    print(); print('*'*80, end='\n')
'''


def main(
    *,
    source_path: str,
    bench_env_dir: str,
    true_cache_dir: str,
    gen_cache_dir: str,
    vis_cache_dir: str,
    output_file: str,
    batch_size: int = 1,
    run_using: str = "threads",  ## "processes"
    cpus_per_query: int = 3,
    gpus_per_query: int = 0,
    seed: int = 42,
):
    os.makedirs(bench_env_dir, exist_ok=True)
    os.makedirs(true_cache_dir, exist_ok=True)
    os.makedirs(gen_cache_dir, exist_ok=True)
    os.makedirs(vis_cache_dir, exist_ok=True)

    source_queries: List[Dict] = json.loads(read_file(source_path))
    for source_query_d in source_queries:
        source_query_d["env_path"] = source_query_env_path(
            bench_env_dir, source_query_d=source_query_d
        )

    results: List[Dict] = []
    if os.path.exists(output_file):
        results: List[Dict] = json.load(open(output_file))

    source_queries_to_process: List[Dict] = [
        q for q in source_queries if q["uid"] not in {r["uid"] for r in results}
    ]

    ray.init(
        address="auto",
        ignore_reinit_error=True,
    )
    print("Ray initialized. Cluster info:")
    pprint(ray.cluster_resources())
    RAY_NUM_CPUS = ray.cluster_resources().get("CPU", 0)
    RAY_NUM_GPUS = ray.cluster_resources().get("GPU", 0)

    start_time = time.time()
    futs = [
        run_parallel_ray(
            eval_source_query_d_batch,
            jdict_batch=source_query_d_batch,
            true_cache_dir=true_cache_dir,
            gen_cache_dir=gen_cache_dir,
            vis_cache_dir=vis_cache_dir,
            skip_test=False,
            temp_caching=False,
            min_diff_only=False,
            run_using=run_using,
            num_cpus=cpus_per_query,
            num_gpus=gpus_per_query,
        )
        for source_query_d_batch in iter_batches(
            np.random.RandomState(seed=seed)
            .permutation(source_queries_to_process)
            .tolist(),
            batch_size=batch_size,
        )
    ]

    for fut in tqdm(futs, desc="Collecting"):
        try:
            res = get_result(fut)
            results.extend(res)
        except Exception as e:
            results.extend([e])

    end_time = time.time()
    print(f"Execution completed in {end_time - start_time:.2f} seconds.")

    print(
        pd.Series(
            [res["processing_test"]["pro_success"] for res in results]
        ).value_counts()
    )
    print(
        pd.Series(
            [res["visualization_test"]["vis_success"] for res in results]
        ).value_counts()
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Execution Engine for AstroCodeBench")

    default_source_path: str = os.path.join(
        os.getenv("WORK"),
        "AstroVisBench",
        "astrovisbench_queries.json",
    )
    default_output_file: str = os.path.join(
        os.getenv("WORK"),
        "AstroVisBench",
        "astrovisbench_output.json",
    )
    default_bench_env_dir: str = (
        os.path.join(
            os.getenv("WORK"),
            "AstroVisBench",
            "bench_environment",
        )
        + os.path.sep
    )
    default_true_cache_dir: str = (
        os.path.join(
            os.getenv("WORK"),
            "AstroVisBench",
            "true_cache_dir",
        )
        + os.path.sep
    )
    default_gen_cache_dir: str = (
        os.path.join(
            os.getenv("WORK"),
            "AstroVisBench",
            "gen_cache_dir",
        )
        + os.path.sep
    )

    default_vis_cache_dir: str = (
        os.path.join(
            os.getenv("WORK"),
            "AstroVisBench",
            "vis_cache_dir",
        )
        + os.path.sep
    )

    parser.add_argument(
        "source",
        type=str,
        help="Source queries JSON file path",
        default=default_source_path,
    )
    parser.add_argument(
        "env",
        type=str,
        help="Path to execution environment directory",
        default=default_bench_env_dir,
    )
    parser.add_argument(
        "--true-cache",
        type=str,
        help="Path to cache directory for ground truth query processing",
        default=default_true_cache_dir,
    )
    parser.add_argument(
        "--gen-cache",
        type=str,
        help="Path to cache directory of generated code",
        default=default_gen_cache_dir,
    )
    parser.add_argument(
        "--vis-cache",
        type=str,
        help="Path to cache directory of generated visualization notebooks",
        default=default_vis_cache_dir,
    )
    parser.add_argument(
        "--outfile",
        type=str,
        help="Output JSON file path to store results",
        default=default_output_file,
    )
    parser.add_argument(
        "--batch-size", type=int, default=1, help="Batch size for processing queries"
    )
    parser.add_argument(
        "--run-using",
        choices=["threads", "processes"],
        default="threads",
        help="Execution mode: threads or processes",
    )
    parser.add_argument(
        "--cpus-per-query",
        type=int,
        default=3,
        help="Number of CPUs to allocate per query",
    )
    parser.add_argument(
        "--gpus-per-query",
        type=int,
        default=0,
        help="Number of GPUs to allocate per query",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for query shuffling"
    )

    args = parser.parse_args()

    main(
        source_path=args.source,
        bench_env_dir=args.env,
        true_cache_dir=args.true_cache,
        gen_cache_dir=args.gen_cache,
        vis_cache_dir=args.vis_cache,
        output_file=args.outfile,
        batch_size=args.batch_size,
        run_using=args.run_using,
        cpus_per_query=args.cpus_per_query,
        gpus_per_query=args.gpus_per_query,
        seed=args.seed,
    )
