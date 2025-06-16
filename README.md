<p align="center">
  <a href="https://astrovisbench.github.io">
    <img src="imgs/avb_logo.svg" style="height: 10em" alt="big dipper and polaris" />
  </a>
</p>

<p align="center"><strong>[&nbsp;<a href="https://astrovisbench.github.io">Website & Leaderboard</a>&nbsp;]</strong></p>
<br>

![AstroVisBench](imgs/overview-1.png)

This is the repository containing the code needed to run the AstroVisBench benchmark as detailed in the paper ["AstroVisBench: A Code Benchmark for Scientific Computing and Visualization in Astronomy"](https://arxiv.org/abs/2505.20538).

The benchmark is available [here](https://huggingface.co/datasets/sebajoe/AstroVisBench) as a huggingface dataset. 

However, we highly recommend converting this benchmark into a JSON file in order to be used in along with scripts in this repository. You can find this raw JSON file [here](https://utexas.box.com/s/2evj5cs3u2gqndvgc9sd66cmlggl9fg1) under `astrovisbench_queries.json`.


## Environment Setup (UT Austin TACC Cluster)

### Step 1: One-time environment and data setup:

If using the UT Austin TACC clusters (Vista or Stampede3), do the following: 
1. First navigate to work directory and create a directory for the benchmark:
   ```bash
   cd $WORK
   mkdir -p $WORK/AstroVisBench
   ```
2. Then run the corresponding `tacc_*_install_packages.sh` script to set up the environment on the corresponding cluster. 
  - The conda environment will be set up in `$WORK/AstroVisBench-env` directory.
  - The conda environment is large, so it may take 10-20 minutes to set up.
3. After the conda environment is set up, you will need to unzip the `bench_env.tar.gz` file from the UT box folder above into the `$WORK/AstroVisBench` directory. 
   - You can do this by running: 
     ```bash
     tar -xzf bench_env.tar.gz -C $WORK/AstroVisBench
     ```
   - This will create a `bench_env` directory inside `$WORK/AstroVisBench`, which contains the necessary files for running the benchmark.
4. You will also need to download the `astrovisbench_queries.json` file from the UT box folder above and place it in the `$WORK/AstroVisBench` directory.

### Step 2: Provisioning resources to run the benchmark:
The following steps are to be done every time you want to run the benchmark on the UT Austin TACC clusters:

5. Run ONE of the following idev commands to start a TACC interactive session.
   - The `idev` command will allocate resources for you to run the benchmark interactively.
   - Make sure to activate the conda environment after starting the interactive session.
  ```bash
  ## Vista
  idev -p gh-dev -N 2 -n 2 -t 00:30:00
  conda activate $WORK/AstroVisBench-env

  ## Stampede3
  idev -N 3 -n 3 -t 01:00:00 
  conda activate $WORK/AstroVisBench-env
  ```
6. Finally, run the corresponding `tacc_*_ray_cluster_start` script to set up the Ray cluster.


## Running the Benchmark

The benchmark is a JSON file of a list-of-dicts. One dict corresponds to one query. 

Each dict has the following schema:

```
| - uid : Unique identifier for the query
| - setup_query : Natural-language query (i.e. an instruction) for setting up imports and environments
| - setup_gt_code : Initial code for setting up imports and environments
| - processing_query : Natural-language query (i.e. an instruction) for processing and analyzing data prior to visualization
| - processing_gt_code : Ground truth code associated with the processing query
| - visualization_query : Natural-language query (i.e. an instruction) for visualizing results
| - visualization_gt_code : Ground truth code associated with the visualization query
| - processing_underspecifications : clarifications for underspecified portions of the processing query
| - visualization_underspecifications : clarifications for underspecified portions of the visualization query
| - gt_visualization: ground truth visualization in base64 form
| - processing_gen_code : (TO BE FILLED OUT BY LLM) - generated code responding to the processing query. This is the code which will be evaluated.
| - visualization_gen_code : (TO BE FILLED OUT BY LLM) - generated code responding to the visualization query. This is the code which will be evaluated.
```

The purpose of this benchmark is to evaluate a target LLM's ability to generate code that can process and visualize astronomical data. 
So, you will need to use your target LLM to fill out `processing_gen_code` and `visualization_gen_code` fields; these will contain the code to be evaluated.
The helper file `generate_code/generate_code.py` can be used to generate these code fields (you can also do it yourself, just make sure it lies in the queries JSON file).

This is the setup we used to evaluate LLM in the AstroVisBench paper:
- **Processing:** Setup Query, Ground Truth Setup Code, Processing Query + Processing Underspecifications
- **Visualization:** Setup Query, Ground Truth Setup Code, Processing Query + Processing Underspecifications, Ground Truth Processing Code, Visualization Query

### Execution

After you have properly setup your environment and finished filling out the benchmark, you can start the evaluation process. 
This involves running the `ray_exec_bench.py` script. 

First, activate the environment you set up earlier. If you are using the UT Austin TACC clusters, you can do this by running:

```bash
conda activate $WORK/AstroVisBench-env
```

Here are the arguments you need to specify:

```
python ray_exec_bench.py \
--source <BENCHMARK FILLED WITH LLM-GENERATED CODE> \  ## astrovisbench_queries.json from the UT box folder above. IMPORTANT: this needs to be filled out with LLM-generated code first!
--env <PATH TO BENCH ENVIRONMENT DIRECTORY> \   ## Unzipped bench_env.tar.gz from the UT box folder above
--true-cache <PATH TO GROUND TRUTH CACHE DIRECTORY> \
--gen-cache <PATH TO GENERATED CODE PROCESSING CACHE DIRECTORY> \
--vis-cache <PATH TO GENERATED VISUALIZATION CACHE DIRECTORY> \
--outfile <PATH TO OUTPUT JSON FILE>
```

For example:
```
python ray_exec_bench.py \
--source astrovisbench_queries.json \
--env bench_environment \
--true-cache cache/true_cache \
--gen-cache cache/gen_cache \
--vis-cache cache/vis_cache \
--outfile output.json
```

There are other arguments but you can leave them as-is.

### Processing Evaluation

The variable inspection test is performed in tandem with the code execution. The results of this test is located in the output json under `processing_test`. 
- This contains the execution output, whether or not the code executed without error, and the results of the test under `inspection_results`. 
- Under `agg_scores` you will find the unweighted and weighted (by variable depth) scores for this task. We also provide you with the detailed breakdown of how each variable was compared and the distance scores attached to each comparison. 

### Visualization Evaluation

After you are done with the "Execution using Ray" steps, you can do the visualization evaluation with our LLM-as-a-judge method. 
Use the `vis_evaluation.py` script under the `vis_evaluation` directory to do so as follows:

```
python vis_evaluation.py \
<EXECUTED BENCHMARK JSON> \
<PATH TO OUTPUT JSON FILE> \
```

You will get three trials of error categorization and rationale output from the LLM for each query.  

### Results Aggregation

You can aggregate all the results after passing you benchmark json file through all these pipelines using the `aggregate_results.py`. This script will output a JSON dict containing the execution success rates for both the processing and visualization tasks, the average variable inspection scores, and the distribution of errors from the visualization evaluation.

## Citation & License

We release the benchmark under the [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/deed.en) license.

Please use the following citation if you found our work to be useful in your work.

```
@misc{joseph2025astrovisbenchcodebenchmarkscientific,
      title={AstroVisBench: A Code Benchmark for Scientific Computing and Visualization in Astronomy}, 
      author={Sebastian Antony Joseph and Syed Murtaza Husain and Stella S. R. Offner and Stéphanie Juneau and Paul Torrey and Adam S. Bolton and Juan P. Farias and Niall Gaffney and Greg Durrett and Junyi Jessy Li},
      year={2025},
      eprint={2505.20538},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2505.20538}, 
}
```
