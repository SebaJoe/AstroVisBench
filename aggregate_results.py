import json
import argparse
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('results_file')
parser.add_argument('json_outfile')
from collections import Counter

args = parser.parse_args()

all_res = json.load(open(args.results_file))

process_wo_error = [q['processing_test']['pro_success'] for q in all_res]

vis_wo_error = [q['visualization_test']['vis_success'] for q in all_res]

processing_uw_score_exclude_error = [q['processing_test']['inspection_results']['agg_scores']['unweighted'] for q in all_res if len(q['processing_test']['inspection_results']['results'].keys()) and q['proces\
sing_test']['pro_success']]

vis_eval_agg = [q['visualization_llm_eval']['errors'] for q in all_res]


avg_pro_err = np.mean(process_wo_error)
avg_vis_err = np.mean(vis_wo_error)

avg_vi_score = np.mean(processing_uw_score_exclude_error)

vis_counts = Counter(vis_eval_agg)

results_dict = {
    "per_success_pro": avg_pro_err,
    "per_success_vis": avg_vis_err,
    "avg_vi_score": avg_vi_score,
    "vis_distrib": dict(vis_counts),
}

with open(args.json_outfile, 'w') as f:
    json.dump(results_dict, f, indent=4)