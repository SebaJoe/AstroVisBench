from llmchain import *

from copy import deepcopy
import json

from tqdm import tqdm
import argparse

import nltk
import time
import os

import threading

from concurrent.futures import ThreadPoolExecutor, as_completed

def remove_ticks(code):

    code_lines = code.split("\n")
    new_code_lines = [line for line in code_lines if not line.startswith("```") and not "<CELL END>" in line] 
    return "\n".join(new_code_lines)


def generate_code(jdict, model, library, generate_pro=True, generate_vis=True, add_pro_uds=True, generate_vis_loose=False, add_vis_uds=False):


    generate_model = llmchain(model=model, library=library)

    sys_prompt = """
You are tasked with completing a jupyter notebook about astronomy. You will be given some markdown cells and some python code cells for context, and in response you must output only python code that accurately fulfills the goals of the notebook as described by the markdown text.
    """

    generate_model.set_sys_message(sys_prompt)

    def generate_code_pro(jdict):
        

        context = [jdict['setup_query'], 
                   '```\n\n' + jdict['setup_gt_code'] + '\n\n```',
                   jdict['processing_query'], 
                   ]
        
        if add_pro_uds:
            context.append(jdict['processing_underspecifications'])

        out = generate_model.chat('\n\n'.join(context))

        return out

    def generate_code_vis(jdict):
        

        context = [jdict['setup_query'], 
                   '```\n\n' + jdict['setup_gt_code'] + '\n\n```',
                   jdict['processing_query'],
                   '```\n\n' + jdict['processing_gt_code'] + '\n\n```',
                   jdict['visualization_query'],
                   ]
    
        if add_vis_uds:
            context.append(jdict['visualization_underspecifications'])
        
        out = generate_model.chat('\n\n'.join(context))

        return out
    

    gen_code_pro = None
    gen_code_vis = None

    if generate_pro:
        while gen_code_pro is None:
            gen_code_pro = generate_code_pro(jdict)

    
    if generate_vis or (generate_vis_loose and jdict['visualization_gen_code'] == ""):
        while gen_code_vis is None:
            gen_code_vis = generate_code_vis(jdict)

    ret_dict = deepcopy(jdict)
    if generate_pro:
        ret_dict['processing_gen_code'] = remove_ticks(gen_code_pro)
    
    if generate_vis or (generate_vis_loose and jdict['visualization_gen_code'] == ""):
        ret_dict['visualization_gen_code'] = remove_ticks(gen_code_vis)

    return ret_dict

def match_and_fill(source, to_fill):

    ret_to_fill = []

    for q in to_fill:

        source_q = [sq for sq in source if sq['setup_query'] == q['setup_query']][0]
        
        q['processing_gen_code'] = source_q['processing_gen_code']
        q['visualization_gen_code'] = source_q['visualization_gen_code']

        ret_to_fill.append(q)

    return ret_to_fill

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument('filename')
    parser.add_argument('outfile')
    parser.add_argument('model_name')
    parser.add_argument('library')
    parser.add_argument('--generate-pro-only', action='store_true')
    parser.add_argument('--generate-vis-only', action='store_true')
    parser.add_argument('--generate-vis-loose', action='store_true')
    parser.add_argument('--staple-results')

    args = parser.parse_args()

    queries = json.load(open(args.filename))

    if args.staple_results:

        stapler = json.load(open(args.staple_results))
        queries = match_and_fill(stapler, queries)

    gen_queries = json.load(open(args.outfile)) if os.path.exists(args.outfile) else []
    gen_queries_setups = [q['setup_query'] for q in gen_queries]

    gen_tuple = (True, True)

    if args.generate_pro_only:
        gen_tuple = (True, False)
    elif args.generate_vis_only:
        gen_tuple = (False, True)
    
    if args.generate_vis_loose:
        gen_tuple = (gen_tuple[0], False)

    lock = threading.Lock()

    queries_to_generate = [
        query for query in queries
        if query['setup_query'] not in gen_queries_setups
    ]

    def worker(query):

        result = generate_code(
            query,
            args.model_name,
            args.library,
            generate_pro=gen_tuple[0],
            generate_vis=gen_tuple[1],
            generate_vis_loose=args.generate_vis_loose
        )
        with lock:
            gen_queries.append(result)
            with open(args.outfile, 'w') as f:
                json.dump(gen_queries, f, indent=4)
        return result


    with ThreadPoolExecutor(max_workers=20) as executor:
        list(tqdm(executor.map(worker, queries_to_generate), total=len(queries_to_generate)))

if __name__ == "__main__":
    main()