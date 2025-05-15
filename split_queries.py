import json
import random
import argparse


parser = argparse.ArgumentParser()
parser.add_argument('infile')
parser.add_argument('prefix')
parser.add_argument('split_count')

args = parser.parse_args()



ffq_all = json.load(open(args.infile))

prefix  = args.prefix

def split_equal(lst, n):
    k, m = divmod(len(lst), n)
    i = 0
    for _ in range(n):
        chunk = lst[i : i + k + (1 if m > 0 else 0)]
        yield chunk
        i += k + (1 if m > 0 else 0)
        m -= 1

n = 10 if args.split_count is None else int(args.split_count)

random.shuffle(ffq_all)

parts = split_equal(ffq_all, n)

for i, part in enumerate(parts):

    with open(f'{prefix}{i}.json', 'w') as f:
        json.dump(part, f, indent=4)


        
