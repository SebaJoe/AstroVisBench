## This assumes you are inside an idev env using Vista TACC cluster, via the following command:
## idev -p gh-dev -N 2 -n 2 -t 00:20:00  ## Wait for the interactive session to start
## conda activate AstroVisBench-env

# --- Setup Ray Cluster ---
# Determine GPUs per node (if SLURM provides this info; default to 1)
gpus_per_node=${SLURM_GPUS_PER_NODE:-1}
# Get the node names
nodes=$(scontrol show hostnames "$SLURM_JOB_NODELIST")
nodes_array=($nodes)
head_node=${nodes_array[0]}
echo "Initializing head node setup..."
head_node_ip=$(srun --nodes=1 --ntasks=1 -w "$head_node" hostname --ip-address)

if [[ "$head_node_ip" == *" "* ]]; then  
    IFS=' ' read -ra ADDR <<<"$head_node_ip"  # Handle IPv6 case if necessary
    if [[ ${#ADDR[0]} -gt 16 ]]; then
        head_node_ip=${ADDR[1]}
    else
        head_node_ip=${ADDR[0]}
    fi
    echo "Detected IPv6, using IPv4 address: $head_node_ip"
fi

port=6379
ip_head=$head_node_ip:$port
export ip_head
echo "Ray Head Node Address: $ip_head"
# Cleanup previous Ray instances
echo "Stopping Ray head node on $head_node..."
srun --nodes=1 --ntasks=1 -w "$head_node" ray stop --force && rm -rf /tmp/ray/* 

worker_num=$((SLURM_JOB_NUM_NODES - 1))
for ((i = 1; i <= worker_num; i++)); do
    node_i=${nodes_array[$i]}
    echo "Stopping Ray worker on node: $node_i"
    srun --nodes=1 --ntasks=1 -w "$node_i" ray stop --force && rm -rf /tmp/ray/*
done

# Start the head node with all GPUs available on the node
echo "Starting Ray head node on $head_node..."
srun --nodes=1 --ntasks=1 -w "$head_node" \
    ray start --head --node-ip-address="$head_node_ip" --port=$port \
    --num-cpus=72 --num-gpus=$gpus_per_node --block &
sleep 5  # Give head node some time to initialize

# Start worker nodes
worker_num=$((SLURM_JOB_NUM_NODES - 1))
for ((i = 1; i <= worker_num; i++)); do
    node_i=${nodes_array[$i]}
    echo "Starting Ray worker on node: $node_i"
    srun --nodes=1 --ntasks=1 -w "$node_i" \
        ray start --address "$ip_head" \
        --num-cpus=72 --num-gpus=$gpus_per_node --block &
    sleep 5
done
sleep 5  # Allow time for the Ray cluster to stabilize
ray status
