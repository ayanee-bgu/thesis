# Skill: Adapt and Submit Experiments to the BGU HPC Cluster

## Trigger
When the user wants to run/adapt an `experiments/{name}/` POC on the BGU CIS Slurm
cluster (`slurm.bgu.ac.il`), write or debug an `sbatch` file, request GPU/CPU
resources, set up a job array sweep, or troubleshoot a Slurm job.

## Cluster Facts (2024 HPC, BGU CIS)
- Login node: `slurm.bgu.ac.il` (SSH with BGU credentials). **Never** run compute
  there — it's shared for submitting/monitoring only.
- Scheduler: Slurm. Jobs run on GPU or CPU compute nodes; a job = an allocation of
  resources for a limited time.
- Conda/Anaconda3 is preinstalled — never reinstall it. Create environments on the
  **manager node**: `conda create -n <env> python=3.x`.
- Two ways to run code:
  - **Non-interactive** (`sbatch script.sbatch`) — fire-and-forget, output to a log file. Use this for real experiment runs.
  - **Interactive** (`sinteractive [--gpu 1]`, or `srun --jobid=<id> --pty bash` to reattach) — for debugging, env setup, or IDE remoting (PyCharm/VS Code point at the **compute node's** hostname, never the manager node's).
- Resource etiquette (this is a shared cluster):
  - 1 GPU per job max without asking.
  - Minimum RAM you actually need (check with `sacct -j <id> --format=JobName,MaxRSS` after a run); don't ask for more than 24G per GPU without reason, never >58G without IT.
  - Don't set `--cpus-per-task` when allocating a GPU (auto-assigned).
  - `scancel <job_id>` as soon as you're done, including interactive sessions you're stepping away from.
- Never submit thousands of tiny jobs — use job arrays instead (scheduler + I/O
  overhead). Write logs to the compute node's local `/scratch` (via
  `$SLURM_SCRATCH_DIR`, requires `#SBATCH --tmp=<size>`) and copy final results
  back to `$SLURM_SUBMIT_DIR` at the end, if a job is I/O heavy.

## Workflow

### Phase 1: Assess the experiment
1. Read `experiments/{name}/main.py`, `config.yaml`, `requirements.txt` to find:
   - Does it need a GPU (torch/tensorflow/local LLM inference)? → request `--gpus=1`.
   - Does it launch a **local model server** (e.g. Ollama, vLLM) that the script
     talks to over HTTP (check for `ollama_url` / `localhost:PORT` patterns)? If
     so, the server process and the Python script must run **in the same job, on
     the same compute node** — start the server in the sbatch file, poll until
     it's ready, then run the script, then kill the server.
   - Does it need external data (HuggingFace `datasets.load_dataset`, an
     `HF_TOKEN`, S3, Google Drive)? Compute nodes have outbound internet, so this
     works, but confirm secrets (`.env`, `HF_TOKEN`) are already on the cluster
     home directory — don't hardcode tokens into the sbatch script.
   - Is it a single run, or does it sweep a parameter (conditions, seeds, K
     values)? → candidate for `--array`.

### Phase 2: Environment
1. SSH to `slurm.bgu.ac.il`, `conda deactivate` if active.
2. Create/activate the env from `experiments/{name}/requirements.txt`:
   ```
   conda create -n <env_name> python=3.10
   conda activate <env_name>
   pip install -r experiments/{name}/requirements.txt
   ```
   Use `pip install torch ... --index-url https://download.pytorch.org/whl/cuXXX`
   for GPU pytorch, and `pip install 'tensorflow[and-cuda]'` for GPU tensorflow —
   never `pip install tensorflow-gpu` (use `conda install -c anaconda tensorflow-gpu`
   if truly needed).
3. `conda deactivate` before submitting (sbatch jobs activate the env themselves).

### Phase 3: Write the sbatch file
Put it at `experiments/{name}/cluster/run.sbatch` (create the `cluster/` subdir).
Base template:

```bash
#!/bin/bash
#SBATCH --partition main
#SBATCH --time 0-04:00:00
#SBATCH --job-name {name}
#SBATCH --output {name}-%J.out
#SBATCH --gpus=1                      ### omit / set 0 if no GPU needed
#SBATCH --mail-user=<bgu_user>@post.bgu.ac.il
#SBATCH --mail-type=BEGIN,END,FAIL

echo "SLURM_JOBID=$SLURM_JOBID"
echo "SLURM_JOB_NODELIST=$SLURM_JOB_NODELIST"

module load anaconda
source activate <env_name>

# --- If the experiment needs a local LLM server (e.g. Ollama), start it here ---
# ollama serve --port 11435 &
# SERVER_PID=$!
# until curl -s http://localhost:11435 >/dev/null; do sleep 2; done

cd "$SLURM_SUBMIT_DIR"
python experiments/{name}/main.py --num-tasks 5

# kill $SERVER_PID   # if a server was started above
```

Adjust `--time` generously above the expected runtime (long LLM inference loops
with retries can take hours) but not past the partition limit (7 days).

### Phase 4: Sweeps via job arrays
If the experiment takes a CLI flag to vary (e.g. `--condition`, `--max-questions`,
a seed), prefer one array job over N separate `sbatch` calls:

```bash
#SBATCH --array=0-3
#SBATCH --output {name}-%A_%a.out
```
```bash
CONDITIONS=(no_memory graph_session full_handoff interactive_oracle)
python experiments/{name}/main.py --condition "${CONDITIONS[$SLURM_ARRAY_TASK_ID]}"
```
Limit concurrency with `--array=0-3%2` if the sweep would otherwise grab too many
GPUs at once.

### Phase 5: Submit and monitor
```
sbatch experiments/{name}/cluster/run.sbatch      # submit
squeue --me                                       # check status (ST: PD/R)
less {name}-<jobid>.out                           # tail output
sacct -j <jobid> --format=JobName,MaxRSS,State,Elapsed,ExitCode
scancel <jobid>                                   # cancel if needed
```
Pending jobs: check `REASON` in `squeue --me` — `Resources` (cluster busy),
`Priority` (queued behind others), `PartitionTimeLimit` (fix `--time`).

### Phase 6: Pull results back
Results/logs land in `experiments/{name}/results/` or wherever `config.yaml`
points (shared storage — visible from the manager node too). Copy to local
machine with WinSCP/`scp`, or just read them directly next time you SSH in for
`analyze.py`.

## Quality Checks
- [ ] `--gpus`, `--mem`, `--time` match what the experiment actually needs (check
      `sacct ... MaxRSS` after a first run and tighten)
- [ ] Any local model server started in the job is torn down (or the job just
      ends, releasing it) — don't leave GPUs allocated idle
- [ ] Job array used instead of many individual `sbatch` calls for sweeps
- [ ] Output/log filenames include `%J` or `%A_%a` so parallel runs don't clobber
      each other
- [ ] Secrets (`HF_TOKEN`, API keys) come from `.env`/env vars already on the
      cluster, never pasted into the sbatch script
