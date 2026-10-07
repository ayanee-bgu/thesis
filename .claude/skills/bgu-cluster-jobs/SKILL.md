---
name: bgu-cluster-jobs
description: Submit, check, and monitor SLURM jobs for this project on the BGU ISE/CS/DT cluster (slurm.bgu.ac.il). Use whenever the user asks to run experiments on the cluster, check job status, see if runs finished, investigate a failure, or resume monitoring a previously-submitted job.
---

# BGU cluster: submitting and monitoring jobs

> Adapted from the `graph-diffusion-tabpfn` project. The operational knowledge
> (VPN gate, line-ending trap, tqdm `\r`, `sacct` tallying, QoS reasons,
> monitoring pattern) carries over verbatim. The project-specific coordinates
> below are **placeholders** — fill them in the first time this project actually
> runs on the cluster, then delete this note.

Project home on the cluster: `<TBD — e.g. /home/<user>/thesis/<experiment>>`
Conda env: `<TBD>`
SSH alias `bgu` is assumed configured (`ssh bgu` connects with a key, no
password). Confirm `~/.ssh/config` has a `Host bgu` entry before relying on it;
otherwise use `ssh <user>@slurm.bgu.ac.il`.

## Prerequisite: VPN — HARD STOP if not connected

The cluster is only reachable from on-campus or through BGU's VPN (Check
Point Endpoint Security).

**Before doing anything else, run:**

```bash
ssh -o BatchMode=yes -o ConnectTimeout=8 bgu "echo ok"
```

- If this prints `ok` → proceed.
- If it times out (`Connection timed out`, `Operation timed out`, no output) →
  **the VPN is not connected. STOP executing this skill.** Do not attempt
  `sbatch`, `squeue`, `sacct`, `scp`, or any further `ssh bgu` call. Tell the
  user the VPN (Check Point Endpoint Security) needs to be connected, and wait
  for them to confirm it is before retrying the check above.
- If it says `Permission denied` → that's a credentials/key problem, not the
  VPN; surface it to the user, still don't proceed with job commands.

Occasionally the VPN reconnects for one call and drops again on the next —
retry the check once or twice before concluding it's down, but never skip it.

## Submitting a job

GPU jobs should pin the GPU type explicitly rather than submitting a bare
`--gres=gpu:1`. A generic request can land on an old GTX 1080 Ti (compute
capability 6.1), which recent `torch` CUDA wheels do **not** ship kernels for
(`sm_61` unsupported; only `sm_75`+ is built). Use `rtx_4090`, `rtx_3090`, or
`rtx_6000` — all confirmed compatible.

```bash
ssh bgu "cd <project home> && mkdir -p logs && \
  sbatch --array=0-N%6 <job>.sbatch"
```

Cap array concurrency at `%6`–`%8`. If this account has no protected slots, a
large `%N` just queues behind `QOSMaxGRESPerUser` rather than failing.

**Windows line-ending trap**: if you edit an `.sbatch` file locally on
Windows (even via a text-editing tool, not just Notepad) and `scp` it back
up, `sbatch` rejects it with `Batch script contains DOS line breaks`. Fix in
place with `sed -i 's/\r$//' <file>.sbatch` on the cluster before resubmitting.

For quick one-off checks (smoke tests, sanity runs) prefer `sbatch --wrap=`
over writing a throwaway script:

```bash
ssh bgu "cd <project home> && sbatch --partition=main --qos=normal \
  --gres=gpu:rtx_6000:1 --time=00:20:00 --mem=16G --job-name=smoke \
  --output=logs/smoke_%j.out --wrap='source /storage/modules/packages/anaconda/etc/profile.d/conda.sh \
  && conda activate <env> && python3 -u main.py --smoke'"
```

Use `python3 -u` (or `export PYTHONUNBUFFERED=1`) for anything you'll tail
live — otherwise output buffers and `tail -f` shows nothing for a long time.

## Checking status

```bash
ssh -o BatchMode=yes bgu "squeue --me -o '%i %j %T %M %L %D %R'"
```

Columns: jobid, name, state, time used, time left, nodes, reason (for
PENDING). Common PENDING reasons that are **not errors** (don't resubmit):

- `QOSMaxGRESPerUser` — the per-user GPU cap is full; tasks start as slots free.
- `JobArrayTaskLimit` — the array's own `%N` cap is full.
- `Resources` / `Priority` — cluster busy / queued behind others.

Once a job leaves `squeue` it's finished (completed or failed) — check with
`sacct`:

```bash
ssh -o BatchMode=yes bgu "cd <project home> && \
  sacct -j <jobid> -o JobID,State,ExitCode,Elapsed --array 2>&1 | grep -v '\.\(batch\|extern\)'"
```

For a whole array, tally states in one shot:

```bash
ssh -o BatchMode=yes bgu "sacct -j <jobid> -o JobID,State --array 2>&1 | \
  grep -E '^<jobid>_[0-9]+ ' | awk '{print \$2}' | sort | uniq -c"
```

## Reading logs

Logs land in `logs/<name>_<jobid>_<taskid>.out` (array) or
`logs/<name>_<jobid>.out` (single job). **Progress bars (tqdm) use `\r`, not
`\n`**, so a plain `cat`/`tail` shows one enormous line and `grep` on it looks
like it's returning garbage or nothing. Always pipe through `tr '\r' '\n'`
first:

```bash
ssh -o BatchMode=yes bgu "tr '\r' '\n' < <project home>/logs/<file>.out | grep -iE 'error|traceback|done'"
```

`grep -c` on the raw file will report a nonzero count even though the
un-filtered `grep` looks like it printed nothing useful — same `\r`-vs-`\n`
issue; always run text extraction through `tr` first.

A run that looks "stuck" for hours is often just legitimately slow. Check for
forward progress (`tail -c 2000 <file> | tr '\r' '\n'`) and absence of
`Traceback`/`Error` before assuming something's wrong. Two consecutive restart
banners in the same log means the task was preempted (`REQUEUE`) and restarted
from scratch — expected occasionally under `normal` QoS, not a bug.

## Long-running monitoring pattern

For a job expected to take a while, don't poll in a tight loop. Submit,
confirm it started (`squeue`), then either:

- Use a short bash wait-loop with `sleep` capped at a few minutes when you
  need to block briefly (`timeout 280 bash -c 'while ...; sleep 30; done'`
  degrades to a background task automatically past 120s — that's fine, it
  reports back).
- For anything that'll take 20+ minutes, prefer `ScheduleWakeup` to check
  back later rather than blocking the conversation.

When reporting status, always give concrete numbers (N done / M running /
K pending / F failed) pulled from `sacct`, not just "still running" —
`squeue` alone doesn't show completed-vs-failed history.
