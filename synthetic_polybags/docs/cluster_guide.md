# OVGU Cluster — Quick Reference

**Cluster**: `ants.cs.ovgu.de`  
**Username**: `diky85bu`  
**GPU partition**: `gpu-stud` — NVIDIA A40, 46 GB VRAM, 1-day time limit

---

## 1. Connect

1. Connect to **OVGU VPN** first.
2. SSH into the login node:

```bash
ssh diky85bu@ants.cs.ovgu.de
```

To avoid typing your username each time, add to `~/.ssh/config`:
```
Host ants.cs.ovgu.de
    User diky85bu
```

To avoid typing your password each time (SSH key):
```bash
ssh-keygen
ssh-copy-id ants.cs.ovgu.de
```

---

## 2. Load the software environment

Run this after every login (or add to `~/.bashrc` to load automatically):

```bash
source /opt/spack/main/env.sh
```

To add permanently:
```bash
cat >> ~/.bashrc << 'EOF'

if [ -n "$PS1" ]
then
    source /opt/spack/main/env.sh
fi
EOF
```

---

## 3. Start an interactive GPU session

```bash
srun -p gpu-stud --gres=gpu:1 --pty bash
```

You will land on a compute node (e.g. `ant7`). Verify the GPU:

```bash
nvidia-smi
```

To request more CPUs or memory:
```bash
srun -p gpu-stud --gres=gpu:1 -c 8 --mem=32G --pty bash
```

Exit the GPU session when done:
```bash
exit
```

---

## 4. Submit a non-interactive job (sbatch)

Create a job script, e.g. `train.slurm`:

```bash
#!/bin/bash
#SBATCH --job-name=yolo_train
#SBATCH --partition=gpu-stud
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=0-20:00:00
#SBATCH --output=logs/train_%j.out
#SBATCH --error=logs/train_%j.err

source /opt/spack/main/env.sh
source ~/venv/bin/activate

python training/pseudo_label_train.py --setup 0
yolo obb train data=pseudo_label/round_0/data.yaml \
     model=yolo11n-obb.pt epochs=100 imgsz=1920 batch=4 \
     name=round_0 project=pseudo_label/runs
```

Submit and monitor:
```bash
sbatch train.slurm
squeue --me          # check status
scancel <job-id>     # cancel a job if needed
scancel --me         # cancel all your jobs
```

Output goes to `logs/train_<job-id>.out`.

---

## 5. Transfer files to/from the cluster

```bash
# Upload a file or directory
scp local/file diky85bu@ants.cs.ovgu.de:remote/path/
scp -r local/directory diky85bu@ants.cs.ovgu.de:remote/path/

# Download a file or directory
scp diky85bu@ants.cs.ovgu.de:remote/file local/path/
scp -r diky85bu@ants.cs.ovgu.de:remote/dir local/path/
```

---

## 6. Check available nodes

```bash
sinfo
```

Key partitions:

| Partition | Time limit | Notes |
|-----------|-----------|-------|
| `gpu-stud` | 1 day | GPU nodes (ant1, ant2, ant7, ant8) — use this for training |
| `all` | 1 day | CPU nodes only |
| `vl-parcio` | 6 hours | Teaching partition |

---

## 7. Python environment setup (first time only)

```bash
# On a GPU node or login node
python3 -m venv ~/venv
source ~/venv/bin/activate
pip install --upgrade pip
pip install ultralytics opencv-python scipy matplotlib
```

Activate in future sessions:
```bash
source ~/venv/bin/activate
```

---

## Further reading

- Full cluster wiki: https://code.ovgu.de/fin-all/cluster/-/wikis/home
- GPU guide: https://code.ovgu.de/fin-all/cluster/-/wikis/GPU-Guide
- SLURM docs: https://slurm.schedmd.com/documentation.html
