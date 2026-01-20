# Quick Run

```bash
git clone https://github.com/Traffic-Alpha/TransSimHub.git
cd TransSimHub
pip install -e .
cd ..
```

```bash
git clone https://github.com/hzy4869/TrafficMonitor.git
cd TrafficMonitor
pip install -r requirements.txt
```

```bash
which sumo
export SUMO_HOME= "/path/to/bin/sumo"
```

# some convenient command
export SUMO_HOME="/home/zeyun/miniconda3/envs/drone_rl/share/sumo"
tmux new -s process1
tmux attach -t process1

conda activate drone_rl
cd TrafficMonitor