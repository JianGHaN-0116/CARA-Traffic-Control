# OVS/tc Replay

This directory contains the Mininet/Open vSwitch replay materials.

## Requirements

- Linux with root privileges (or WSL2 with OVS installed)
- Open vSwitch (`ovs-vsctl`, `ovs-ofctl`)
- Mininet (`mn`)
- iperf3

## Directory Structure

```
ovs_replay/
├── README.md
├── topology/           # Mininet topology files
│   ├── single_targeted.py
│   └── two_switch_mixed.py
├── rules/              # OVS flow rules
│   ├── forward.txt
│   ├── inspect.txt
│   ├── mirror.txt
│   ├── throttle.txt
│   ├── drop.txt
│   └── isolate.txt
└── scripts/            # Replay automation
    ├── run_single_targeted.sh
    ├── run_two_switch_mixed.sh
    └── collect_metrics.sh
```

## Running the Replay

### Single-Switch Targeted Replay

```bash
sudo bash scripts/run_single_targeted.sh
```

### Two-Switch Mixed Bottleneck Replay

```bash
sudo bash scripts/run_two_switch_mixed.sh
```

## Without Root Privileges

If you cannot run OVS, use the pre-computed replay logs in `outputs/logs/ovs_*.log` to reproduce the summary tables.

## Action Realization

| Action | OVS/tc Command |
|--------|----------------|
| Forward | Default flow-table entry |
| Inspect | `ovs-ofctl add-flow s1 priority=100,ip,nw_dst=10.0.0.200,actions=output:IDS_PORT` |
| Mirror | `ovs-ofctl add-flow s1 priority=100,ip,actions=output:NORMAL,output:IDS_PORT` |
| Throttle | `tc qdisc add dev eth0 root tbf rate 10mbit burst 32kbit latency 10ms` |
| Drop | `ovs-ofctl add-flow s1 priority=200,ip,nw_src=ATTACKER_IP,actions=drop` |
| Isolate | `ovs-ofctl add-flow s1 priority=200,ip,nw_src=ATTACKER_IP,actions=drop` + port isolation |
