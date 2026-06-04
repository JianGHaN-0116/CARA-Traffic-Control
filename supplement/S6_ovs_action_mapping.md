# S6. OVS/Mininet Replay — Action-to-tc Mapping

## Overview

The Enhanced OVS/Mininet replay maps each logical controller action to concrete Open vSwitch (OVS) flow rules and Linux traffic control (tc) commands. The mapping is implemented in `ovs_replay/topology/enhanced_replay.py`.

---

## Topology

```
benign1 (10.0.1.1) ----\
benign2 (10.0.1.2) ----+--> s1 (edge) --bottleneck--> s2 (agg) --> s3 (core) --> server (10.0.3.2)
attacker1 (10.0.1.3) -/        |                        |               ^
attacker2 (10.0.1.4) -/        +--> IDS/mirror port      |               |
ids (10.0.1.5) -------/                                 |               |
                                                         +--> s4 (scrub) -+
```

- **Edge-to-agg bottleneck**: 40 Mbps, 5 ms delay (simulates constrained edge link)
- **Reroute path** (s1 → s4 → s3): 100 Mbps, 3 ms delay per hop
- **Other links**: 100 Mbps, 2 ms delay
- **4 switches**: s1 (edge), s2 (aggregation), s3 (core), s4 (scrubbing/reroute)

---

## Action Mapping

| Controller Action | OVS Rule / tc Action | Description |
|---|---|---|
| **Forward** | `ovs-ofctl add-flow s1 priority=100,ip,nw_src=$HOST,actions=output:$PORT` | Normal forwarding through the default L2 path. Packets pass from s1 to s2 to s3 without inspection. |
| **Inspect** | `ovs-ofctl add-flow s1 priority=200,ip,nw_src=$HOST,actions=output:IDS_PORT` | Steer traffic from the flagged host to the IDS mirror port for deep packet inspection. The IDS host runs Suricata-style analysis. |
| **Mirror** | `ovs-ofctl add-flow s1 priority=200,ip,nw_src=$HOST,actions=output:$NORMAL_PORT,output:IDS_PORT` | Duplicate traffic: original goes to the normal path, a copy goes to the IDS monitoring port. Used for passive monitoring without disruption. |
| **Throttle** | `tc qdisc add dev $HOST-eth0 root handle 1: htb default 30; tc class add dev $HOST-eth0 parent 1: classid 1:1 htb rate ${RATE}mbit ceil ${RATE}mbit` | Apply Linux tc rate limiting to the host's virtual interface. Limits bandwidth to a configurable rate (typically 5–10 Mbps for throttled hosts). |
| **Reroute** | `ovs-ofctl add-flow s1 priority=300,ip,nw_src=$HOST,actions=output:s4-eth1; ovs-ofctl add-flow s4 priority=300,ip,nw_src=$HOST,actions=output:s4-eth2` | Redirect traffic through the s4 scrubbing switch. The s4 path has higher bandwidth (100 Mbps) and lower latency, bypassing the congested bottleneck link. Packets reach s3 via s4 instead of s2. |
| **Drop** | `ovs-ofctl add-flow s1 priority=400,ip,nw_src=$HOST,actions=drop` | Install a drop rule on the edge switch. All packets from the flagged host are dropped at ingress. Equivalent to ACL blackhole. |
| **Isolate** | `ovs-ofctl add-flow s1 priority=400,ip,nw_src=$HOST,nw_dst=10.0.0.0/8,actions=drop; ovs-ofctl add-flow s1 priority=300,ip,nw_src=$HOST,actions=output:IDS_PORT` | Drop all production traffic from the quarantined host while maintaining IDS visibility. A quarantine/blackhole-like rule that isolates the host from the production network. |

---

## Implementation Notes

### Controller Loop

Each replay step:
1. **Read** the window-level detector summary (attack ratio, confidence) from the pre-computed plan
2. **Consult** the selected controller's policy to choose an action
3. **Apply** the mapped OVS/tc rule
4. **Wait** for the inter-step replay duration
5. **Measure** metrics: throughput (iperf3), loss (packet counts), latency (ping), OVS flow entries (`ovs-ofctl dump-flows` count), CPU usage
6. **Remove** the previous step's flow rules before installing the next step's rules

### Label Usage

**Labels (ground-truth attack/benign) are used ONLY for offline scoring.** The live controller receives only detection-derived summaries (detector confidence, estimated attack ratio per window) and emits window-level actions. No ground-truth labels are available to the controller during live operation.

### Action Durability

OVS flow rules persist until explicitly removed. The controller loop installs new rules at each step and removes old ones, maintaining a constant per-step flow footprint.

---

## Enhanced Replay Metrics

| Metric | Collection Method |
|---|---|
| Benign TCP throughput (Mbps) | iperf3 server-side report |
| Benign UDP throughput (Mbps) | iperf3 server-side report |
| Attack TCP throughput (Mbps) | iperf3 server-side report |
| Attack UDP throughput (Mbps) | iperf3 server-side report |
| Benign loss (%) | `(sent − received) / sent × 100` from iperf3 |
| Attack suppression (%) | `(no_control_attack − controlled_attack) / no_control_attack × 100` |
| Rule-install latency (ms) | `time ovs-ofctl add-flow` wall-clock measurement |
| OVS flow entries | `ovs-ofctl dump-flows s1 \| wc -l` |
| OVS CPU (%) | `top -bn1` OVS process CPU |
| IDS received bytes | `ifconfig ids-eth0` RX bytes |
| RTT (ms) | `ping -c 5` average RTT |
| Aggressive action rate | % of steps where Drop/Isolate/Throttle were applied |

---

## Replay Plan Format

The action plan file (`enhanced_replay_plan.csv`) contains pre-computed detection summaries per replay step:

```csv
step,timestamp,host,detector_confidence,attack_ratio,source_ip
0,0.0,attacker1,0.92,0.85,10.0.1.3
1,2.5,benign1,0.08,0.02,10.0.1.1
...
```

### Required Files for Replay

| File | Description |
|---|---|
| `ovs_replay/topology/enhanced_replay.py` | Main replay controller script |
| `ovs_replay/topology/merge_results.py` | Metric aggregation script |
| `new_experiments/enhanced_ovs_replay/enhanced_replay_plan.csv` | Action plan with per-step detection summaries |
| `new_experiments/enhanced_ovs_replay/results_cara/enhanced_summary.csv` | CARA-TC replay results |
| `ovs_replay/rules/` | OVS flow rule templates |
| `ovs_replay/scripts/run_enhanced_replay.sh` | Automation script (available in the public GitHub repository, not in this supplementary evidence package) |

### How to Verify Enhanced OVS Results

This supplementary evidence package includes pre-computed OVS replay results in `tables/review_table17.csv` (manuscript Table 10 source). Full executable OVS replay scripts are provided in the public GitHub repository (`CARA-Traffic-Control`), not in this supplementary evidence package.
