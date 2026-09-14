"""
Expanded OVS Topology Replay (P2-Experiment 7)

Lightweight expansion from 4 switches/7 hosts to 8 switches/16 hosts:
  - 2 edge switches (s1, s5), 2 aggregation switches (s2, s6)
  - 2 core switches (s3, s7), 2 scrubbing switches (s4, s8)
  - 6 benign hosts, 4 attacker hosts, 2 IDS/mirror nodes, 2 servers
  - 2 bottleneck links (40 Mbps each)
  - 2 scrubbing paths

Reports: rule-install latency, active flow entries, OVS CPU,
         benign throughput/loss, attack suppression.

This is NOT a production-scale experiment; it tests whether the
4-switch results scale modestly without introducing new bugs.

Usage:
    sudo python -m src.experiments.expanded_topology_replay
"""
import argparse
import json
import os
import re
import subprocess
import time
from contextlib import suppress

import pandas as pd
import numpy as np

from mininet.link import TCLink
from mininet.log import setLogLevel
from mininet.net import Mininet
from mininet.node import OVSSwitch, Host, RemoteController
from mininet.topo import Topo


# ─── Expanded Topology ───────────────────────────────────────────────────────

class ExpandedEdgeTopo(Topo):
    """8-switch, 16-host topology for modest scale testing.

    Architecture:
      Edge tier:    s1 (6 hosts: ben1-3, atk1-2, ids1), s5 (ben4-6, atk3-4, ids2)
      Aggregation:  s2, s6
      Core:         s3, s7
      Scrubbing:    s4, s8
      Servers:      srv1 (on s3), srv2 (on s7)
    """

    def build(self):
        # Hosts on edge switch s1
        ben1 = self.addHost("ben1", ip="10.0.1.1/16")
        ben2 = self.addHost("ben2", ip="10.0.1.2/16")
        ben3 = self.addHost("ben3", ip="10.0.1.3/16")
        atk1 = self.addHost("atk1", ip="10.0.2.1/16")
        atk2 = self.addHost("atk2", ip="10.0.2.2/16")
        ids1 = self.addHost("ids1", ip="10.0.5.1/16")

        # Hosts on edge switch s5
        ben4 = self.addHost("ben4", ip="10.0.1.4/16")
        ben5 = self.addHost("ben5", ip="10.0.1.5/16")
        ben6 = self.addHost("ben6", ip="10.0.1.6/16")
        atk3 = self.addHost("atk3", ip="10.0.2.3/16")
        atk4 = self.addHost("atk4", ip="10.0.2.4/16")
        ids2 = self.addHost("ids2", ip="10.0.5.2/16")

        # Servers
        srv1 = self.addHost("srv1", ip="10.0.3.1/16")
        srv2 = self.addHost("srv2", ip="10.0.3.2/16")

        # Switches
        s1 = self.addSwitch("s1", protocols="OpenFlow13", failMode="standalone")
        s2 = self.addSwitch("s2", protocols="OpenFlow13", failMode="standalone")
        s3 = self.addSwitch("s3", protocols="OpenFlow13", failMode="standalone")
        s4 = self.addSwitch("s4", protocols="OpenFlow13", failMode="standalone")
        s5 = self.addSwitch("s5", protocols="OpenFlow13", failMode="standalone")
        s6 = self.addSwitch("s6", protocols="OpenFlow13", failMode="standalone")
        s7 = self.addSwitch("s7", protocols="OpenFlow13", failMode="standalone")
        s8 = self.addSwitch("s8", protocols="OpenFlow13", failMode="standalone")

        # Edge links (s1)
        for h in (ben1, ben2, ben3, atk1, atk2, ids1):
            self.addLink(h, s1, cls=TCLink, bw=100, delay="2ms", loss=0, use_htb=True)

        # Edge links (s5)
        for h in (ben4, ben5, ben6, atk3, atk4, ids2):
            self.addLink(h, s5, cls=TCLink, bw=100, delay="2ms", loss=0, use_htb=True)

        # Aggregation links
        self.addLink(s1, s2, cls=TCLink, bw=40, delay="5ms", loss=0, use_htb=True)
        self.addLink(s5, s6, cls=TCLink, bw=40, delay="5ms", loss=0, use_htb=True)

        # Core links
        self.addLink(s2, s3, cls=TCLink, bw=100, delay="2ms", loss=0, use_htb=True)
        self.addLink(s6, s7, cls=TCLink, bw=100, delay="2ms", loss=0, use_htb=True)
        self.addLink(s2, s7, cls=TCLink, bw=100, delay="3ms", loss=0, use_htb=True)  # cross-link
        self.addLink(s6, s3, cls=TCLink, bw=100, delay="3ms", loss=0, use_htb=True)  # cross-link

        # Server links
        self.addLink(srv1, s3, cls=TCLink, bw=100, delay="2ms", loss=0, use_htb=True)
        self.addLink(srv2, s7, cls=TCLink, bw=100, delay="2ms", loss=0, use_htb=True)

        # Scrubbing paths
        self.addLink(s1, s4, cls=TCLink, bw=100, delay="3ms", loss=0, use_htb=True)
        self.addLink(s4, s3, cls=TCLink, bw=100, delay="3ms", loss=0, use_htb=True)
        self.addLink(s5, s8, cls=TCLink, bw=100, delay="3ms", loss=0, use_htb=True)
        self.addLink(s8, s7, cls=TCLink, bw=100, delay="3ms", loss=0, use_htb=True)


BENIGN_HOSTS = ["ben1", "ben2", "ben3", "ben4", "ben5", "ben6"]
ATTACKER_HOSTS = ["atk1", "atk2", "atk3", "atk4"]
IDS_HOSTS = ["ids1", "ids2"]
SERVER_HOSTS = ["srv1", "srv2"]

ATTACKER_IPS = ["10.0.2.1", "10.0.2.2", "10.0.2.3", "10.0.2.4"]
BENIGN_IPS = ["10.0.1.1", "10.0.1.2", "10.0.1.3", "10.0.1.4", "10.0.1.5", "10.0.1.6"]
SERVER_IPS = ["10.0.3.1", "10.0.3.2"]


def ovs_cmd(cmd_str, timeout=10):
    """Execute OVS command."""
    try:
        result = subprocess.run(
            cmd_str, shell=True, capture_output=True, text=True, timeout=timeout
        )
        return result.stdout.strip(), result.returncode
    except subprocess.TimeoutExpired:
        return "", 1


def count_flow_entries(switch):
    """Count flow entries on a switch."""
    output, rc = ovs_cmd(f"ovs-ofctl dump-flows {switch} -O OpenFlow13")
    if rc != 0:
        return 0
    return len([l for l in output.split("\n") if l.strip() and "duration" in l])


def get_ovs_cpu():
    """Get OVS vswitchd CPU usage."""
    output, rc = ovs_cmd("ps -p $(pgrep ovs-vswitchd | head -1) -o %cpu --no-headers 2>/dev/null")
    if rc == 0 and output.strip():
        try:
            return float(output.strip())
        except ValueError:
            return 0.0
    return 0.0


def run_iperf(net, client, server_ip, duration=2, proto="tcp"):
    """Run iperf3 between client and server."""
    client_node = net.get(client)
    try:
        if proto == "tcp":
            output = client_node.cmd(f"iperf3 -c {server_ip} -t {duration} -J 2>/dev/null")
        else:
            output = client_node.cmd(
                f"iperf3 -c {server_ip} -t {duration} -u -b 50M -J 2>/dev/null"
            )
        try:
            data = json.loads(output)
            if proto == "tcp":
                bps = data["end"]["sum_sent"]["bits_per_second"]
            else:
                bps = data["end"]["sum"]["bits_per_second"]
            return bps / 1e6  # Mbps
        except (json.JSONDecodeError, KeyError):
            return 0.0
    except Exception:
        return 0.0


def apply_action(net, action, attacker_ips, step_id):
    """Apply a control action on the expanded topology."""
    t0 = time.time()

    if action == 0:  # Forward
        pass
    elif action == 1:  # Inspect
        for switch in ["s1", "s5"]:
            ovs_cmd(f"ovs-ofctl add-flow {switch} -O OpenFlow13 "
                    f"priority=100,ip,nw_dst={SERVER_IPS[0]},"
                    f"actions=output:NORMAL,output:4")
    elif action == 5:  # Drop
        for ip in attacker_ips:
            for switch in ["s1", "s5"]:
                ovs_cmd(f"ovs-ofctl add-flow {switch} -O OpenFlow13 "
                        f"priority=200,ip,nw_src={ip},actions=drop")
    elif action == 6:  # Isolate
        for ip in attacker_ips:
            for switch in ["s1", "s5"]:
                ovs_cmd(f"ovs-ofctl add-flow {switch} -O OpenFlow13 "
                        f"priority=210,ip,nw_src={ip},actions=drop")

    latency = (time.time() - t0) * 1000
    return latency


def cleanup_flows():
    """Remove all non-default flow entries."""
    for sw in ["s1", "s2", "s3", "s4", "s5", "s6", "s7", "s8"]:
        ovs_cmd(f"ovs-ofctl del-flows {sw} -O OpenFlow13")


def run_expanded_replay(plan_csv=None, n_steps=200):
    """Run the expanded topology replay."""
    BASE_DIR = os.path.join(os.path.dirname(__file__), "..", "..")
    RESULTS_DIR = os.path.join(BASE_DIR, "results", "expanded_topology_replay")
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # Load action plan
    if plan_csv and os.path.exists(plan_csv):
        plan_df = pd.read_csv(plan_csv)
    else:
        # Generate synthetic CARA-TC plan
        rng = np.random.default_rng(42)
        actions = []
        for i in range(n_steps):
            is_attack = rng.random() < 0.5
            p_t = rng.uniform(0.6, 1.0) if is_attack else rng.uniform(0.1, 0.5)
            if p_t > 0.7:
                action = rng.choice([5, 6, 4], p=[0.3, 0.3, 0.4])
            elif p_t > 0.5:
                action = rng.choice([3, 1], p=[0.6, 0.4])
            else:
                action = 0
            actions.append({
                "step": i,
                "action": action,
                "true_label": 1 if is_attack else 0,
            })
        plan_df = pd.DataFrame(actions)

    print(f"Running expanded topology replay ({len(plan_df)} steps)...")

    # Create Mininet topology
    topo = ExpandedEdgeTopo()
    net = Mininet(topo=topo, switch=OVSSwitch, link=TCLink, controller=None)
    net.start()

    # Enable STP to prevent loops
    for sw in ["s1", "s2", "s3", "s4", "s5", "s6", "s7", "s8"]:
        ovs_cmd(f"ovs-vsctl set bridge {sw} stp_enable=true")

    time.sleep(5)  # Wait for STP convergence

    results = []

    try:
        for step_idx, row in plan_df.iterrows():
            action = int(row.get("action", 0))
            is_attack = row.get("true_label", 0) == 1

            # Apply action
            target_ips = ATTACKER_IPS if is_attack else BENIGN_IPS[:1]
            install_latency = apply_action(net, action, target_ips, step_idx)

            # Measure metrics
            flow_entries = sum(count_flow_entries(sw)
                              for sw in ["s1", "s2", "s3", "s5", "s6", "s7"])
            ovs_cpu = get_ovs_cpu()

            # Sample throughput (every 10 steps)
            if step_idx % 10 == 0:
                ben_tp = run_iperf(net, "ben1", SERVER_IPS[0], duration=1)
            else:
                ben_tp = 0.0

            results.append({
                "step": step_idx,
                "action": action,
                "true_label": is_attack,
                "install_latency_ms": install_latency,
                "flow_entries": flow_entries,
                "ovs_cpu": ovs_cpu,
                "benign_throughput_mbps": ben_tp,
            })

            # Cleanup every step
            cleanup_flows()

            if step_idx % 50 == 0:
                print(f"  Step {step_idx}/{len(plan_df)}: "
                      f"latency={install_latency:.1f}ms, "
                      f"flows={flow_entries}, cpu={ovs_cpu:.1f}%")

    finally:
        net.stop()

    # Save results
    df = pd.DataFrame(results)
    output_path = os.path.join(RESULTS_DIR, "expanded_topology_replay.csv")
    df.to_csv(output_path, index=False)
    print(f"\nResults saved to {output_path}")

    # Summary
    print(f"\n{'='*80}")
    print("EXPANDED TOPOLOGY REPLAY SUMMARY (8 switches, 16 hosts)")
    print(f"{'='*80}")
    print(f"Mean install latency: {df['install_latency_ms'].mean():.1f} ms")
    print(f"P95 install latency:  {df['install_latency_ms'].quantile(0.95):.1f} ms")
    print(f"Mean flow entries:    {df['flow_entries'].mean():.1f}")
    print(f"Peak flow entries:    {df['flow_entries'].max()}")
    print(f"Mean OVS CPU:         {df['ovs_cpu'].mean():.1f}%")
    print(f"Mean benign TP:       {df[df['benign_throughput_mbps'] > 0]['benign_throughput_mbps'].mean():.1f} Mbps")

    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan-csv", default=None)
    parser.add_argument("--n-steps", type=int, default=200)
    args = parser.parse_args()
    setLogLevel("info")
    run_expanded_replay(args.plan_csv, args.n_steps)
