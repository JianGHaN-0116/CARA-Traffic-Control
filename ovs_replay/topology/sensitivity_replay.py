"""
Sensitivity Replay: Bottleneck Rate and Attack Intensity Variants

Runs abbreviated 100-step OVS/Mininet replays with configurable:
  - Bottleneck bandwidth (default 40 Mbps, variant 20 Mbps)
  - Number of attacker hosts (default 2, variant 3)

Usage (no root):
  python src/experiments/export_sensitivity_replay_plan.py --per-label 25 --bottleneck-mbps 20

Usage (root, after plan export):
  sudo python ovs_replay/topology/sensitivity_replay.py \
    --plan-csv new_experiments/sensitivity_replay/plan_20mbps.csv \
    --out-dir new_experiments/sensitivity_replay/results_20mbps \
    --bottleneck-mbps 20 \
    --num-attackers 2

  sudo python ovs_replay/topology/sensitivity_replay.py \
    --plan-csv new_experiments/sensitivity_replay/plan_3attackers.csv \
    --out-dir new_experiments/sensitivity_replay/results_3attackers \
    --bottleneck-mbps 40 \
    --num-attackers 3
"""
import argparse
import json
import os
import re
import subprocess
import time
from contextlib import suppress

import pandas as pd

from mininet.link import TCLink
from mininet.log import setLogLevel
from mininet.net import Mininet
from mininet.node import OVSSwitch, Host
from mininet.topo import Topo


class SensitivityEdgeTopo(Topo):
    def build(self, bottleneck_mbps=40, num_attackers=2):
        benign1 = self.addHost("benign1", ip="10.0.1.1/16")
        benign2 = self.addHost("benign2", ip="10.0.1.2/16")
        attacker_hosts = []
        for i in range(num_attackers):
            ip_suffix = 3 + i
            h = self.addHost(f"attacker{i+1}", ip=f"10.0.1.{ip_suffix}/16")
            attacker_hosts.append(h)
        ids_ip = 3 + num_attackers
        ids = self.addHost("ids", ip=f"10.0.1.{ids_ip}/16")
        server = self.addHost("server", ip="10.0.3.2/16")

        s1 = self.addSwitch("s1", protocols="OpenFlow13", failMode="standalone")
        s2 = self.addSwitch("s2", protocols="OpenFlow13", failMode="standalone")
        s3 = self.addSwitch("s3", protocols="OpenFlow13", failMode="standalone")
        s4 = self.addSwitch("s4", protocols="OpenFlow13", failMode="standalone")

        for h in (benign1, benign2) + tuple(attacker_hosts) + (ids,):
            self.addLink(h, s1, cls=TCLink, bw=100, delay="2ms", loss=0, use_htb=True)

        self.addLink(s1, s2, cls=TCLink, bw=bottleneck_mbps, delay="5ms", loss=0, use_htb=True)
        self.addLink(s2, s3, cls=TCLink, bw=100, delay="2ms", loss=0, use_htb=True)
        self.addLink(server, s3, cls=TCLink, bw=100, delay="2ms", loss=0, use_htb=True)

        self.addLink(s1, s4, cls=TCLink, bw=100, delay="3ms", loss=0, use_htb=True)
        self.addLink(s4, s3, cls=TCLink, bw=100, delay="3ms", loss=0, use_htb=True)


def host_port(switch, host):
    return switch.ports[switch.connectionsTo(host)[0][0]]


def link_intf_name(node_a, node_b):
    intf_a, _ = node_a.connectionsTo(node_b)[0]
    return intf_a.name


def clear_tc(host):
    host.cmd(f"tc qdisc del dev {host.defaultIntf().name} root >/dev/null 2>&1 || true")


def clear_mirror():
    for sw in ("s1", "s2", "s3", "s4"):
        os.system(f"ovs-vsctl clear Bridge {sw} mirrors >/dev/null 2>&1 || true")


def clear_flows():
    for sw in ("s1", "s2", "s3", "s4"):
        os.system(f"ovs-ofctl -O OpenFlow13 del-flows {sw} >/dev/null 2>&1 || true")


def install_l2_forwarding(net, server_ip="10.0.3.2"):
    for sw in ("s1", "s2", "s3"):
        os.system(
            f'ovs-ofctl -O OpenFlow13 add-flow {sw} '
            f'"priority=0,actions=NORMAL" >/dev/null 2>&1'
        )
    os.system(
        f'ovs-ofctl -O OpenFlow13 add-flow s4 '
        f'"priority=0,actions=drop" >/dev/null 2>&1'
    )


def count_flow_entries(switch_name="s1"):
    try:
        out = subprocess.check_output(
            f"ovs-ofctl -O OpenFlow13 dump-flows {switch_name}", shell=True, stderr=subprocess.DEVNULL
        ).decode()
        return max(0, len(out.strip().split("\n")) - 1)
    except Exception:
        return 0


def get_cpu_usage():
    try:
        out = subprocess.check_output(
            "ps -C ovs-vswitchd -o %cpu= 2>/dev/null || echo 0", shell=True
        ).decode().strip()
        vals = out.strip().split("\n")
        return float(vals[0]) if vals else 0.0
    except Exception:
        return 0.0


def start_iperf_servers(server, num_attackers=2):
    server.cmd("pkill -f 'iperf3 -s' >/dev/null 2>&1 || true")
    num_ports = 2 + num_attackers * 2
    for port_offset in range(num_ports):
        server.cmd(f"iperf3 -s -p {5201 + port_offset} -D")
    time.sleep(0.8)


def stop_iperf_processes(net, all_hosts):
    for h in all_hosts:
        try:
            net.get(h).cmd("pkill -f iperf3 >/dev/null 2>&1 || true")
        except Exception:
            pass


def run_iperf3_tcp(client, target_ip, duration=1, timeout_s=5, port=5201):
    output = client.cmd(
        f"timeout {timeout_s}s iperf3 -J --connect-timeout 1200 -t {duration} -p {port} -c {target_ip} || true"
    )
    return _parse_iperf_json(output)


def run_iperf3_udp(client, target_ip, bandwidth="50M", duration=1, timeout_s=5, port=5203):
    output = client.cmd(
        f"timeout {timeout_s}s iperf3 -J --connect-timeout 1200 -u -b {bandwidth} -t {duration} -p {port} -c {target_ip} || true"
    )
    return _parse_iperf_json(output)


def _parse_iperf_json(output):
    start = output.find("{")
    end = output.rfind("}")
    if start == -1 or end == -1:
        return {"throughput_mbps": 0.0, "packets": None, "lost_percent": None}
    try:
        payload = json.loads(output[start:end + 1])
        summary = (
            payload.get("end", {}).get("sum_received")
            or payload.get("end", {}).get("sum")
            or payload.get("end", {}).get("sum_sent")
            or {}
        )
        udp_summary = payload.get("end", {}).get("sum", {})
        lost_pct = udp_summary.get("lost_percent") if "lost_percent" in udp_summary else None
        return {
            "throughput_mbps": float(summary.get("bits_per_second", 0.0)) / 1e6,
            "packets": summary.get("packets"),
            "lost_percent": lost_pct,
        }
    except Exception:
        return {"throughput_mbps": 0.0, "packets": None, "lost_percent": None}


def run_ping(host, target_ip, count=2, timeout_s=4):
    output = host.cmd(f"timeout {timeout_s}s ping -c {count} -i 0.2 -W 1 {target_ip} || true")
    loss_match = re.search(r"(\d+(?:\.\d+)?)% packet loss", output)
    rtt_match = re.search(r"= ([\d\.]+)/([\d\.]+)/([\d\.]+)/([\d\.]+) ms", output)
    return {
        "packet_loss_pct": float(loss_match.group(1)) if loss_match else 100.0,
        "avg_rtt_ms": float(rtt_match.group(2)) if rtt_match else None,
    }


def read_rx_bytes(host, intf_name):
    output = host.cmd(f"cat /sys/class/net/{intf_name}/statistics/rx_bytes").strip()
    with suppress(ValueError):
        return int(output)
    return 0


def apply_forward(_net, _target_name, _mode=None):
    return


def apply_throttle(net, _target_name, _mode=None, rate_mbit=5):
    edge = net.get("s1")
    agg = net.get("s2")
    intf_name = link_intf_name(edge, agg)
    os.system(
        f"tc qdisc replace dev {intf_name} root tbf rate {rate_mbit}mbit burst 32kbit latency 80ms >/dev/null 2>&1"
    )


def apply_drop_attackers(net, attacker_ips):
    for attacker_ip in attacker_ips:
        os.system(
            f"ovs-ofctl -O OpenFlow13 add-flow s1 "
            f"\"priority=200,ip,nw_src={attacker_ip},actions=drop\" >/dev/null 2>&1 || true"
        )


def apply_isolate_attackers(net, attacker_host_names):
    switch = net.get("s1")
    for h_name in attacker_host_names:
        h = net.get(h_name)
        h_port = host_port(switch, h)
        os.system(
            f"ovs-ofctl -O OpenFlow13 add-flow s1 "
            f"\"priority=210,in_port={h_port},actions=drop\" >/dev/null 2>&1 || true"
        )


def apply_reroute_attackers(net, attacker_ips):
    switch = net.get("s1")
    s4 = net.get("s4")
    s3 = net.get("s3")
    server_ip = "10.0.3.2"

    s1_s4_port = host_port(switch, s4)
    s4_s3_port = host_port(s4, s3)
    s4_s1_port = host_port(s4, switch)

    os.system(
        f"ovs-ofctl -O OpenFlow13 add-flow s1 "
        f"\"priority=150,ip,nw_dst={server_ip},actions=output:{s1_s4_port}\" >/dev/null 2>&1 || true"
    )
    os.system(
        f"ovs-ofctl -O OpenFlow13 add-flow s4 "
        f"\"priority=150,ip,nw_dst={server_ip},actions=output:{s4_s3_port}\" >/dev/null 2>&1 || true"
    )
    os.system(
        f"ovs-ofctl -O OpenFlow13 add-flow s4 "
        f"\"priority=150,ip,nw_src={server_ip},actions=output:{s4_s1_port}\" >/dev/null 2>&1 || true"
    )

    for attacker_ip in attacker_ips:
        os.system(
            f"ovs-ofctl -O OpenFlow13 add-flow s4 "
            f"\"priority=200,ip,nw_src={attacker_ip},actions=drop\" >/dev/null 2>&1 || true"
        )


def apply_action(net, action_name, attacker_ips, attacker_host_names):
    if action_name == "Forward":
        apply_forward(net, "", "")
    elif action_name == "Throttle":
        apply_throttle(net, "", "")
    elif action_name == "Drop":
        apply_drop_attackers(net, attacker_ips)
    elif action_name == "Isolate":
        apply_isolate_attackers(net, attacker_host_names)
    elif action_name == "Reroute":
        apply_reroute_attackers(net, attacker_ips)


def reset_network_controls(net, benign_host_names, attacker_host_names,
                           server_ip="10.0.3.2", bottleneck_mbps=40):
    with suppress(Exception):
        clear_flows()
    with suppress(Exception):
        install_l2_forwarding(net, server_ip)
    with suppress(Exception):
        clear_mirror()
    with suppress(Exception):
        for h_name in benign_host_names + attacker_host_names:
            clear_tc(net.get(h_name))
    with suppress(Exception):
        edge = net.get("s1")
        agg = net.get("s2")
        intf_name = link_intf_name(edge, agg)
        os.system(f"tc qdisc del dev {intf_name} root >/dev/null 2>&1 || true")
        os.system(
            f"tc qdisc replace dev {intf_name} root handle 5: htb default 10 "
            f"r2q {max(1, bottleneck_mbps * 1000 // 1600)} >/dev/null 2>&1"
        )
        os.system(
            f"tc class replace dev {intf_name} parent 5: classid 5:10 "
            f"htb rate {bottleneck_mbps}mbit ceil {bottleneck_mbps}mbit >/dev/null 2>&1"
        )
    time.sleep(0.3)


def measure_step(net, action_name, attacker_ips, attacker_host_names,
                 benign_host_names, server_ip, step_id, bottleneck_mbps=40):
    reset_network_controls(net, benign_host_names, attacker_host_names,
                           server_ip=server_ip, bottleneck_mbps=bottleneck_mbps)

    flows_before_s1 = count_flow_entries("s1")
    ovs_cpu_before = get_cpu_usage()

    t0 = time.perf_counter()
    apply_action(net, action_name, attacker_ips, attacker_host_names)
    apply_ms = (time.perf_counter() - t0) * 1000.0
    time.sleep(0.2)

    flows_after_s1 = count_flow_entries("s1")
    new_flow_entries = max(0, flows_after_s1 - flows_before_s1)

    benign1 = net.get(benign_host_names[0])
    benign2 = net.get(benign_host_names[1])
    benign1_tcp = run_iperf3_tcp(benign1, server_ip, duration=1, port=5201)
    benign2_udp = run_iperf3_udp(benign2, server_ip, bandwidth="50M", duration=1, port=5203)

    attack_tcp_results = []
    attack_udp_results = []
    for i, att_name in enumerate(attacker_host_names):
        att = net.get(att_name)
        att_tcp = run_iperf3_tcp(att, server_ip, duration=1, port=5202 + i)
        att_udp = run_iperf3_udp(att, server_ip, bandwidth="50M", duration=1, port=5204 + i)
        attack_tcp_results.append(att_tcp["throughput_mbps"])
        attack_udp_results.append(att_udp["throughput_mbps"])

    benign_rtt = run_ping(benign1, server_ip)
    ovs_cpu_after = get_cpu_usage()

    is_aggressive = action_name in ("Drop", "Isolate")

    return {
        "control_apply_ms": apply_ms,
        "benign_tcp_mbps": benign1_tcp["throughput_mbps"],
        "benign_udp_mbps": benign2_udp["throughput_mbps"],
        "attack_tcp_mbps": sum(attack_tcp_results) / len(attack_tcp_results),
        "attack_udp_mbps": sum(attack_udp_results) / len(attack_udp_results),
        "benign_loss_pct": benign_rtt["packet_loss_pct"],
        "benign_rtt_ms": benign_rtt["avg_rtt_ms"],
        "ovs_flow_entries_new": new_flow_entries,
        "ovs_flow_entries_total_s1": flows_after_s1,
        "ovs_cpu_pct": ovs_cpu_after,
        "is_aggressive": int(is_aggressive),
    }


def run_sensitivity_replay(net, controller_df, attacker_ips, attacker_host_names,
                           benign_host_names, server_ip, bottleneck_mbps=40):
    rows = []
    for _, row in controller_df.sort_values("replay_step").iterrows():
        metrics = measure_step(
            net, row["action_name"], attacker_ips, attacker_host_names,
            benign_host_names, server_ip, int(row["replay_step"]),
            bottleneck_mbps=bottleneck_mbps
        )
        rows.append({**row.to_dict(), **metrics})
        if len(rows) % 25 == 0:
            print(f"  Step {len(rows)}/{len(controller_df)}")
    return pd.DataFrame(rows)


def summarize_sensitivity(df, num_attackers):
    benign = df[df["true_label"] == 0]
    attack = df[df["true_label"] == 1]
    nocontrol_attack_tcp = df[df["controller_family"] == "NoControl"]["attack_tcp_mbps"].mean() if len(df[df["controller_family"] == "NoControl"]) > 0 else 1.0
    return {
        "controller": df["controller"].iloc[0],
        "controller_family": df["controller_family"].iloc[0],
        "seed": df["seed"].iloc[0] if "seed" in df.columns else "",
        "steps": len(df),
        "benign_steps": len(benign),
        "attack_steps": len(attack),
        "num_attackers": num_attackers,
        "benign_tcp_mbps": float(benign["benign_tcp_mbps"].mean()) if len(benign) else 0.0,
        "benign_udp_mbps": float(benign["benign_udp_mbps"].mean()) if len(benign) else 0.0,
        "attack_tcp_mbps": float(attack["attack_tcp_mbps"].mean()) if len(attack) else 0.0,
        "attack_udp_mbps": float(attack["attack_udp_mbps"].mean()) if len(attack) else 0.0,
        "benign_loss_pct": float(benign["benign_loss_pct"].mean()) if len(benign) else 0.0,
        "attack_suppression_ratio": float(
            1.0 - attack["attack_tcp_mbps"].mean() / max(nocontrol_attack_tcp, 0.01)
        ) if len(attack) else 0.0,
        "mean_rtt_ms": float(benign["benign_rtt_ms"].mean()) if len(benign) and benign["benign_rtt_ms"].notna().any() else None,
        "mean_rule_install_ms": float(df["control_apply_ms"].mean()),
        "mean_ovs_flow_entries": float(df["ovs_flow_entries_total_s1"].mean()),
        "ovs_cpu_pct": float(df["ovs_cpu_pct"].mean()),
        "aggressive_on_benign_rate": float(
            benign["is_aggressive"].mean() if len(benign) else 0.0
        ),
        "aggressive_on_attack_rate": float(
            attack["is_aggressive"].mean() if len(attack) else 0.0
        ),
        "rule_changes_per_step": float(df["ovs_flow_entries_new"].mean()),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan-csv", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--bottleneck-mbps", type=int, default=40)
    parser.add_argument("--num-attackers", type=int, default=2)
    args = parser.parse_args()

    setLogLevel("warning")
    os.makedirs(args.out_dir, exist_ok=True)

    plan_df = pd.read_csv(args.plan_csv)
    all_step_frames = []
    summary_rows = []

    attacker_host_names = [f"attacker{i+1}" for i in range(args.num_attackers)]
    attacker_ips = [f"10.0.1.{3+i}" for i in range(args.num_attackers)]
    benign_host_names = ["benign1", "benign2"]
    server_ip = "10.0.3.2"
    all_hosts = benign_host_names + attacker_host_names + ["ids", "server"]

    for controller_name, controller_df in plan_df.groupby("controller", sort=False):
        print(f"\n=== Running controller: {controller_name} ({len(controller_df)} steps) ===")
        print(f"  Bottleneck: {args.bottleneck_mbps} Mbps, Attackers: {args.num_attackers}")

        topo = SensitivityEdgeTopo(bottleneck_mbps=args.bottleneck_mbps,
                                    num_attackers=args.num_attackers)
        net = Mininet(
            topo=topo,
            switch=OVSSwitch,
            controller=None,
            autoSetMacs=True,
            autoStaticArp=True,
            cleanup=True,
            link=TCLink,
        )
        try:
            net.start()
            reset_network_controls(net, benign_host_names, attacker_host_names,
                                   server_ip=server_ip,
                                   bottleneck_mbps=args.bottleneck_mbps)
            start_iperf_servers(net.get("server"), num_attackers=args.num_attackers)
            step_df = run_sensitivity_replay(
                net, controller_df.copy(), attacker_ips, attacker_host_names,
                benign_host_names, server_ip, bottleneck_mbps=args.bottleneck_mbps
            )
            step_df["bottleneck_mbps"] = args.bottleneck_mbps
            step_df["num_attackers"] = args.num_attackers
            all_step_frames.append(step_df)
            summary_row = summarize_sensitivity(step_df, args.num_attackers)
            summary_row["bottleneck_mbps"] = args.bottleneck_mbps
            summary_rows.append(summary_row)
            print(f"  Summary: {summary_row}")
        finally:
            with suppress(Exception):
                stop_iperf_processes(net, all_hosts)
            with suppress(Exception):
                reset_network_controls(net, benign_host_names, attacker_host_names,
                                       server_ip=server_ip,
                                       bottleneck_mbps=args.bottleneck_mbps)
            with suppress(Exception):
                net.stop()
            with suppress(Exception):
                os.system("mn -c >/dev/null 2>&1 || true")

    all_steps = pd.concat(all_step_frames, ignore_index=True)
    summary_df = pd.DataFrame(summary_rows)

    all_steps.to_csv(os.path.join(args.out_dir, "sensitivity_step_trace.csv"), index=False)
    summary_df.to_csv(os.path.join(args.out_dir, "sensitivity_summary.csv"), index=False)

    print("\n=== Sensitivity Replay Complete ===")
    print(summary_df.to_string(index=False))


if __name__ == "__main__":
    main()
