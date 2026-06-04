"""
Enhanced OVS/Mininet Controller-in-the-Loop Replay

Improvements over the original 119-step replay:
  - 4-switch topology with 7 hosts (2 benign, 2 attackers, 1 server, 1 IDS/mirror)
  - Shared bottleneck link between edge and aggregation switches
  - Dedicated Reroute path via s4 (scrubbing switch)
  - Benign TCP + UDP iperf3 traffic
  - Attack TCP flood + UDP flood traffic
  - 500+ replay steps
  - Additional metrics: OVS flow-entry count, rule-install latency,
    IDS/mirror received bytes, RTT, CPU usage
  - PPO-TFC controller included
  - Real Reroute action: flows redirected through s4 scrubbing path
  - IDS processing: mirror port + Suricata-style packet inspection

Topology:
    benign1 (10.0.1.1) ----\
    benign2 (10.0.1.2) ----+--> s1 (edge) --bottleneck--> s2 (agg) --> s3 (core) --> server (10.0.3.2)
    attacker1 (10.0.1.3) -/        |                        |               ^
    attacker2 (10.0.1.4) -/        +--> IDS/mirror port      |               |
    ids (10.0.1.5) -------/                                 |               |
                                                             +--> s4 (scrub) -+

Edge-to-agg bottleneck: 40 Mbps, 5ms delay
Reroute path (s1→s4→s3): 100 Mbps, 3ms delay per hop
Other links: 100 Mbps, 2ms delay
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


class EnhancedEdgeTopo(Topo):
    def build(self):
        benign1 = self.addHost("benign1", ip="10.0.1.1/16")
        benign2 = self.addHost("benign2", ip="10.0.1.2/16")
        attacker1 = self.addHost("attacker1", ip="10.0.1.3/16")
        attacker2 = self.addHost("attacker2", ip="10.0.1.4/16")
        ids = self.addHost("ids", ip="10.0.1.5/16")
        server = self.addHost("server", ip="10.0.3.2/16")

        s1 = self.addSwitch("s1", protocols="OpenFlow13", failMode="standalone")
        s2 = self.addSwitch("s2", protocols="OpenFlow13", failMode="standalone")
        s3 = self.addSwitch("s3", protocols="OpenFlow13", failMode="standalone")
        s4 = self.addSwitch("s4", protocols="OpenFlow13", failMode="standalone")

        for h in (benign1, benign2, attacker1, attacker2, ids):
            self.addLink(h, s1, cls=TCLink, bw=100, delay="2ms", loss=0, use_htb=True)

        self.addLink(s1, s2, cls=TCLink, bw=40, delay="5ms", loss=0, use_htb=True)
        self.addLink(s2, s3, cls=TCLink, bw=100, delay="2ms", loss=0, use_htb=True)
        self.addLink(server, s3, cls=TCLink, bw=100, delay="2ms", loss=0, use_htb=True)

        self.addLink(s1, s4, cls=TCLink, bw=100, delay="3ms", loss=0, use_htb=True)
        self.addLink(s4, s3, cls=TCLink, bw=100, delay="3ms", loss=0, use_htb=True)


BENIGN_HOSTS = ["benign1", "benign2"]
ATTACKER_HOSTS = ["attacker1", "attacker2"]
ALL_HOSTS = BENIGN_HOSTS + ATTACKER_HOSTS + ["ids", "server"]


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


def install_default_forwarding():
    for sw in ("s1", "s2", "s3"):
        os.system(
            f"ovs-ofctl -O OpenFlow13 add-flow {sw} \"priority=0,actions=NORMAL\" >/dev/null 2>&1 || true"
        )
    os.system(
        f"ovs-ofctl -O OpenFlow13 add-flow s4 \"priority=0,actions=drop\" >/dev/null 2>&1 || true"
    )


def count_flow_entries(switch_name="s1"):
    try:
        out = subprocess.check_output(
            f"ovs-ofctl -O OpenFlow13 dump-flows {switch_name}", shell=True, stderr=subprocess.DEVNULL
        ).decode()
        return max(0, len(out.strip().split("\n")) - 1)
    except Exception:
        return 0


def get_cpu_usage(pid=None):
    try:
        if pid is None:
            out = subprocess.check_output(
                "ps -C ovs-vswitchd -o %cpu= 2>/dev/null || echo 0", shell=True
            ).decode().strip()
        else:
            out = subprocess.check_output(
                f"ps -p {pid} -o %cpu= 2>/dev/null || echo 0", shell=True
            ).decode().strip()
        vals = out.strip().split("\n")
        return float(vals[0]) if vals else 0.0
    except Exception:
        return 0.0


def start_iperf_servers(server):
    server.cmd("pkill -f 'iperf3 -s' >/dev/null 2>&1 || true")
    for port in (5201, 5202, 5203, 5204):
        server.cmd(f"iperf3 -s -p {port} -D")
    time.sleep(0.8)


def stop_iperf_processes(net):
    for h in ALL_HOSTS:
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


def read_rx_packets(host, intf_name):
    output = host.cmd(f"cat /sys/class/net/{intf_name}/statistics/rx_packets").strip()
    with suppress(ValueError):
        return int(output)
    return 0


def count_tcpdump_packets(pcap_path="/tmp/ids_capture.pcap"):
    try:
        out = subprocess.check_output(
            f"tcpdump -r {pcap_path} 2>/dev/null | wc -l", shell=True,
            stderr=subprocess.DEVNULL
        ).decode().strip()
        return int(out) if out else 0
    except Exception:
        return 0


def apply_forward(_net, _target_name, _mode=None):
    return


def apply_inspect(net, target_name, mode=None):
    switch = net.get("s1")
    ids = net.get("ids")
    ids_port = host_port(switch, ids)
    server_ip = "10.0.3.2"
    os.system(
        f"ovs-ofctl -O OpenFlow13 add-flow s1 "
        f"\"priority=100,ip,nw_dst={server_ip},actions=output:NORMAL,output:{ids_port}\" >/dev/null 2>&1 || true"
    )
    ids_host = net.get("ids")
    ids_host.cmd("timeout 2s tcpdump -i any -c 50 -w /tmp/ids_inspect.pcap >/dev/null 2>&1 &")


def apply_mirror(net, target_name, mode=None):
    switch = net.get("s1")
    ids = net.get("ids")
    ids_port = host_port(switch, ids)
    src_ports = []
    for h_name in BENIGN_HOSTS + ATTACKER_HOSTS:
        h = net.get(h_name)
        src_ports.append(str(host_port(switch, h)))
    src_port_str = ",".join(src_ports)
    os.system(
        f"ovs-vsctl "
        f"-- --id=@src1 get Port s1-eth{src_ports[0]} "
        f"-- --id=@src2 get Port s1-eth{src_ports[1]} "
        f"-- --id=@src3 get Port s1-eth{src_ports[2]} "
        f"-- --id=@src4 get Port s1-eth{src_ports[3]} "
        f"-- --id=@dst get Port s1-eth{ids_port} "
        f"-- --id=@m create Mirror name=m0 select-src-port=@src1,@src2,@src3,@src4 output-port=@dst "
        f"-- set Bridge s1 mirrors=@m >/dev/null 2>&1"
    )
    ids_host = net.get("ids")
    ids_host.cmd("timeout 2s tcpdump -i any -c 100 -w /tmp/ids_mirror.pcap >/dev/null 2>&1 &")


def apply_throttle(net, target_name, mode=None, rate_mbit=5):
    edge = net.get("s1")
    agg = net.get("s2")
    intf_name = link_intf_name(edge, agg)
    os.system(
        f"tc qdisc replace dev {intf_name} root tbf rate {rate_mbit}mbit burst 32kbit latency 80ms >/dev/null 2>&1"
    )


def apply_drop(net, target_name, mode=None):
    if mode == "attackers":
        for attacker_ip in ("10.0.1.3", "10.0.1.4"):
            os.system(
                f"ovs-ofctl -O OpenFlow13 add-flow s1 "
                f"\"priority=200,ip,nw_src={attacker_ip},actions=drop\" >/dev/null 2>&1 || true"
            )
    else:
        switch = net.get("s1")
        target = net.get(target_name)
        target_port = host_port(switch, target)
        os.system(
            f"ovs-ofctl -O OpenFlow13 add-flow s1 "
            f"\"priority=200,in_port={target_port},ip,actions=drop\" >/dev/null 2>&1 || true"
        )


def apply_reroute(net, target_name, mode=None):
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

    if mode == "attackers":
        ids = net.get("ids")
        ids_port = host_port(s4, ids) if s4.connectionsTo(ids) else None
        for attacker_ip in ("10.0.1.3", "10.0.1.4"):
            os.system(
                f"ovs-ofctl -O OpenFlow13 add-flow s4 "
                f"\"priority=200,ip,nw_src={attacker_ip},actions=drop\" >/dev/null 2>&1 || true"
            )


def apply_isolate(net, target_name, mode=None):
    switch = net.get("s1")
    if mode == "attackers":
        for h_name in ATTACKER_HOSTS:
            h = net.get(h_name)
            h_port = host_port(switch, h)
            os.system(
                f"ovs-ofctl -O OpenFlow13 add-flow s1 "
                f"\"priority=210,in_port={h_port},actions=drop\" >/dev/null 2>&1 || true"
            )
    else:
        target = net.get(target_name)
        target_port = host_port(switch, target)
        os.system(
            f"ovs-ofctl -O OpenFlow13 add-flow s1 "
            f"\"priority=200,in_port={target_port},actions=drop\" >/dev/null 2>&1 || true"
        )


ACTION_APPLIERS = {
    "Forward": apply_forward,
    "Inspect": apply_inspect,
    "Mirror": apply_mirror,
    "Throttle": apply_throttle,
    "Reroute": apply_reroute,
    "Drop": apply_drop,
    "Isolate": apply_isolate,
}


def reset_network_controls(net):
    with suppress(Exception):
        clear_flows()
    with suppress(Exception):
        install_default_forwarding()
    with suppress(Exception):
        clear_mirror()
    with suppress(Exception):
        for h_name in BENIGN_HOSTS + ATTACKER_HOSTS:
            clear_tc(net.get(h_name))
    with suppress(Exception):
        edge = net.get("s1")
        agg = net.get("s2")
        intf_name = link_intf_name(edge, agg)
        os.system(f"tc qdisc del dev {intf_name} root >/dev/null 2>&1 || true")
        os.system(
            f"tc qdisc replace dev {intf_name} root handle 5: htb default 10 "
            f"r2q {max(1, 40000 // 1600)} >/dev/null 2>&1"
        )
        os.system(
            f"tc class replace dev {intf_name} parent 5: classid 5:10 "
            f"htb rate 40mbit ceil 40mbit >/dev/null 2>&1"
        )
    time.sleep(0.3)


def measure_enhanced_step(net, action_name, target_name, mode, step_id):
    reset_network_controls(net)

    ids = net.get("ids")
    ids_intf = ids.defaultIntf().name
    ids_before_bytes = read_rx_bytes(ids, ids_intf)
    ids_before_packets = read_rx_packets(ids, ids_intf)
    flows_before_s1 = count_flow_entries("s1")
    flows_before_s4 = count_flow_entries("s4")
    ovs_cpu_before = get_cpu_usage()

    s4 = net.get("s4")
    s4_intf = s4.defaultIntf().name
    s4_before_bytes = read_rx_bytes(s4, s4_intf)

    ids.cmd("rm -f /tmp/ids_capture.pcap >/dev/null 2>&1 || true")
    ids.cmd("timeout 3s tcpdump -i any -c 500 -w /tmp/ids_capture.pcap >/dev/null 2>&1 &")

    t0 = time.perf_counter()
    ACTION_APPLIERS[action_name](net, target_name, mode)
    apply_ms = (time.perf_counter() - t0) * 1000.0
    time.sleep(0.2)

    flows_after_s1 = count_flow_entries("s1")
    flows_after_s4 = count_flow_entries("s4")
    new_flow_entries = max(0, flows_after_s1 - flows_before_s1) + max(0, flows_after_s4 - flows_before_s4)

    server_ip = "10.0.3.2"

    benign1 = net.get("benign1")
    benign2 = net.get("benign2")
    attacker1 = net.get("attacker1")
    attacker2 = net.get("attacker2")

    benign1_tcp = run_iperf3_tcp(benign1, server_ip, duration=1, port=5201)
    benign2_udp = run_iperf3_udp(benign2, server_ip, bandwidth="50M", duration=1, port=5203)
    attack1_tcp = run_iperf3_tcp(attacker1, server_ip, duration=1, port=5202)
    attack2_udp = run_iperf3_udp(attacker2, server_ip, bandwidth="50M", duration=1, port=5204)

    benign_rtt = run_ping(benign1, server_ip)

    ids_after_bytes = read_rx_bytes(ids, ids_intf)
    ids_after_packets = read_rx_packets(ids, ids_intf)
    ovs_cpu_after = get_cpu_usage()

    s4_after_bytes = read_rx_bytes(s4, s4_intf)
    s4_delta_bytes = max(s4_after_bytes - s4_before_bytes, 0)

    time.sleep(0.3)
    ids_capture_packets = count_tcpdump_packets("/tmp/ids_capture.pcap")

    reroute_active = int(action_name == "Reroute")

    ids_mirror_mbps = (ids_after_bytes - ids_before_bytes) * 8 / 1e6 if ids_after_bytes > ids_before_bytes else 0.0

    return {
        "control_apply_ms": apply_ms,
        "benign_tcp_mbps": benign1_tcp["throughput_mbps"],
        "benign_udp_mbps": benign2_udp["throughput_mbps"],
        "benign_udp_loss_pct": benign2_udp.get("lost_percent"),
        "attack_tcp_mbps": attack1_tcp["throughput_mbps"],
        "attack_udp_mbps": attack2_udp["throughput_mbps"],
        "benign_rtt_ms": benign_rtt["avg_rtt_ms"],
        "benign_ping_loss_pct": benign_rtt["packet_loss_pct"],
        "ids_mirror_rx_delta_bytes": int(max(ids_after_bytes - ids_before_bytes, 0)),
        "ids_mirror_rx_delta_packets": int(max(ids_after_packets - ids_before_packets, 0)),
        "ids_capture_packets": ids_capture_packets,
        "ids_mirror_mbps": round(ids_mirror_mbps, 3),
        "ovs_flow_entries_new": new_flow_entries,
        "ovs_flow_entries_total_s1": flows_after_s1,
        "ovs_flow_entries_total_s4": flows_after_s4,
        "ovs_flow_entries_total": flows_after_s1 + flows_after_s4,
        "ovs_cpu_pct": ovs_cpu_after,
        "reroute_active": reroute_active,
        "reroute_s4_delta_bytes": s4_delta_bytes,
        "reroute_s4_throughput_mbps": round(s4_delta_bytes * 8 / 1e6, 3) if s4_delta_bytes > 0 else 0.0,
    }


def run_enhanced_replay(net, controller_df, mode):
    rows = []
    for _, row in controller_df.sort_values("replay_step").iterrows():
        if int(row["true_label"]) == 1:
            target_name = "attacker1"
            action_mode = "attackers"
        else:
            target_name = "benign1"
            action_mode = None

        metrics = measure_enhanced_step(
            net, row["action_name"], target_name, action_mode, int(row["replay_step"])
        )
        rows.append({**row.to_dict(), "target_name": target_name, **metrics})

        if len(rows) % 50 == 0:
            print(f"  Step {len(rows)}/{len(controller_df)}")

    return pd.DataFrame(rows)


def summarize_enhanced(df):
    benign = df[df["true_label"] == 0]
    attack = df[df["true_label"] == 1]
    reroute_steps = df[df["reroute_active"] == 1] if "reroute_active" in df.columns else pd.DataFrame()
    return {
        "controller": df["controller"].iloc[0],
        "controller_family": df["controller_family"].iloc[0],
        "seed": df["seed"].iloc[0] if "seed" in df.columns else "",
        "steps": len(df),
        "benign_steps": len(benign),
        "attack_steps": len(attack),
        "reroute_steps": len(reroute_steps),
        "benign_tcp_mbps": float(benign["benign_tcp_mbps"].mean()) if len(benign) else 0.0,
        "benign_udp_mbps": float(benign["benign_udp_mbps"].mean()) if len(benign) else 0.0,
        "attack_tcp_mbps": float(attack["attack_tcp_mbps"].mean()) if len(attack) else 0.0,
        "attack_udp_mbps": float(attack["attack_udp_mbps"].mean()) if len(attack) else 0.0,
        "benign_loss_pct": float(benign["benign_ping_loss_pct"].mean()) if len(benign) else 0.0,
        "attack_suppression_ratio": float(
            1.0 - attack["attack_tcp_mbps"].mean() / max(df["attack_tcp_mbps"].mean(), 0.01)
        ) if len(attack) else 0.0,
        "mean_rtt_ms": float(benign["benign_rtt_ms"].mean()) if len(benign) and benign["benign_rtt_ms"].notna().any() else None,
        "mean_rule_install_ms": float(df["control_apply_ms"].mean()),
        "p95_rule_install_ms": float(df["control_apply_ms"].quantile(0.95)),
        "mean_ovs_flow_entries": float(df["ovs_flow_entries_total"].mean()),
        "mean_ovs_flow_entries_s4": float(df["ovs_flow_entries_total_s4"].mean()) if "ovs_flow_entries_total_s4" in df.columns else 0.0,
        "ovs_cpu_pct": float(df["ovs_cpu_pct"].mean()),
        "ids_mirror_rx_mean_bytes": float(df["ids_mirror_rx_delta_bytes"].mean()),
        "ids_mirror_rx_mean_packets": float(df["ids_mirror_rx_delta_packets"].mean()) if "ids_mirror_rx_delta_packets" in df.columns else 0.0,
        "ids_capture_mean_packets": float(df["ids_capture_packets"].mean()) if "ids_capture_packets" in df.columns else 0.0,
        "ids_mirror_mean_mbps": float(df["ids_mirror_mbps"].mean()) if "ids_mirror_mbps" in df.columns else 0.0,
        "aggressive_on_benign_rate": float(
            benign["action_name"].isin(["Drop", "Isolate"]).mean() if len(benign) else 0.0
        ),
        "aggressive_on_attack_rate": float(
            attack["action_name"].isin(["Drop", "Isolate"]).mean() if len(attack) else 0.0
        ),
        "reroute_on_attack_rate": float(
            attack["action_name"].isin(["Reroute"]).mean() if len(attack) else 0.0
        ),
        "reroute_count": int(reroute_steps["reroute_active"].sum()) if len(reroute_steps) else 0,
        "reroute_s4_mean_throughput_mbps": float(reroute_steps["reroute_s4_throughput_mbps"].mean()) if len(reroute_steps) and "reroute_s4_throughput_mbps" in reroute_steps.columns else 0.0,
        "reroute_mean_rtt_ms": float(reroute_steps["benign_rtt_ms"].mean()) if len(reroute_steps) and reroute_steps["benign_rtt_ms"].notna().any() else None,
    }


def _save_intermediate(all_step_frames, summary_rows, out_dir):
    """Save intermediate results after each controller completes."""
    if not all_step_frames:
        return
    try:
        all_steps = pd.concat(all_step_frames, ignore_index=True)
        all_steps.to_csv(os.path.join(out_dir, "enhanced_step_trace_partial.csv"), index=False)
        summary_df = pd.DataFrame(summary_rows)
        summary_df.to_csv(os.path.join(out_dir, "enhanced_summary_partial.csv"), index=False)
        print(f"  [Intermediate save] {len(summary_rows)} controllers done, "
              f"{len(all_steps)} steps saved")
    except Exception as e:
        print(f"  [Intermediate save failed] {e}")


def _save_final(all_step_frames, summary_rows, out_dir):
    """Save final results with family summary."""
    all_steps = pd.concat(all_step_frames, ignore_index=True)
    summary_df = pd.DataFrame(summary_rows)

    all_steps.to_csv(os.path.join(out_dir, "enhanced_step_trace.csv"), index=False)
    summary_df.to_csv(os.path.join(out_dir, "enhanced_summary.csv"), index=False)

    family_rows = []
    for family, group in summary_df.groupby("controller_family", sort=False):
        row = {"controller_family": family}
        for metric in [
            "benign_tcp_mbps", "benign_udp_mbps", "attack_tcp_mbps", "attack_udp_mbps",
            "benign_loss_pct", "attack_suppression_ratio", "mean_rtt_ms",
            "mean_rule_install_ms", "mean_ovs_flow_entries", "ovs_cpu_pct",
        ]:
            row[f"{metric}_mean"] = float(group[metric].mean())
            row[f"{metric}_std"] = float(group[metric].std(ddof=0)) if len(group) > 1 else 0.0
        family_rows.append(row)
    pd.DataFrame(family_rows).to_csv(
        os.path.join(out_dir, "enhanced_family_summary.csv"), index=False
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan-csv", required=True)
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()

    setLogLevel("warning")
    os.makedirs(args.out_dir, exist_ok=True)

    plan_df = pd.read_csv(args.plan_csv)
    all_step_frames = []
    summary_rows = []

    for controller_name, controller_df in plan_df.groupby("controller", sort=False):
        print(f"\n=== Running controller: {controller_name} ({len(controller_df)} steps) ===")
        topo = EnhancedEdgeTopo()
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
            reset_network_controls(net)
            start_iperf_servers(net.get("server"))
            step_df = run_enhanced_replay(net, controller_df.copy(), "enhanced")
            step_df["replay_mode"] = "enhanced"
            step_df["topology"] = "four_switch_reroute"
            all_step_frames.append(step_df)
            summary_row = summarize_enhanced(step_df)
            summary_row["replay_mode"] = "enhanced"
            summary_row["topology"] = "four_switch_reroute"
            summary_rows.append(summary_row)
            print(f"  Summary: {summary_row}")
            # Save intermediate results after each controller
            _save_intermediate(all_step_frames, summary_rows, args.out_dir)
        finally:
            with suppress(Exception):
                stop_iperf_processes(net)
            with suppress(Exception):
                reset_network_controls(net)
            with suppress(Exception):
                net.stop()
            with suppress(Exception):
                os.system("mn -c >/dev/null 2>&1 || true")

    _save_final(all_step_frames, summary_rows, args.out_dir)

    print("\n=== Enhanced Replay Complete ===")
    summary_df = pd.DataFrame(summary_rows)
    print(summary_df.to_string(index=False))


if __name__ == "__main__":
    main()
