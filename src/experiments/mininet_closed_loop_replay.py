"""
Small controller-in-the-loop Mininet replay for Edge-IIoTset windows.

The replay plan is exported separately (potentially on a different machine with
the trained DQN checkpoints available). This script focuses only on packet-level
execution: it reads the per-window controller actions, applies each action to a
target host through OVS/tc, and measures benign throughput, attack throughput,
packet loss, and control-application latency.
"""
import argparse
import json
import os
import re
import time
from contextlib import suppress

import pandas as pd

from mininet.link import TCLink
from mininet.log import setLogLevel
from mininet.net import Mininet
from mininet.node import OVSSwitch
from mininet.topo import Topo


class MinimalEdgeTopo(Topo):
    def build(self):
        client = self.addHost("client", ip="10.0.0.1/24")
        attacker = self.addHost("attacker", ip="10.0.0.2/24")
        ids = self.addHost("ids", ip="10.0.0.3/24")
        server = self.addHost("server", ip="10.0.0.4/24")
        switch = self.addSwitch("s1", protocols="OpenFlow13", failMode="standalone")

        for host in (client, attacker, ids, server):
            self.addLink(
                host,
                switch,
                cls=TCLink,
                bw=100,
                delay="2ms",
                loss=0,
                use_htb=True,
            )


class TwoSwitchBottleneckTopo(Topo):
    def build(self):
        client = self.addHost("client", ip="10.0.0.1/24")
        attacker = self.addHost("attacker", ip="10.0.0.2/24")
        ids = self.addHost("ids", ip="10.0.0.3/24")
        server = self.addHost("server", ip="10.0.0.4/24")
        edge = self.addSwitch("s1", protocols="OpenFlow13", failMode="standalone")
        agg = self.addSwitch("s2", protocols="OpenFlow13", failMode="standalone")

        self.addLink(client, edge, cls=TCLink, bw=100, delay="2ms", loss=0, use_htb=True)
        self.addLink(attacker, edge, cls=TCLink, bw=100, delay="2ms", loss=0, use_htb=True)
        self.addLink(ids, edge, cls=TCLink, bw=100, delay="2ms", loss=0, use_htb=True)
        self.addLink(edge, agg, cls=TCLink, bw=40, delay="5ms", loss=0, use_htb=True)
        self.addLink(server, agg, cls=TCLink, bw=100, delay="2ms", loss=0, use_htb=True)


def parse_ping(output):
    loss_match = re.search(r"(\d+(?:\.\d+)?)% packet loss", output)
    rtt_match = re.search(r"= ([\d\.]+)/([\d\.]+)/([\d\.]+)/([\d\.]+) ms", output)
    return {
        "packet_loss_pct": float(loss_match.group(1)) if loss_match else 100.0,
        "avg_rtt_ms": float(rtt_match.group(2)) if rtt_match else None,
        "raw": output.strip(),
    }


def run_ping(host, target_ip, count=1, timeout_s=4):
    output = host.cmd(f"timeout {timeout_s}s ping -c {count} -i 0.2 -W 1 {target_ip} || true")
    return parse_ping(output)


def run_iperf3(client, target_ip, duration=1, timeout_s=4, port=5201):
    output = client.cmd(
        f"timeout {timeout_s}s iperf3 -J --connect-timeout 1200 -t {duration} -p {port} -c {target_ip} || true"
    )
    start = output.find("{")
    end = output.rfind("}")
    if start == -1 or end == -1:
        return {"throughput_mbps": 0.0, "retransmits": None, "raw": output.strip()}

    payload = json.loads(output[start:end + 1])
    summary = (
        payload.get("end", {}).get("sum_received")
        or payload.get("end", {}).get("sum")
        or payload.get("end", {}).get("sum_sent")
        or {}
    )
    return {
        "throughput_mbps": float(summary.get("bits_per_second", 0.0)) / 1e6,
        "retransmits": summary.get("retransmits"),
        "raw": output.strip(),
    }


def read_rx_bytes(host, intf_name):
    output = host.cmd(f"cat /sys/class/net/{intf_name}/statistics/rx_bytes").strip()
    with suppress(ValueError):
        return int(output)
    return 0


def clear_tc(host):
    host.cmd(f"tc qdisc del dev {host.defaultIntf().name} root >/dev/null 2>&1 || true")


def clear_mirror():
    os.system("ovs-vsctl clear Bridge s1 mirrors >/dev/null 2>&1 || true")


def clear_flows():
    os.system("ovs-ofctl -O OpenFlow13 del-flows s1 >/dev/null 2>&1 || true")


def install_default_forwarding():
    os.system(
        "ovs-ofctl -O OpenFlow13 add-flow s1 \"priority=0,actions=NORMAL\" >/dev/null 2>&1 || true"
    )
    os.system(
        "ovs-ofctl -O OpenFlow13 add-flow s2 \"priority=0,actions=NORMAL\" >/dev/null 2>&1 || true"
    )


def start_iperf_server(server):
    server.cmd("pkill -f 'iperf3 -s' >/dev/null 2>&1 || true")
    server.cmd("iperf3 -s -p 5201 -D")
    server.cmd("iperf3 -s -p 5202 -D")
    time.sleep(0.6)


def stop_iperf_processes(net):
    for host_name in ("client", "attacker", "server"):
        net.get(host_name).cmd("pkill -f iperf3 >/dev/null 2>&1 || true")


def host_port(switch, host):
    return switch.ports[switch.connectionsTo(host)[0][0]]


def link_intf_name(node_a, node_b):
    intf_a, _ = node_a.connectionsTo(node_b)[0]
    return intf_a.name


def apply_forward(_net, _target_name, _mode=None):
    return


def apply_mirror(net, target_name, mode=None):
    switch = net.get("s1")
    ids = net.get("ids")
    ids_port = host_port(switch, ids)

    if mode == "mixed":
        client = net.get("client")
        attacker = net.get("attacker")
        client_port = host_port(switch, client)
        attacker_port = host_port(switch, attacker)
        cmd = (
            "ovs-vsctl "
            f"-- --id=@c get Port s1-eth{client_port} "
            f"-- --id=@a get Port s1-eth{attacker_port} "
            f"-- --id=@dst get Port s1-eth{ids_port} "
            "-- --id=@m create Mirror name=m0 select-src-port=@c,@a select-dst-port=@c,@a output-port=@dst "
            "-- set Bridge s1 mirrors=@m >/dev/null 2>&1"
        )
    else:
        target = net.get(target_name)
        src_port = host_port(switch, target)
        cmd = (
            "ovs-vsctl "
            f"-- --id=@src get Port s1-eth{src_port} "
            f"-- --id=@dst get Port s1-eth{ids_port} "
            "-- --id=@m create Mirror name=m0 select-src-port=@src select-dst-port=@src output-port=@dst "
            "-- set Bridge s1 mirrors=@m >/dev/null 2>&1"
        )
    os.system(cmd)


def apply_throttle(net, target_name, mode=None, rate_mbit=5):
    if mode == "mixed":
        edge = net.get("s1")
        agg = net.get("s2")
        intf_name = link_intf_name(edge, agg)
        os.system(
            f"tc qdisc replace dev {intf_name} root tbf rate {rate_mbit}mbit burst 32kbit latency 80ms >/dev/null 2>&1"
        )
    else:
        target = net.get(target_name)
        target.cmd(
            f"tc qdisc replace dev {target.defaultIntf().name} root tbf rate {rate_mbit}mbit burst 32kbit latency 80ms"
        )


def apply_drop(net, target_name, mode=None):
    switch = net.get("s1")
    if mode == "mixed":
        os.system(
            "ovs-ofctl -O OpenFlow13 add-flow s1 \"priority=200,ip,nw_dst=10.0.0.4,actions=drop\" >/dev/null 2>&1"
        )
    else:
        target = net.get(target_name)
        target_port = host_port(switch, target)
        os.system(
            f"ovs-ofctl -O OpenFlow13 add-flow s1 \"priority=200,in_port={target_port},ip,nw_dst=10.0.0.4,actions=drop\" >/dev/null 2>&1"
        )


def apply_isolate(net, target_name, mode=None):
    switch = net.get("s1")
    if mode == "mixed":
        client = net.get("client")
        attacker = net.get("attacker")
        client_port = host_port(switch, client)
        attacker_port = host_port(switch, attacker)
        os.system(
            f"ovs-ofctl -O OpenFlow13 add-flow s1 \"priority=210,in_port={client_port},actions=drop\" >/dev/null 2>&1"
        )
        os.system(
            f"ovs-ofctl -O OpenFlow13 add-flow s1 \"priority=210,in_port={attacker_port},actions=drop\" >/dev/null 2>&1"
        )
    else:
        target = net.get(target_name)
        target_port = host_port(switch, target)
        os.system(
            f"ovs-ofctl -O OpenFlow13 add-flow s1 \"priority=200,in_port={target_port},actions=drop\" >/dev/null 2>&1"
        )


ACTION_APPLIERS = {
    "Forward": apply_forward,
    "Inspect": apply_forward,
    "Mirror": apply_mirror,
    "Throttle": apply_throttle,
    "Reroute": apply_forward,
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
        clear_tc(net.get("client"))
    with suppress(Exception):
        clear_tc(net.get("attacker"))
    with suppress(Exception):
        edge = net.get("s1")
        agg = net.get("s2")
        intf_name = link_intf_name(edge, agg)
        os.system(f"tc qdisc del dev {intf_name} root >/dev/null 2>&1 || true")
    time.sleep(0.1)


def run_dual_iperf(net, duration=1):
    client = net.get("client")
    attacker = net.get("attacker")
    server_ip = "10.0.0.4"

    proc_client = client.popen(
        f"timeout 4s iperf3 -J --connect-timeout 1200 -t {duration} -p 5201 -c {server_ip}",
        shell=True,
        stdout=-1,
        stderr=-1,
        text=True,
    )
    proc_attacker = attacker.popen(
        f"timeout 4s iperf3 -J --connect-timeout 1200 -t {duration} -p 5202 -c {server_ip}",
        shell=True,
        stdout=-1,
        stderr=-1,
        text=True,
    )

    out_client, _ = proc_client.communicate()
    out_attacker, _ = proc_attacker.communicate()

    def parse_output(output):
        start = output.find("{")
        end = output.rfind("}")
        if start == -1 or end == -1:
            return {"throughput_mbps": 0.0}
        payload = json.loads(output[start:end + 1])
        summary = (
            payload.get("end", {}).get("sum_received")
            or payload.get("end", {}).get("sum")
            or payload.get("end", {}).get("sum_sent")
            or {}
        )
        return {"throughput_mbps": float(summary.get("bits_per_second", 0.0)) / 1e6}

    return parse_output(out_client), parse_output(out_attacker)


def measure_step(net, action_name, target_name, mode):
    client = net.get("client")
    attacker = net.get("attacker")
    ids = net.get("ids")

    reset_network_controls(net)

    ids_intf = ids.defaultIntf().name
    ids_before = read_rx_bytes(ids, ids_intf)

    t0 = time.perf_counter()
    ACTION_APPLIERS[action_name](net, target_name, mode)
    apply_ms = (time.perf_counter() - t0) * 1000.0
    time.sleep(0.2)

    benign_ping = run_ping(client, "10.0.0.4")
    attack_ping = run_ping(attacker, "10.0.0.4")
    if mode == "mixed":
        benign_iperf, attack_iperf = run_dual_iperf(net, duration=1)
    else:
        benign_iperf = run_iperf3(client, "10.0.0.4", duration=1, port=5201)
        attack_iperf = run_iperf3(attacker, "10.0.0.4", duration=1, port=5202)

    ids_after = read_rx_bytes(ids, ids_intf)

    return {
        "control_apply_ms": apply_ms,
        "benign_ping_avg_rtt_ms": benign_ping["avg_rtt_ms"],
        "benign_ping_loss_pct": benign_ping["packet_loss_pct"],
        "benign_tcp_throughput_mbps": benign_iperf["throughput_mbps"],
        "attack_ping_avg_rtt_ms": attack_ping["avg_rtt_ms"],
        "attack_ping_loss_pct": attack_ping["packet_loss_pct"],
        "attack_tcp_throughput_mbps": attack_iperf["throughput_mbps"],
        "ids_mirror_rx_bytes_delta": int(max(ids_after - ids_before, 0)),
    }


def summarize_controller(df):
    benign_rows = df[df["true_label"] == 0]
    attack_rows = df[df["true_label"] == 1]
    return {
        "controller": df["controller"].iloc[0],
        "controller_family": df["controller_family"].iloc[0],
        "seed": df["seed"].iloc[0] if "seed" in df.columns else "",
        "steps": int(len(df)),
        "benign_steps": int(len(benign_rows)),
        "attack_steps": int(len(attack_rows)),
        "benign_tcp_mean_mbps": float(benign_rows["benign_tcp_throughput_mbps"].mean()),
        "benign_loss_mean_pct": float(benign_rows["benign_ping_loss_pct"].mean()),
        "attack_tcp_mean_mbps": float(attack_rows["attack_tcp_throughput_mbps"].mean()),
        "attack_loss_mean_pct": float(attack_rows["attack_ping_loss_pct"].mean()),
        "control_apply_mean_ms": float(df["control_apply_ms"].mean()),
        "mirror_rx_mean_bytes": float(df["ids_mirror_rx_bytes_delta"].mean()),
        "aggressive_on_benign_rate": float(
            benign_rows["action_name"].isin(["Drop", "Isolate"]).mean() if len(benign_rows) else 0.0
        ),
        "aggressive_on_attack_rate": float(
            attack_rows["action_name"].isin(["Drop", "Isolate"]).mean() if len(attack_rows) else 0.0
        ),
    }


def run_controller_replay(net, controller_df, mode):
    rows = []
    for row in controller_df.sort_values("replay_step").to_dict("records"):
        target_name = "shared_bottleneck" if mode == "mixed" else ("client" if int(row["true_label"]) == 0 else "attacker")
        metrics = measure_step(net, row["action_name"], target_name, mode)
        rows.append(
            {
                **row,
                "target_name": target_name,
                **metrics,
            }
        )
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan-csv", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--mode", choices=["targeted", "mixed"], default="targeted")
    parser.add_argument("--topology", choices=["single", "two_switch"], default="single")
    args = parser.parse_args()

    setLogLevel("warning")
    os.makedirs(args.out_dir, exist_ok=True)

    plan_df = pd.read_csv(args.plan_csv)
    all_step_frames = []
    summary_rows = []

    for controller_name, controller_df in plan_df.groupby("controller", sort=False):
        topo = MinimalEdgeTopo() if args.topology == "single" else TwoSwitchBottleneckTopo()
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
            start_iperf_server(net.get("server"))
            step_df = run_controller_replay(net, controller_df.copy(), args.mode)
            step_df["replay_mode"] = args.mode
            step_df["topology"] = args.topology
            all_step_frames.append(step_df)
            summary_row = summarize_controller(step_df)
            summary_row["replay_mode"] = args.mode
            summary_row["topology"] = args.topology
            summary_rows.append(summary_row)
        finally:
            with suppress(Exception):
                stop_iperf_processes(net)
            with suppress(Exception):
                reset_network_controls(net)
            with suppress(Exception):
                net.stop()
            with suppress(Exception):
                os.system("mn -c >/dev/null 2>&1 || true")

    all_steps = pd.concat(all_step_frames, ignore_index=True)
    summary_df = pd.DataFrame(summary_rows)

    all_steps.to_csv(os.path.join(args.out_dir, "closed_loop_step_trace.csv"), index=False)
    summary_df.to_csv(os.path.join(args.out_dir, "closed_loop_summary.csv"), index=False)

    family_summary_rows = []
    for family, group in summary_df.groupby("controller_family", sort=False):
        row = {"controller_family": family}
        for metric in [
            "benign_tcp_mean_mbps",
            "benign_loss_mean_pct",
            "attack_tcp_mean_mbps",
            "attack_loss_mean_pct",
            "control_apply_mean_ms",
            "aggressive_on_benign_rate",
            "aggressive_on_attack_rate",
        ]:
            row[f"{metric}_mean"] = float(group[metric].mean())
            row[f"{metric}_std"] = float(group[metric].std(ddof=0)) if len(group) > 1 else 0.0
        family_summary_rows.append(row)
    pd.DataFrame(family_summary_rows).to_csv(
        os.path.join(args.out_dir, "closed_loop_family_summary.csv"),
        index=False,
    )

    print(summary_df.to_string(index=False))


if __name__ == "__main__":
    main()
