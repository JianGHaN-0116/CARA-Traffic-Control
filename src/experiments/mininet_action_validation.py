"""
Minimal Mininet-based validation of action-cost ordering.

Topology:
    benign client ----\
                        \ 
    attacker ---------- edge switch ---- server
                        /
    ids/mirror node ---/

This script validates whether the simulator's action-cost intuition is at least
qualitatively aligned with a small packet-level emulation. It measures:
    - benign ping latency and loss to the server
    - benign TCP throughput to the server
    - attacker ping latency and loss to the server
    - attacker TCP throughput to the server
    - mirrored bytes observed on the IDS interface

Implemented actions:
    - Forward: default switching behavior
    - Mirror: OVS port mirroring from attacker-facing switch port to IDS port
    - Throttle: tc-based egress rate limit on the attacker host interface
    - Drop: OpenFlow drop rule on attacker -> server traffic
    - Isolate: OpenFlow drop rule on any traffic entering from attacker port

Reroute is intentionally left out of the measurement table because this minimal
single-switch topology cannot represent a meaningful alternate path. The script
focuses on the actions needed for a small validation of relative cost patterns.
"""
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


def parse_ping(output):
    loss_match = re.search(r"(\d+(?:\.\d+)?)% packet loss", output)
    rtt_match = re.search(r"= ([\d\.]+)/([\d\.]+)/([\d\.]+)/([\d\.]+) ms", output)
    return {
        "packet_loss_pct": float(loss_match.group(1)) if loss_match else 100.0,
        "avg_rtt_ms": float(rtt_match.group(2)) if rtt_match else None,
        "raw": output.strip(),
    }


def run_ping(host, target_ip, count=5, timeout_s=8):
    output = host.cmd(f"timeout {timeout_s}s ping -c {count} -i 0.2 -W 1 {target_ip} || true")
    return parse_ping(output)


def run_iperf3(client, target_ip, duration=3, timeout_s=8):
    output = client.cmd(
        f"timeout {timeout_s}s iperf3 -J --connect-timeout 1500 -t {duration} -c {target_ip} || true"
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


def clear_tc(host, intf_name):
    host.cmd(f"tc qdisc del dev {intf_name} root >/dev/null 2>&1 || true")


def clear_mirror():
    os.system(
        "ovs-vsctl clear Bridge s1 mirrors >/dev/null 2>&1 || true"
    )


def clear_flows():
    os.system(
        "ovs-ofctl -O OpenFlow13 del-flows s1 >/dev/null 2>&1 || true"
    )


def reset_switch_learning():
    os.system(
        "ovs-appctl fdb/flush s1 >/dev/null 2>&1 || true"
    )


def apply_forward(_net):
    return


def apply_mirror(net):
    switch = net.get("s1")
    attacker = net.get("attacker")
    ids = net.get("ids")

    attacker_port = switch.ports[switch.connectionsTo(attacker)[0][0]]
    ids_port = switch.ports[switch.connectionsTo(ids)[0][0]]
    cmd = (
        "ovs-vsctl "
        f"-- --id=@src get Port s1-eth{attacker_port} "
        f"-- --id=@dst get Port s1-eth{ids_port} "
        "-- --id=@m create Mirror name=m0 select-src-port=@src select-dst-port=@src output-port=@dst "
        "-- set Bridge s1 mirrors=@m >/dev/null 2>&1"
    )
    os.system(cmd)


def apply_throttle(net, rate_mbit=5):
    attacker = net.get("attacker")
    intf_name = attacker.defaultIntf().name
    attacker.cmd(
        f"tc qdisc replace dev {intf_name} root tbf rate {rate_mbit}mbit burst 32kbit latency 80ms"
    )


def apply_drop(net):
    switch = net.get("s1")
    attacker = net.get("attacker")
    attacker_port = switch.ports[switch.connectionsTo(attacker)[0][0]]
    os.system(
        f"ovs-ofctl -O OpenFlow13 add-flow s1 \"priority=200,in_port={attacker_port},ip,nw_dst=10.0.0.4,actions=drop\""
    )


def apply_isolate(net):
    switch = net.get("s1")
    attacker = net.get("attacker")
    attacker_port = switch.ports[switch.connectionsTo(attacker)[0][0]]
    os.system(
        f"ovs-ofctl -O OpenFlow13 add-flow s1 \"priority=200,in_port={attacker_port},actions=drop\""
    )


ACTION_APPLIERS = {
    "Forward": apply_forward,
    "Mirror": apply_mirror,
    "Throttle": apply_throttle,
    "Drop": apply_drop,
    "Isolate": apply_isolate,
}


def measure_action(net, action_name):
    client = net.get("client")
    attacker = net.get("attacker")
    ids = net.get("ids")
    server = net.get("server")

    ACTION_APPLIERS[action_name](net)
    time.sleep(0.8)

    ids_intf = ids.defaultIntf().name
    ids_before = read_rx_bytes(ids, ids_intf)

    benign_ping = run_ping(client, "10.0.0.4")
    benign_iperf = run_iperf3(client, "10.0.0.4")

    attack_ping = run_ping(attacker, "10.0.0.4")
    attack_iperf = run_iperf3(attacker, "10.0.0.4")

    ids_after = read_rx_bytes(ids, ids_intf)

    result = {
        "action": action_name,
        "benign_ping_avg_rtt_ms": benign_ping["avg_rtt_ms"],
        "benign_ping_loss_pct": benign_ping["packet_loss_pct"],
        "benign_tcp_throughput_mbps": benign_iperf["throughput_mbps"],
        "attack_ping_avg_rtt_ms": attack_ping["avg_rtt_ms"],
        "attack_ping_loss_pct": attack_ping["packet_loss_pct"],
        "attack_tcp_throughput_mbps": attack_iperf["throughput_mbps"],
        "ids_mirror_rx_bytes_delta": int(max(ids_after - ids_before, 0)),
        "notes": "",
    }

    if action_name == "Mirror":
        result["notes"] = "Mirror should increase IDS RX bytes while preserving connectivity."
    elif action_name == "Throttle":
        result["notes"] = "Throttle should primarily reduce attacker throughput."
    elif action_name == "Drop":
        result["notes"] = "Drop should cause near-total attacker packet loss."
    elif action_name == "Isolate":
        result["notes"] = "Isolate approximated as full attacker ingress block."
    else:
        result["notes"] = "Forward baseline."

    return result


def start_iperf_server(server):
    server.cmd("pkill -f 'iperf3 -s' >/dev/null 2>&1 || true")
    server.cmd("iperf3 -s -D")
    time.sleep(1.0)


def stop_iperf_processes(net):
    for host_name in ("client", "attacker", "server"):
        net.get(host_name).cmd("pkill -f iperf3 >/dev/null 2>&1 || true")


def main():
    setLogLevel("warning")
    out_dir = os.path.join(
        os.path.dirname(__file__), "..", "..", "new_experiments", "mininet_validation"
    )
    out_dir = os.path.abspath(out_dir)
    os.makedirs(out_dir, exist_ok=True)

    rows = []
    for action_name in ("Forward", "Mirror", "Throttle", "Drop", "Isolate"):
        net = Mininet(
            topo=MinimalEdgeTopo(),
            switch=OVSSwitch,
            controller=None,
            autoSetMacs=True,
            autoStaticArp=True,
            cleanup=True,
            link=TCLink,
        )
        try:
            net.start()
            start_iperf_server(net.get("server"))
            print(f"Running action validation: {action_name}", flush=True)
            rows.append(measure_action(net, action_name))
        finally:
            with suppress(Exception):
                stop_iperf_processes(net)
            with suppress(Exception):
                clear_flows()
            with suppress(Exception):
                clear_mirror()
            with suppress(Exception):
                net.stop()
            with suppress(Exception):
                os.system("mn -c >/dev/null 2>&1 || true")

    summary_df = pd.DataFrame(rows)
    summary_csv = os.path.join(out_dir, "action_validation_summary.csv")
    summary_df.to_csv(summary_csv, index=False)

    md_lines = [
        "# Minimal Mininet Validation",
        "",
        "This table validates whether the simulator action-cost ordering has a plausible packet-level analogue.",
        "",
        summary_df.to_markdown(index=False),
        "",
        "Notes:",
        "- Mirror is validated through IDS RX-byte growth rather than IDS application processing.",
        "- Throttle is implemented with host-interface `tc tbf` on the attacker egress path.",
        "- Drop and Isolate are implemented with OVS OpenFlow drop rules.",
        "- Reroute is not measured in this single-switch topology because it would require a meaningful alternate path.",
    ]
    with open(os.path.join(out_dir, "summary.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(md_lines))

    print(summary_df.to_string(index=False))
    print(f"\nSaved to {summary_csv}")


if __name__ == "__main__":
    main()
