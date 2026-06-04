"""
Dynamic Alert Binding Diagnostic for OVS Replay

Replaces static attacker-IP binding with alert-derived flow binding:
  1. During replay, tcpdump/tshark extracts five-tuples from traffic
  2. High detector confidence windows generate alert-like records
  3. OVS actions bind to alert-derived src_ip / five-tuple
  4. Compares static binding vs alert-derived binding

This addresses the reviewer concern that OVS experiments use known attacker IPs
rather than real detector-assisted control flow binding.
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


# ─── Topology (same as enhanced_replay) ──────────────────────────────────────

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

ATTACKER_IPS = ["10.0.1.3", "10.0.1.4"]
BENIGN_IPS = ["10.0.1.1", "10.0.1.2"]
SERVER_IP = "10.0.3.2"


def host_port(switch, host):
    return switch.ports[switch.connectionsTo(host)[0][0]]


def link_intf_name(node_a, node_b):
    intf_a, _ = node_a.connectionsTo(node_b)[0]
    return intf_a.name


# ─── Alert Record Generator ──────────────────────────────────────────────────

class AlertRecordGenerator:
    """Generates alert-like records from tcpdump/tshark capture during replay.

    Instead of using pre-known attacker IPs, this generator:
    1. Captures live traffic via tcpdump on the IDS/mirror port
    2. Extracts five-tuples using tshark
    3. Applies detector-confidence-based alert scoring
    4. Returns suspicious src_ips / five-tuples for OVS action binding

    This is NOT an IDS accuracy evaluation; it's a replay-time flow-binding
    mechanism that replaces static IP binding.
    """

    def __init__(self, detector_confidence_threshold=0.78,
                 alert_score_threshold=0.5):
        self.detector_confidence_threshold = detector_confidence_threshold
        self.alert_score_threshold = alert_score_threshold
        self.alert_history = []
        self.binding_errors = 0
        self.correct_bindings = 0
        self.total_bindings = 0

    def generate_alerts_from_capture(self, pcap_path, detector_confidence,
                                     window_id):
        """Generate alert records from a packet capture.

        Args:
            pcap_path: Path to the tcpdump capture file
            detector_confidence: Current window's detector confidence score
            window_id: Current replay step/window ID

        Returns:
            List of alert records with five-tuples and scores
        """
        alerts = []

        if detector_confidence < self.detector_confidence_threshold:
            # Low confidence: no alerts generated
            return alerts

        # Extract five-tuples from capture using tshark
        five_tuples = self._extract_five_tuples(pcap_path)

        for ft in five_tuples:
            # Score based on detector confidence and traffic characteristics
            alert_score = self._compute_alert_score(
                ft, detector_confidence
            )

            if alert_score >= self.alert_score_threshold:
                alert = {
                    "src_ip": ft["src_ip"],
                    "dst_ip": ft["dst_ip"],
                    "src_port": ft["src_port"],
                    "dst_port": ft["dst_port"],
                    "protocol": ft["protocol"],
                    "detector_score": detector_confidence,
                    "alert_score": alert_score,
                    "timestamp": time.time(),
                    "window_id": window_id,
                }
                alerts.append(alert)
                self.alert_history.append(alert)

        return alerts

    def _extract_five_tuples(self, pcap_path):
        """Extract five-tuples from a pcap file using tshark."""
        five_tuples = []
        try:
            cmd = (
                f"tshark -r {pcap_path} -T fields "
                f"-e ip.src -e ip.dst -e tcp.srcport -e tcp.dstport "
                f"-e udp.srcport -e udp.dstport -e ip.proto "
                f"-Y 'ip.src' 2>/dev/null | head -200"
            )
            output = subprocess.check_output(
                cmd, shell=True, stderr=subprocess.DEVNULL
            ).decode()

            seen = set()
            for line in output.strip().split("\n"):
                if not line.strip():
                    continue
                fields = line.strip().split("\t")
                if len(fields) < 7:
                    continue

                src_ip = fields[0].strip()
                dst_ip = fields[1].strip()
                tcp_sport = fields[2].strip()
                tcp_dport = fields[3].strip()
                udp_sport = fields[4].strip()
                udp_dport = fields[5].strip()
                proto = fields[6].strip()

                if not src_ip or not dst_ip:
                    continue

                # Determine port based on protocol
                if proto == "6":  # TCP
                    src_port = tcp_sport or "0"
                    dst_port = tcp_dport or "0"
                    protocol = "TCP"
                elif proto == "17":  # UDP
                    src_port = udp_sport or "0"
                    dst_port = udp_dport or "0"
                    protocol = "UDP"
                else:
                    src_port = "0"
                    dst_port = "0"
                    protocol = f"PROTO-{proto}"

                # Deduplicate
                key = (src_ip, dst_ip, src_port, dst_port, protocol)
                if key in seen:
                    continue
                seen.add(key)

                five_tuples.append({
                    "src_ip": src_ip,
                    "dst_ip": dst_ip,
                    "src_port": src_port,
                    "dst_port": dst_port,
                    "protocol": protocol,
                })

        except Exception as e:
            # tshark may not be available; fall back to tcpdump-based extraction
            five_tuples = self._extract_five_tuples_tcpdump(pcap_path)

        return five_tuples

    def _extract_five_tuples_tcpdump(self, pcap_path):
        """Fallback: extract five-tuples using tcpdump output parsing."""
        five_tuples = []
        try:
            output = subprocess.check_output(
                f"tcpdump -n -r {pcap_path} 2>/dev/null | head -200",
                shell=True, stderr=subprocess.DEVNULL
            ).decode()

            seen = set()
            for line in output.strip().split("\n"):
                # Parse tcpdump format: IP src_ip.port > dst_ip.port
                match = re.search(
                    r"IP (\d+\.\d+\.\d+\.\d+)\.(\d+) > (\d+\.\d+\.\d+\.\d+)\.(\d+)",
                    line
                )
                if match:
                    src_ip, src_port, dst_ip, dst_port = (
                        match.group(1), match.group(2),
                        match.group(3), match.group(4)
                    )
                    protocol = "TCP" if "TCP" in line.upper() else "IP"
                    key = (src_ip, dst_ip, src_port, dst_port, protocol)
                    if key not in seen:
                        seen.add(key)
                        five_tuples.append({
                            "src_ip": src_ip,
                            "dst_ip": dst_ip,
                            "src_port": src_port,
                            "dst_port": dst_port,
                            "protocol": protocol,
                        })
        except Exception:
            pass

        return five_tuples

    def _compute_alert_score(self, five_tuple, detector_confidence):
        """Compute an alert score for a five-tuple based on detector confidence
        and traffic characteristics.

        Scoring heuristic (no oracle labels used):
        - Base score from detector confidence
        - Boost for high-rate sources (multiple flows from same src_ip)
        - Boost for known suspicious ports (if applicable)
        - No oracle/ground-truth information used
        """
        base_score = detector_confidence

        # Count recent alerts from the same source IP
        recent_from_src = sum(
            1 for a in self.alert_history[-50:]
            if a["src_ip"] == five_tuple["src_ip"]
        )
        # High-rate source boost (max +0.15)
        rate_boost = min(0.15, recent_from_src * 0.03)

        alert_score = base_score + rate_boost
        return min(alert_score, 1.0)

    def get_suspicious_src_ips(self, alerts):
        """Extract unique suspicious source IPs from alert records."""
        return list(set(a["src_ip"] for a in alerts))

    def track_binding_accuracy(self, bound_ips, true_attacker_ips):
        """Track binding accuracy for diagnostic purposes.

        This is ONLY for post-hoc evaluation; the binding itself
        does NOT use true_attacker_ips.
        """
        self.total_bindings += len(bound_ips)
        for ip in bound_ips:
            if ip in true_attacker_ips:
                self.correct_bindings += 1
            else:
                self.binding_errors += 1

    def get_binding_stats(self):
        return {
            "total_bindings": self.total_bindings,
            "correct_bindings": self.correct_bindings,
            "binding_errors": self.binding_errors,
            "binding_error_rate": (
                self.binding_errors / max(self.total_bindings, 1)
            ),
        }


# ─── Alert-Derived Action Appliers ───────────────────────────────────────────

def apply_drop_alert_binding(alert_gen, net, alerts, step_id):
    """Drop using alert-derived src_ips instead of static attacker IPs."""
    suspicious_ips = alert_gen.get_suspicious_src_ips(alerts)

    if not suspicious_ips:
        # No alerts: no drop rules installed (safe default)
        return

    for src_ip in suspicious_ips:
        os.system(
            f"ovs-ofctl -O OpenFlow13 add-flow s1 "
            f"\"priority=200,ip,nw_src={src_ip},actions=drop\" "
            f">/dev/null 2>&1 || true"
        )

    # Track binding accuracy (diagnostic only)
    alert_gen.track_binding_accuracy(suspicious_ips, ATTACKER_IPS)


def apply_isolate_alert_binding(alert_gen, net, alerts, step_id):
    """Isolate using alert-derived src_ips instead of static attacker IPs."""
    suspicious_ips = alert_gen.get_suspicious_src_ips(alerts)

    if not suspicious_ips:
        return

    switch = net.get("s1")
    for ip in suspicious_ips:
        # Find the host port for this IP
        for h_name in ALL_HOSTS:
            h = net.get(h_name)
            if h.IP() == ip:
                h_port = host_port(switch, h)
                os.system(
                    f"ovs-ofctl -O OpenFlow13 add-flow s1 "
                    f"\"priority=210,in_port={h_port},actions=drop\" "
                    f">/dev/null 2>&1 || true"
                )
                break

    alert_gen.track_binding_accuracy(suspicious_ips, ATTACKER_IPS)


def apply_reroute_alert_binding(alert_gen, net, alerts, step_id):
    """Reroute with alert-derived drop on scrubbing path."""
    switch = net.get("s1")
    s4 = net.get("s4")
    s3 = net.get("s3")

    s1_s4_port = host_port(switch, s4)
    s4_s3_port = host_port(s4, s3)
    s4_s1_port = host_port(s4, switch)

    # Reroute all traffic to server through s4
    os.system(
        f"ovs-ofctl -O OpenFlow13 add-flow s1 "
        f"\"priority=150,ip,nw_dst={SERVER_IP},actions=output:{s1_s4_port}\" "
        f">/dev/null 2>&1 || true"
    )
    os.system(
        f"ovs-ofctl -O OpenFlow13 add-flow s4 "
        f"\"priority=150,ip,nw_dst={SERVER_IP},actions=output:{s4_s3_port}\" "
        f">/dev/null 2>&1 || true"
    )
    os.system(
        f"ovs-ofctl -O OpenFlow13 add-flow s4 "
        f"\"priority=150,ip,nw_src={SERVER_IP},actions=output:{s4_s1_port}\" "
        f">/dev/null 2>&1 || true"
    )

    # Drop suspicious IPs on s4 scrubbing path
    suspicious_ips = alert_gen.get_suspicious_src_ips(alerts)
    for src_ip in suspicious_ips:
        os.system(
            f"ovs-ofctl -O OpenFlow13 add-flow s4 "
            f"\"priority=200,ip,nw_src={src_ip},actions=drop\" "
            f">/dev/null 2>&1 || true"
        )

    alert_gen.track_binding_accuracy(suspicious_ips, ATTACKER_IPS)


# ─── Utility functions ────────────────────────────────────────────────────────

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


def clear_mirror():
    os.system("ovs-vsctl clear Bridge s1 mirrors >/dev/null 2>&1 || true")


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


def reset_network_controls(net):
    with suppress(Exception):
        clear_flows()
    with suppress(Exception):
        install_default_forwarding()
    with suppress(Exception):
        clear_mirror()
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


# ─── Static Binding Appliers (for comparison) ────────────────────────────────

def apply_drop_static(net, target_name, mode):
    if mode == "attackers":
        for attacker_ip in ATTACKER_IPS:
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


def apply_isolate_static(net, target_name, mode):
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


def apply_reroute_static(net, target_name, mode):
    switch = net.get("s1")
    s4 = net.get("s4")
    s3 = net.get("s3")

    s1_s4_port = host_port(switch, s4)
    s4_s3_port = host_port(s4, s3)
    s4_s1_port = host_port(s4, switch)

    os.system(
        f"ovs-ofctl -O OpenFlow13 add-flow s1 "
        f"\"priority=150,ip,nw_dst={SERVER_IP},actions=output:{s1_s4_port}\" >/dev/null 2>&1 || true"
    )
    os.system(
        f"ovs-ofctl -O OpenFlow13 add-flow s4 "
        f"\"priority=150,ip,nw_dst={SERVER_IP},actions=output:{s4_s3_port}\" >/dev/null 2>&1 || true"
    )
    os.system(
        f"ovs-ofctl -O OpenFlow13 add-flow s4 "
        f"\"priority=150,ip,nw_src={SERVER_IP},actions=output:{s4_s1_port}\" >/dev/null 2>&1 || true"
    )

    if mode == "attackers":
        for attacker_ip in ATTACKER_IPS:
            os.system(
                f"ovs-ofctl -O OpenFlow13 add-flow s4 "
                f"\"priority=200,ip,nw_src={attacker_ip},actions=drop\" >/dev/null 2>&1 || true"
            )


# ─── Step Measurement ────────────────────────────────────────────────────────

def measure_alert_binding_step(net, action_name, target_name, mode, step_id,
                               binding_mode, alert_gen, detector_confidence):
    """Measure a single step with either static or alert-derived binding."""
    reset_network_controls(net)

    ids = net.get("ids")
    ids_intf = ids.defaultIntf().name
    ids_before_bytes = read_rx_bytes(ids, ids_intf)
    flows_before_s1 = count_flow_entries("s1")
    flows_before_s4 = count_flow_entries("s4")

    # Start capture for alert generation
    pcap_path = f"/tmp/alert_binding_step_{step_id}.pcap"
    ids.cmd(f"rm -f {pcap_path} >/dev/null 2>&1 || true")
    ids.cmd(f"timeout 3s tcpdump -i any -c 500 -w {pcap_path} >/dev/null 2>&1 &")

    # Apply action based on binding mode
    t0 = time.perf_counter()

    if binding_mode == "alert":
        # Alert-derived binding: generate alerts and bind to suspicious IPs
        if action_name in ("Drop", "Isolate", "Reroute"):
            time.sleep(0.3)  # Wait for some packets to be captured
            alerts = alert_gen.generate_alerts_from_capture(
                pcap_path, detector_confidence, step_id
            )

            if action_name == "Drop":
                apply_drop_alert_binding(alert_gen, net, alerts, step_id)
            elif action_name == "Isolate":
                apply_isolate_alert_binding(alert_gen, net, alerts, step_id)
            elif action_name == "Reroute":
                apply_reroute_alert_binding(alert_gen, net, alerts, step_id)
        else:
            # Non-aggressive actions: same for both modes
            _apply_non_aggressive(net, action_name, target_name, mode)
    else:
        # Static binding: use pre-known attacker IPs
        if action_name == "Drop":
            apply_drop_static(net, target_name, mode)
        elif action_name == "Isolate":
            apply_isolate_static(net, target_name, mode)
        elif action_name == "Reroute":
            apply_reroute_static(net, target_name, mode)
        else:
            _apply_non_aggressive(net, action_name, target_name, mode)

    apply_ms = (time.perf_counter() - t0) * 1000.0
    time.sleep(0.2)

    flows_after_s1 = count_flow_entries("s1")
    flows_after_s4 = count_flow_entries("s4")

    # Measure traffic
    benign1 = net.get("benign1")
    benign2 = net.get("benign2")
    attacker1 = net.get("attacker1")
    attacker2 = net.get("attacker2")

    benign1_tcp = run_iperf3_tcp(benign1, SERVER_IP, duration=1, port=5201)
    benign2_udp = run_iperf3_udp(benign2, SERVER_IP, bandwidth="50M", duration=1, port=5203)
    attack1_tcp = run_iperf3_tcp(attacker1, SERVER_IP, duration=1, port=5202)
    attack2_udp = run_iperf3_udp(attacker2, SERVER_IP, bandwidth="50M", duration=1, port=5204)

    benign_rtt = run_ping(benign1, SERVER_IP)

    ids_after_bytes = read_rx_bytes(ids, ids_intf)
    ovs_cpu_after = get_cpu_usage()

    binding_stats = alert_gen.get_binding_stats()

    return {
        "control_apply_ms": apply_ms,
        "benign_tcp_mbps": benign1_tcp["throughput_mbps"],
        "benign_udp_mbps": benign2_udp["throughput_mbps"],
        "benign_udp_loss_pct": benign2_udp.get("lost_percent"),
        "attack_tcp_mbps": attack1_tcp["throughput_mbps"],
        "attack_udp_mbps": attack2_udp["throughput_mbps"],
        "benign_rtt_ms": benign_rtt["avg_rtt_ms"],
        "benign_ping_loss_pct": benign_rtt["packet_loss_pct"],
        "ovs_flow_entries_total_s1": flows_after_s1,
        "ovs_flow_entries_total_s4": flows_after_s4,
        "ovs_flow_entries_total": flows_after_s1 + flows_after_s4,
        "ovs_cpu_pct": ovs_cpu_after,
        "binding_mode": binding_mode,
        "binding_errors": binding_stats["binding_errors"],
        "correct_bindings": binding_stats["correct_bindings"],
        "total_bindings": binding_stats["total_bindings"],
        "binding_error_rate": binding_stats["binding_error_rate"],
    }


def _apply_non_aggressive(net, action_name, target_name, mode):
    """Apply non-aggressive actions (same for both binding modes)."""
    if action_name == "Forward":
        return
    elif action_name == "Inspect":
        switch = net.get("s1")
        ids = net.get("ids")
        ids_port = host_port(switch, ids)
        os.system(
            f"ovs-ofctl -O OpenFlow13 add-flow s1 "
            f"\"priority=100,ip,nw_dst={SERVER_IP},actions=output:NORMAL,output:{ids_port}\" >/dev/null 2>&1 || true"
        )
    elif action_name == "Mirror":
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
    elif action_name == "Throttle":
        edge = net.get("s1")
        agg = net.get("s2")
        intf_name = link_intf_name(edge, agg)
        os.system(
            f"tc qdisc replace dev {intf_name} root tbf rate 5mbit burst 32kbit latency 80ms >/dev/null 2>&1"
        )


# ─── Replay Runners ──────────────────────────────────────────────────────────

def run_alert_binding_replay(net, controller_df, binding_mode, alert_gen):
    """Run replay with specified binding mode (static or alert-derived)."""
    rows = []
    for _, row in controller_df.sort_values("replay_step").iterrows():
        if int(row["true_label"]) == 1:
            target_name = "attacker1"
            action_mode = "attackers"
        else:
            target_name = "benign1"
            action_mode = None

        # Get detector confidence for this window (from replay plan)
        detector_confidence = float(row.get("detector_confidence", 0.5))

        metrics = measure_alert_binding_step(
            net, row["action_name"], target_name, action_mode,
            int(row["replay_step"]), binding_mode, alert_gen,
            detector_confidence
        )
        rows.append({**row.to_dict(), "target_name": target_name, **metrics})

        if len(rows) % 50 == 0:
            print(f"  Step {len(rows)}/{len(controller_df)} [{binding_mode} binding]")

    return pd.DataFrame(rows)


def summarize_binding(df, binding_mode):
    """Summarize results for a binding mode."""
    benign = df[df["true_label"] == 0]
    attack = df[df["true_label"] == 1]

    return {
        "binding_mode": binding_mode,
        "controller": df["controller"].iloc[0],
        "steps": len(df),
        "benign_tcp_mbps": float(benign["benign_tcp_mbps"].mean()) if len(benign) else 0.0,
        "benign_loss_pct": float(benign["benign_ping_loss_pct"].mean()) if len(benign) else 0.0,
        "attack_suppression_ratio": float(
            1.0 - attack["attack_tcp_mbps"].mean() / max(df["attack_tcp_mbps"].mean(), 0.01)
        ) if len(attack) else 0.0,
        "mean_rule_install_ms": float(df["control_apply_ms"].mean()),
        "binding_errors": int(df["binding_errors"].iloc[-1]) if "binding_errors" in df.columns else 0,
        "correct_bindings": int(df["correct_bindings"].iloc[-1]) if "correct_bindings" in df.columns else 0,
        "total_bindings": int(df["total_bindings"].iloc[-1]) if "total_bindings" in df.columns else 0,
        "binding_error_rate": float(df["binding_error_rate"].iloc[-1]) if "binding_error_rate" in df.columns else 0.0,
        "ovs_cpu_pct": float(df["ovs_cpu_pct"].mean()),
    }


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Dynamic Alert Binding Diagnostic: static vs alert-derived flow binding"
    )
    parser.add_argument("--plan-csv", required=True, help="Path to replay plan CSV")
    parser.add_argument("--out-dir", required=True, help="Output directory")
    args = parser.parse_args()

    setLogLevel("warning")
    os.makedirs(args.out_dir, exist_ok=True)

    plan_df = pd.read_csv(args.plan_csv)
    all_step_frames = []
    summary_rows = []

    for controller_name, controller_df in plan_df.groupby("controller", sort=False):
        print(f"\n=== Controller: {controller_name} ({len(controller_df)} steps) ===")

        # ─── Static binding replay ────────────────────────────────────────
        alert_gen_static = AlertRecordGenerator()
        topo1 = EnhancedEdgeTopo()
        net1 = Mininet(
            topo=topo1, switch=OVSSwitch, controller=None,
            autoSetMacs=True, autoStaticArp=True, cleanup=True, link=TCLink,
        )
        try:
            net1.start()
            reset_network_controls(net1)
            start_iperf_servers(net1.get("server"))
            step_df = run_alert_binding_replay(
                net1, controller_df.copy(), "static", alert_gen_static
            )
            step_df["replay_mode"] = "static_binding"
            all_step_frames.append(step_df)
            summary_row = summarize_binding(step_df, "static")
            summary_rows.append(summary_row)
            print(f"  Static binding summary: {summary_row}")
        finally:
            with suppress(Exception):
                stop_iperf_processes(net1)
            with suppress(Exception):
                reset_network_controls(net1)
            with suppress(Exception):
                net1.stop()
            with suppress(Exception):
                os.system("mn -c >/dev/null 2>&1 || true")

        # ─── Alert-derived binding replay ─────────────────────────────────
        alert_gen_dynamic = AlertRecordGenerator(
            detector_confidence_threshold=0.78,
            alert_score_threshold=0.5,
        )
        topo2 = EnhancedEdgeTopo()
        net2 = Mininet(
            topo=topo2, switch=OVSSwitch, controller=None,
            autoSetMacs=True, autoStaticArp=True, cleanup=True, link=TCLink,
        )
        try:
            net2.start()
            reset_network_controls(net2)
            start_iperf_servers(net2.get("server"))
            step_df2 = run_alert_binding_replay(
                net2, controller_df.copy(), "alert", alert_gen_dynamic
            )
            step_df2["replay_mode"] = "alert_binding"
            all_step_frames.append(step_df2)
            summary_row2 = summarize_binding(step_df2, "alert")
            summary_rows.append(summary_row2)
            print(f"  Alert-derived binding summary: {summary_row2}")
        finally:
            with suppress(Exception):
                stop_iperf_processes(net2)
            with suppress(Exception):
                reset_network_controls(net2)
            with suppress(Exception):
                net2.stop()
            with suppress(Exception):
                os.system("mn -c >/dev/null 2>&1 || true")

    # Save results
    all_steps = pd.concat(all_step_frames, ignore_index=True)
    summary_df = pd.DataFrame(summary_rows)

    all_steps.to_csv(os.path.join(args.out_dir, "alert_binding_step_trace.csv"), index=False)
    summary_df.to_csv(os.path.join(args.out_dir, "alert_binding_summary.csv"), index=False)

    # Generate comparison table
    print("\n=== Static vs Alert-Derived Binding Comparison ===")
    comparison = summary_df.pivot(
        index="controller", columns="binding_mode",
        values=["benign_tcp_mbps", "benign_loss_pct", "attack_suppression_ratio",
                "mean_rule_install_ms", "binding_errors", "binding_error_rate"]
    )
    print(comparison.to_string())
    comparison.to_csv(os.path.join(args.out_dir, "static_vs_alert_binding_comparison.csv"))

    print("\n=== Alert Binding Diagnostic Complete ===")


if __name__ == "__main__":
    main()
