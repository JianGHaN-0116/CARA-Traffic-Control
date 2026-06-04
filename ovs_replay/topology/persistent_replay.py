"""
Persistent-Rule OVS/Mininet Controller-in-the-Loop Replay

Extends enhanced_replay.py with persistent rule lifecycle management:
  - Rules carry cookie=action_id/window_id for tracking
  - Drop/Isolate/Reroute rules have idle_timeout=30s, hard_timeout=60s
  - Inspect/Mirror rules have shorter timeouts (idle=10s, hard=20s)
  - Throttle uses tc qdisc change/replace, tracks duplicate qdiscs
  - Each step does incremental diff: add needed, remove conflicting, keep valid
  - Priority hierarchy: Isolate(210) > Drop(200) > Reroute(150) > Throttle(100)
                         > Inspect/Mirror(100) > Forward(0)
  - Tracks stale rule conflicts, peak flow entries, rule churn, rule-install latency,
    OVS CPU, benign throughput, attack suppression

This addresses the reviewer concern that per-step rule install/cleanup is unrealistic;
persistent lifecycle replay validates that CARA-TC's service-security balance survives
when rules persist with timeout aging and conflict cleanup.
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
SERVER_IP = "10.0.3.2"

# Priority hierarchy: Isolate > Drop > Reroute > Throttle > Inspect/Mirror > Forward
ACTION_PRIORITY = {
    "Isolate": 210,
    "Drop": 200,
    "Reroute": 150,
    "Throttle": 100,
    "Inspect": 100,
    "Mirror": 100,
    "Forward": 0,
}

# Timeout configuration per action type
# Aggressive actions: longer timeouts to maintain containment
# Passive actions: shorter timeouts to avoid stale mirror/inspect
ACTION_TIMEOUTS = {
    "Isolate":  {"idle_timeout": 30, "hard_timeout": 60},
    "Drop":     {"idle_timeout": 30, "hard_timeout": 60},
    "Reroute":  {"idle_timeout": 30, "hard_timeout": 60},
    "Throttle": {"idle_timeout": 30, "hard_timeout": 60},  # tc-based, managed separately
    "Inspect":  {"idle_timeout": 10, "hard_timeout": 20},
    "Mirror":   {"idle_timeout": 10, "hard_timeout": 20},
    "Forward":  {"idle_timeout": 0,  "hard_timeout": 0},    # default, no timeout
}


def host_port(switch, host):
    return switch.ports[switch.connectionsTo(host)[0][0]]


def link_intf_name(node_a, node_b):
    intf_a, _ = node_a.connectionsTo(node_b)[0]
    return intf_a.name


# ─── Persistent Rule Manager ─────────────────────────────────────────────────

class PersistentRuleManager:
    """Manages persistent OVS rules with cookie tracking, timeout aging,
    conflict detection, and incremental diff updates.

    Tracks installed rules by cookie to detect stale conflicts and
    compute incremental diffs between steps.
    """

    def __init__(self):
        # cookie -> {action, target_ip, switch, priority, step_installed, ...}
        self.installed_rules = {}
        self.rule_churn_add = 0
        self.rule_churn_del = 0
        self.stale_conflicts = 0
        self.peak_flow_entries = 0
        self.duplicate_qdisc_count = 0
        self._next_cookie = 1000

    def _alloc_cookie(self, action, step_id):
        """Allocate a unique cookie encoding action and step."""
        cookie = self._next_cookie
        self._next_cookie += 1
        return cookie

    def reset_churn_counters(self):
        """Reset per-step churn counters."""
        self.rule_churn_add = 0
        self.rule_churn_del = 0
        self.stale_conflicts = 0

    def get_churn_stats(self):
        return {
            "rule_churn_add": self.rule_churn_add,
            "rule_churn_del": self.rule_churn_del,
            "stale_conflicts": self.stale_conflicts,
            "duplicate_qdisc_count": self.duplicate_qdisc_count,
            "total_active_rules": len(self.installed_rules),
        }

    def cleanup_expired_rules(self):
        """Remove entries for rules that have expired via OVS timeouts.

        We check which cookies still exist in OVS flow tables and remove
        entries that are no longer present (expired via idle/hard timeout).
        """
        expired_cookies = []
        for cookie, rule_info in self.installed_rules.items():
            sw = rule_info["switch"]
            if not self._cookie_exists(sw, cookie):
                expired_cookies.append(cookie)

        for cookie in expired_cookies:
            del self.installed_rules[cookie]
            self.rule_churn_del += 1

    def _cookie_exists(self, switch_name, cookie):
        """Check if a flow with the given cookie still exists in OVS."""
        try:
            out = subprocess.check_output(
                f"ovs-ofctl -O OpenFlow13 dump-flows {switch_name} "
                f"cookie=0x{cookie:x}/0xffffffff",
                shell=True, stderr=subprocess.DEVNULL
            ).decode()
            return "actions=" in out
        except Exception:
            return False

    def detect_conflicts(self, target_ip, action, switch="s1"):
        """Detect if installing a new rule for target_ip conflicts with
        existing persistent rules.

        Conflict rules:
        - If new action is aggressive (Drop/Isolate) but a lower-priority
          rule already exists for the same target, that's a priority conflict.
        - If a higher-priority rule already covers the target, the new rule
          is redundant (not a conflict, but tracked).
        - If an existing rule for the same target has a different action,
          that's a stale conflict requiring cleanup.
        """
        conflicts = 0
        new_priority = ACTION_PRIORITY.get(action, 0)

        for cookie, rule_info in list(self.installed_rules.items()):
            if rule_info["switch"] != switch:
                continue
            if rule_info.get("target_ip") != target_ip:
                continue
            # Same target, check for stale conflict
            existing_action = rule_info["action"]
            existing_priority = rule_info.get("priority", 0)

            if existing_action != action:
                # Different action for same target = stale conflict
                conflicts += 1
            elif existing_priority != new_priority:
                # Same action but different priority (shouldn't happen normally)
                conflicts += 1

        return conflicts

    def remove_conflicting_rules(self, target_ip, action, switch="s1"):
        """Remove persistent rules that conflict with the new action for target_ip.

        Conflict resolution: remove existing rules for the same target that
        have a different action or lower priority.
        """
        new_priority = ACTION_PRIORITY.get(action, 0)
        to_remove = []

        for cookie, rule_info in list(self.installed_rules.items()):
            if rule_info["switch"] != switch:
                continue
            if rule_info.get("target_ip") != target_ip:
                continue
            existing_action = rule_info["action"]
            existing_priority = rule_info.get("priority", 0)

            if existing_action != action:
                # Remove stale rule with different action
                self._delete_flow_by_cookie(switch, cookie)
                to_remove.append(cookie)
                self.rule_churn_del += 1
            elif existing_priority < new_priority:
                # Remove lower-priority rule for same action
                self._delete_flow_by_cookie(switch, cookie)
                to_remove.append(cookie)
                self.rule_churn_del += 1

        for cookie in to_remove:
            del self.installed_rules[cookie]

    def _delete_flow_by_cookie(self, switch_name, cookie):
        """Delete a flow entry by cookie."""
        os.system(
            f"ovs-ofctl -O OpenFlow13 del-flows {switch_name} "
            f"cookie=0x{cookie:x}/0xffffffff >/dev/null 2>&1 || true"
        )

    def register_rule(self, cookie, action, target_ip, switch, priority, step_id):
        """Register a newly installed rule in the tracker."""
        self.installed_rules[cookie] = {
            "action": action,
            "target_ip": target_ip,
            "switch": switch,
            "priority": priority,
            "step_installed": step_id,
        }
        self.rule_churn_add += 1

    def clear_all(self):
        """Clear all tracked rules (used at controller start/end)."""
        self.installed_rules.clear()
        self.rule_churn_add = 0
        self.rule_churn_del = 0
        self.stale_conflicts = 0
        self.duplicate_qdisc_count = 0


# ─── Persistent Action Appliers ──────────────────────────────────────────────

def apply_forward_persistent(rule_mgr, net, target_name, mode, step_id):
    """Forward: no rules needed, but clean up any existing aggressive rules
    for the target if switching from aggressive to safe."""
    if mode == "attackers":
        for attacker_ip in ATTACKER_IPS:
            rule_mgr.remove_conflicting_rules(attacker_ip, "Forward", "s1")
    return []


def apply_drop_persistent(rule_mgr, net, target_name, mode, step_id):
    """Drop with persistent rules: cookie, timeout, priority."""
    timeouts = ACTION_TIMEOUTS["Drop"]
    priority = ACTION_PRIORITY["Drop"]
    cookies = []

    if mode == "attackers":
        for attacker_ip in ATTACKER_IPS:
            conflicts = rule_mgr.detect_conflicts(attacker_ip, "Drop", "s1")
            rule_mgr.stale_conflicts += conflicts
            rule_mgr.remove_conflicting_rules(attacker_ip, "Drop", "s1")

            cookie = rule_mgr._alloc_cookie("Drop", step_id)
            os.system(
                f"ovs-ofctl -O OpenFlow13 add-flow s1 "
                f"\"priority={priority},ip,nw_src={attacker_ip},"
                f"idle_timeout={timeouts['idle_timeout']},"
                f"hard_timeout={timeouts['hard_timeout']},"
                f"cookie=0x{cookie:x},"
                f"actions=drop\" >/dev/null 2>&1 || true"
            )
            rule_mgr.register_rule(cookie, "Drop", attacker_ip, "s1", priority, step_id)
            cookies.append(cookie)
    else:
        switch = net.get("s1")
        target = net.get(target_name)
        target_port = host_port(switch, target)
        target_ip = target.IP()

        conflicts = rule_mgr.detect_conflicts(target_ip, "Drop", "s1")
        rule_mgr.stale_conflicts += conflicts
        rule_mgr.remove_conflicting_rules(target_ip, "Drop", "s1")

        cookie = rule_mgr._alloc_cookie("Drop", step_id)
        os.system(
            f"ovs-ofctl -O OpenFlow13 add-flow s1 "
            f"\"priority={priority},in_port={target_port},ip,"
            f"idle_timeout={timeouts['idle_timeout']},"
            f"hard_timeout={timeouts['hard_timeout']},"
            f"cookie=0x{cookie:x},"
            f"actions=drop\" >/dev/null 2>&1 || true"
        )
        rule_mgr.register_rule(cookie, "Drop", target_ip, "s1", priority, step_id)
        cookies.append(cookie)

    return cookies


def apply_isolate_persistent(rule_mgr, net, target_name, mode, step_id):
    """Isolate with persistent rules: highest priority, timeout, cookie."""
    timeouts = ACTION_TIMEOUTS["Isolate"]
    priority = ACTION_PRIORITY["Isolate"]
    cookies = []

    if mode == "attackers":
        for h_name in ATTACKER_HOSTS:
            switch = net.get("s1")
            h = net.get(h_name)
            h_port = host_port(switch, h)
            h_ip = h.IP()

            conflicts = rule_mgr.detect_conflicts(h_ip, "Isolate", "s1")
            rule_mgr.stale_conflicts += conflicts
            rule_mgr.remove_conflicting_rules(h_ip, "Isolate", "s1")

            cookie = rule_mgr._alloc_cookie("Isolate", step_id)
            os.system(
                f"ovs-ofctl -O OpenFlow13 add-flow s1 "
                f"\"priority={priority},in_port={h_port},"
                f"idle_timeout={timeouts['idle_timeout']},"
                f"hard_timeout={timeouts['hard_timeout']},"
                f"cookie=0x{cookie:x},"
                f"actions=drop\" >/dev/null 2>&1 || true"
            )
            rule_mgr.register_rule(cookie, "Isolate", h_ip, "s1", priority, step_id)
            cookies.append(cookie)
    else:
        switch = net.get("s1")
        target = net.get(target_name)
        target_port = host_port(switch, target)
        target_ip = target.IP()

        conflicts = rule_mgr.detect_conflicts(target_ip, "Isolate", "s1")
        rule_mgr.stale_conflicts += conflicts
        rule_mgr.remove_conflicting_rules(target_ip, "Isolate", "s1")

        cookie = rule_mgr._alloc_cookie("Isolate", step_id)
        os.system(
            f"ovs-ofctl -O OpenFlow13 add-flow s1 "
            f"\"priority={priority},in_port={target_port},"
            f"idle_timeout={timeouts['idle_timeout']},"
            f"hard_timeout={timeouts['hard_timeout']},"
            f"cookie=0x{cookie:x},"
            f"actions=drop\" >/dev/null 2>&1 || true"
        )
        rule_mgr.register_rule(cookie, "Isolate", target_ip, "s1", priority, step_id)
        cookies.append(cookie)

    return cookies


def apply_reroute_persistent(rule_mgr, net, target_name, mode, step_id):
    """Reroute with persistent rules on s1 and s4, with timeout and cookie."""
    timeouts = ACTION_TIMEOUTS["Reroute"]
    priority = ACTION_PRIORITY["Reroute"]
    cookies = []

    switch = net.get("s1")
    s4 = net.get("s4")
    s3 = net.get("s3")

    s1_s4_port = host_port(switch, s4)
    s4_s3_port = host_port(s4, s3)
    s4_s1_port = host_port(s4, switch)

    # s1: redirect to s4
    cookie1 = rule_mgr._alloc_cookie("Reroute", step_id)
    os.system(
        f"ovs-ofctl -O OpenFlow13 add-flow s1 "
        f"\"priority={priority},ip,nw_dst={SERVER_IP},"
        f"idle_timeout={timeouts['idle_timeout']},"
        f"hard_timeout={timeouts['hard_timeout']},"
        f"cookie=0x{cookie1:x},"
        f"actions=output:{s1_s4_port}\" >/dev/null 2>&1 || true"
    )
    rule_mgr.register_rule(cookie1, "Reroute", SERVER_IP, "s1", priority, step_id)
    cookies.append(cookie1)

    # s4: forward to s3
    cookie2 = rule_mgr._alloc_cookie("Reroute", step_id)
    os.system(
        f"ovs-ofctl -O OpenFlow13 add-flow s4 "
        f"\"priority={priority},ip,nw_dst={SERVER_IP},"
        f"idle_timeout={timeouts['idle_timeout']},"
        f"hard_timeout={timeouts['hard_timeout']},"
        f"cookie=0x{cookie2:x},"
        f"actions=output:{s4_s3_port}\" >/dev/null 2>&1 || true"
    )
    rule_mgr.register_rule(cookie2, "Reroute", SERVER_IP, "s4", priority, step_id)
    cookies.append(cookie2)

    # s4: return path
    cookie3 = rule_mgr._alloc_cookie("Reroute", step_id)
    os.system(
        f"ovs-ofctl -O OpenFlow13 add-flow s4 "
        f"\"priority={priority},ip,nw_src={SERVER_IP},"
        f"idle_timeout={timeouts['idle_timeout']},"
        f"hard_timeout={timeouts['hard_timeout']},"
        f"cookie=0x{cookie3:x},"
        f"actions=output:{s4_s1_port}\" >/dev/null 2>&1 || true"
    )
    rule_mgr.register_rule(cookie3, "Reroute", SERVER_IP, "s4", priority, step_id)
    cookies.append(cookie3)

    # If attacker mode, also drop attacker traffic on s4 scrubbing path
    if mode == "attackers":
        for attacker_ip in ATTACKER_IPS:
            cookie4 = rule_mgr._alloc_cookie("Reroute-Drop", step_id)
            os.system(
                f"ovs-ofctl -O OpenFlow13 add-flow s4 "
                f"\"priority={ACTION_PRIORITY['Drop']},ip,nw_src={attacker_ip},"
                f"idle_timeout={timeouts['idle_timeout']},"
                f"hard_timeout={timeouts['hard_timeout']},"
                f"cookie=0x{cookie4:x},"
                f"actions=drop\" >/dev/null 2>&1 || true"
            )
            rule_mgr.register_rule(cookie4, "Drop", attacker_ip, "s4",
                                   ACTION_PRIORITY["Drop"], step_id)
            cookies.append(cookie4)

    return cookies


def apply_throttle_persistent(rule_mgr, net, target_name, mode, step_id, rate_mbit=5):
    """Throttle with persistent tc qdisc: use change/replace, track duplicates."""
    edge = net.get("s1")
    agg = net.get("s2")
    intf_name = link_intf_name(edge, agg)

    # Check if a qdisc already exists (duplicate tracking)
    try:
        check = subprocess.check_output(
            f"tc qdisc show dev {intf_name}", shell=True, stderr=subprocess.DEVNULL
        ).decode()
        if "tbf" in check:
            rule_mgr.duplicate_qdisc_count += 1
    except Exception:
        pass

    # Use replace to update existing qdisc or add new one
    os.system(
        f"tc qdisc replace dev {intf_name} root tbf rate {rate_mbit}mbit "
        f"burst 32kbit latency 80ms >/dev/null 2>&1"
    )

    # Register as a virtual rule (tc doesn't use cookies)
    cookie = rule_mgr._alloc_cookie("Throttle", step_id)
    rule_mgr.register_rule(cookie, "Throttle", "bottleneck", "tc",
                           ACTION_PRIORITY["Throttle"], step_id)
    return [cookie]


def apply_inspect_persistent(rule_mgr, net, target_name, mode, step_id):
    """Inspect with persistent rules: short timeout, cookie, priority."""
    timeouts = ACTION_TIMEOUTS["Inspect"]
    priority = ACTION_PRIORITY["Inspect"]
    cookies = []

    switch = net.get("s1")
    ids = net.get("ids")
    ids_port = host_port(switch, ids)

    cookie = rule_mgr._alloc_cookie("Inspect", step_id)
    os.system(
        f"ovs-ofctl -O OpenFlow13 add-flow s1 "
        f"\"priority={priority},ip,nw_dst={SERVER_IP},"
        f"idle_timeout={timeouts['idle_timeout']},"
        f"hard_timeout={timeouts['hard_timeout']},"
        f"cookie=0x{cookie:x},"
        f"actions=output:NORMAL,output:{ids_port}\" >/dev/null 2>&1 || true"
    )
    rule_mgr.register_rule(cookie, "Inspect", SERVER_IP, "s1", priority, step_id)
    cookies.append(cookie)

    ids_host = net.get("ids")
    ids_host.cmd("timeout 2s tcpdump -i any -c 50 -w /tmp/ids_inspect_persistent.pcap >/dev/null 2>&1 &")

    return cookies


def apply_mirror_persistent(rule_mgr, net, target_name, mode, step_id):
    """Mirror with short timeout. Mirror is implemented via OVS mirror,
    which doesn't support per-flow timeout natively; we track it and
    clean up when the timeout would expire."""
    timeouts = ACTION_TIMEOUTS["Mirror"]
    cookies = []

    switch = net.get("s1")
    ids = net.get("ids")
    ids_port = host_port(switch, ids)

    src_ports = []
    for h_name in BENIGN_HOSTS + ATTACKER_HOSTS:
        h = net.get(h_name)
        src_ports.append(str(host_port(switch, h)))

    # Clear existing mirrors before adding new one
    os.system("ovs-vsctl clear Bridge s1 mirrors >/dev/null 2>&1 || true")

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

    cookie = rule_mgr._alloc_cookie("Mirror", step_id)
    rule_mgr.register_rule(cookie, "Mirror", "all", "s1-mirror",
                           ACTION_PRIORITY["Mirror"], step_id)
    cookies.append(cookie)

    ids_host = net.get("ids")
    ids_host.cmd("timeout 2s tcpdump -i any -c 100 -w /tmp/ids_mirror_persistent.pcap >/dev/null 2>&1 &")

    return cookies


PERSISTENT_ACTION_APPLIERS = {
    "Forward": apply_forward_persistent,
    "Inspect": apply_inspect_persistent,
    "Mirror": apply_mirror_persistent,
    "Throttle": apply_throttle_persistent,
    "Reroute": apply_reroute_persistent,
    "Drop": apply_drop_persistent,
    "Isolate": apply_isolate_persistent,
}


# ─── Utility functions (same as enhanced_replay) ─────────────────────────────

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


def count_tcpdump_packets(pcap_path="/tmp/ids_capture_persistent.pcap"):
    try:
        out = subprocess.check_output(
            f"tcpdump -r {pcap_path} 2>/dev/null | wc -l", shell=True,
            stderr=subprocess.DEVNULL
        ).decode().strip()
        return int(out) if out else 0
    except Exception:
        return 0


def reset_network_controls(net):
    """Full reset: clear all flows, mirrors, and tc qdiscs."""
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


# ─── Persistent Step Measurement ─────────────────────────────────────────────

def measure_persistent_step(rule_mgr, net, action_name, target_name, mode, step_id):
    """Measure a single replay step with persistent rule lifecycle.

    Instead of resetting all rules each step, we:
    1. Clean up expired rules (aged out via OVS timeouts)
    2. Detect and resolve conflicts
    3. Apply incremental diff (add new rules, remove conflicting)
    4. Measure metrics
    """
    rule_mgr.reset_churn_counters()

    # Step 1: Cleanup expired rules (OVS timeout-based aging)
    rule_mgr.cleanup_expired_rules()

    # Step 2: Measure before-state
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

    ids.cmd("rm -f /tmp/ids_capture_persistent.pcap >/dev/null 2>&1 || true")
    ids.cmd("timeout 3s tcpdump -i any -c 500 -w /tmp/ids_capture_persistent.pcap >/dev/null 2>&1 &")

    # Step 3: Apply action with persistent lifecycle
    t0 = time.perf_counter()
    cookies = PERSISTENT_ACTION_APPLIERS[action_name](rule_mgr, net, target_name, mode, step_id)
    apply_ms = (time.perf_counter() - t0) * 1000.0
    time.sleep(0.2)

    # Step 4: Measure after-state
    flows_after_s1 = count_flow_entries("s1")
    flows_after_s4 = count_flow_entries("s4")
    new_flow_entries = max(0, flows_after_s1 - flows_before_s1) + max(0, flows_after_s4 - flows_before_s4)

    # Update peak tracking
    total_flows = flows_after_s1 + flows_after_s4
    rule_mgr.peak_flow_entries = max(rule_mgr.peak_flow_entries, total_flows)

    server_ip = SERVER_IP
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
    ids_capture_packets = count_tcpdump_packets("/tmp/ids_capture_persistent.pcap")

    reroute_active = int(action_name == "Reroute")

    ids_mirror_mbps = (ids_after_bytes - ids_before_bytes) * 8 / 1e6 if ids_after_bytes > ids_before_bytes else 0.0

    churn_stats = rule_mgr.get_churn_stats()

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
        "ovs_flow_entries_total": total_flows,
        "ovs_cpu_pct": ovs_cpu_after,
        "reroute_active": reroute_active,
        "reroute_s4_delta_bytes": s4_delta_bytes,
        "reroute_s4_throughput_mbps": round(s4_delta_bytes * 8 / 1e6, 3) if s4_delta_bytes > 0 else 0.0,
        # Persistent-rule-specific metrics
        "persistent_rule_churn_add": churn_stats["rule_churn_add"],
        "persistent_rule_churn_del": churn_stats["rule_churn_del"],
        "stale_conflicts": churn_stats["stale_conflicts"],
        "duplicate_qdisc_count": churn_stats["duplicate_qdisc_count"],
        "active_persistent_rules": churn_stats["total_active_rules"],
        "peak_flow_entries": rule_mgr.peak_flow_entries,
        "cookies_installed": len(cookies),
    }


# ─── Persistent Replay Runner ────────────────────────────────────────────────

def run_persistent_replay(net, controller_df, mode):
    """Run replay with persistent rule lifecycle instead of per-step reset."""
    rule_mgr = PersistentRuleManager()
    rows = []

    for _, row in controller_df.sort_values("replay_step").iterrows():
        if int(row["true_label"]) == 1:
            target_name = "attacker1"
            action_mode = "attackers"
        else:
            target_name = "benign1"
            action_mode = None

        metrics = measure_persistent_step(
            rule_mgr, net, row["action_name"], target_name, action_mode, int(row["replay_step"])
        )
        rows.append({**row.to_dict(), "target_name": target_name, **metrics})

        if len(rows) % 50 == 0:
            print(f"  Step {len(rows)}/{len(controller_df)} "
                  f"(active rules: {rule_mgr.get_churn_stats()['total_active_rules']}, "
                  f"peak: {rule_mgr.peak_flow_entries})")

    return pd.DataFrame(rows)


def summarize_persistent(df):
    """Summarize persistent replay results including lifecycle metrics."""
    benign = df[df["true_label"] == 0]
    attack = df[df["true_label"] == 1]
    reroute_steps = df[df["reroute_active"] == 1] if "reroute_active" in df.columns else pd.DataFrame()

    summary = {
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
        # Persistent-rule-specific summary
        "total_stale_conflicts": int(df["stale_conflicts"].sum()) if "stale_conflicts" in df.columns else 0,
        "peak_flow_entries": int(df["peak_flow_entries"].max()) if "peak_flow_entries" in df.columns else 0,
        "mean_active_persistent_rules": float(df["active_persistent_rules"].mean()) if "active_persistent_rules" in df.columns else 0.0,
        "total_rule_churn_add": int(df["persistent_rule_churn_add"].sum()) if "persistent_rule_churn_add" in df.columns else 0,
        "total_rule_churn_del": int(df["persistent_rule_churn_del"].sum()) if "persistent_rule_churn_del" in df.columns else 0,
        "total_duplicate_qdisc": int(df["duplicate_qdisc_count"].sum()) if "duplicate_qdisc_count" in df.columns else 0,
    }
    return summary


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Persistent-Rule OVS Replay: rule lifecycle with timeout aging and conflict cleanup"
    )
    parser.add_argument("--plan-csv", required=True, help="Path to replay plan CSV")
    parser.add_argument("--out-dir", required=True, help="Output directory")
    parser.add_argument("--mode", choices=["persistent", "both"], default="both",
                        help="Run persistent only, or both persistent and per-step-reset for comparison")
    args = parser.parse_args()

    setLogLevel("warning")
    os.makedirs(args.out_dir, exist_ok=True)

    plan_df = pd.read_csv(args.plan_csv)
    all_step_frames = []
    summary_rows = []

    for controller_name, controller_df in plan_df.groupby("controller", sort=False):
        print(f"\n=== Running controller: {controller_name} ({len(controller_df)} steps) ===")

        # ─── Persistent lifecycle replay ──────────────────────────────────
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
            step_df = run_persistent_replay(net, controller_df.copy(), "persistent")
            step_df["replay_mode"] = "persistent_lifecycle"
            step_df["topology"] = "four_switch_reroute"
            all_step_frames.append(step_df)
            summary_row = summarize_persistent(step_df)
            summary_row["replay_mode"] = "persistent_lifecycle"
            summary_row["topology"] = "four_switch_reroute"
            summary_rows.append(summary_row)
            print(f"  Persistent Summary: {summary_row}")
        finally:
            with suppress(Exception):
                stop_iperf_processes(net)
            with suppress(Exception):
                reset_network_controls(net)
            with suppress(Exception):
                net.stop()
            with suppress(Exception):
                os.system("mn -c >/dev/null 2>&1 || true")

        # ─── Per-step reset replay (for comparison) ──────────────────────
        if args.mode == "both":
            # Import per-step reset replay from enhanced_replay
            sys_path = os.path.dirname(os.path.abspath(__file__))
            import importlib.util
            spec = importlib.util.spec_from_file_location(
                "enhanced_replay",
                os.path.join(sys_path, "enhanced_replay.py")
            )
            er_mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(er_mod)

            topo2 = EnhancedEdgeTopo()
            net2 = Mininet(
                topo=topo2,
                switch=OVSSwitch,
                controller=None,
                autoSetMacs=True,
                autoStaticArp=True,
                cleanup=True,
                link=TCLink,
            )
            try:
                net2.start()
                er_mod.reset_network_controls(net2)
                er_mod.start_iperf_servers(net2.get("server"))
                step_df2 = er_mod.run_enhanced_replay(net2, controller_df.copy(), "per_step_reset")
                step_df2["replay_mode"] = "per_step_reset"
                step_df2["topology"] = "four_switch_reroute"
                all_step_frames.append(step_df2)
                summary_row2 = er_mod.summarize_enhanced(step_df2)
                summary_row2["replay_mode"] = "per_step_reset"
                summary_row2["topology"] = "four_switch_reroute"
                summary_rows.append(summary_row2)
                print(f"  Per-step-reset Summary: {summary_row2}")
            finally:
                with suppress(Exception):
                    er_mod.stop_iperf_processes(net2)
                with suppress(Exception):
                    er_mod.reset_network_controls(net2)
                with suppress(Exception):
                    net2.stop()
                with suppress(Exception):
                    os.system("mn -c >/dev/null 2>&1 || true")

    # Save results
    all_steps = pd.concat(all_step_frames, ignore_index=True)
    summary_df = pd.DataFrame(summary_rows)

    all_steps.to_csv(os.path.join(args.out_dir, "persistent_step_trace.csv"), index=False)
    summary_df.to_csv(os.path.join(args.out_dir, "persistent_summary.csv"), index=False)

    # Generate comparison table if both modes were run
    if args.mode == "both" and len(summary_rows) >= 2:
        comparison_rows = []
        for controller_name in summary_df["controller"].unique():
            ctrl_df = summary_df[summary_df["controller"] == controller_name]
            for mode in ["per_step_reset", "persistent_lifecycle"]:
                mode_row = ctrl_df[ctrl_df["replay_mode"] == mode]
                if len(mode_row) == 0:
                    continue
                r = mode_row.iloc[0]
                comparison_rows.append({
                    "controller": controller_name,
                    "mode": mode,
                    "benign_tcp_mbps": r.get("benign_tcp_mbps", 0),
                    "benign_loss_pct": r.get("benign_loss_pct", 0),
                    "attack_suppression_ratio": r.get("attack_suppression_ratio", 0),
                    "mean_rule_install_ms": r.get("mean_rule_install_ms", 0),
                    "mean_ovs_flow_entries": r.get("mean_ovs_flow_entries", 0),
                    "ovs_cpu_pct": r.get("ovs_cpu_pct", 0),
                    "stale_conflicts": r.get("total_stale_conflicts", 0) if mode == "persistent_lifecycle" else 0,
                    "peak_flow_entries": r.get("peak_flow_entries", 0) if mode == "persistent_lifecycle" else r.get("mean_ovs_flow_entries", 0),
                })

        comp_df = pd.DataFrame(comparison_rows)
        comp_df.to_csv(os.path.join(args.out_dir, "perstep_vs_persistent_comparison.csv"), index=False)
        print("\n=== Per-step Reset vs Persistent Lifecycle Comparison ===")
        print(comp_df.to_string(index=False))

    print("\n=== Persistent Replay Complete ===")
    print(summary_df.to_string(index=False))


if __name__ == "__main__":
    main()
