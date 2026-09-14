"""
Single-Switch Targeted Replay Topology

Topology:
    client (10.0.0.1) ----\
    attacker (10.0.0.2) ---+--> s1 (edge) --> server (10.0.0.4)
    ids (10.0.0.3) --------/         |
                                     +--> IDS port (mirror target)

All links: 100 Mbps, 2ms delay
"""
from mininet.link import TCLink
from mininet.topo import Topo


class SingleSwitchTargetedTopo(Topo):
    """Single-switch topology for targeted action replay."""
    
    def build(self):
        # Add hosts
        client = self.addHost("client", ip="10.0.0.1/24")
        attacker = self.addHost("attacker", ip="10.0.0.2/24")
        ids = self.addHost("ids", ip="10.0.0.3/24")
        server = self.addHost("server", ip="10.0.0.4/24")
        
        # Add switch
        switch = self.addSwitch("s1", protocols="OpenFlow13", failMode="standalone")
        
        # Add links (all 100 Mbps, 2ms delay)
        for host in (client, attacker, ids, server):
            self.addLink(
                host, switch,
                cls=TCLink,
                bw=100,      # Bandwidth in Mbps
                delay="2ms",  # One-way delay
                loss=0,       # No packet loss
                use_htb=True, # Use HTB qdisc
            )


# Port mapping:
# s1-eth1 -> client (10.0.0.1)
# s1-eth2 -> attacker (10.0.0.2)
# s1-eth3 -> ids (10.0.0.3)
# s1-eth4 -> server (10.0.0.4)
