"""
Two-Switch Mixed Bottleneck Replay Topology

Topology:
    client (10.0.0.1) ----\
    attacker (10.0.0.2) ---+--> s1 (edge) --> s2 (agg) --> server (10.0.0.4)
    ids (10.0.0.3) --------/       |
                                   +--> IDS port (mirror target)

Edge-to-aggregation link: 40 Mbps, 5ms delay (bottleneck)
Other links: 100 Mbps, 2ms delay
"""
from mininet.link import TCLink
from mininet.topo import Topo


class TwoSwitchBottleneckTopo(Topo):
    """Two-switch topology with bottleneck link for mixed-window replay."""
    
    def build(self):
        # Add hosts
        client = self.addHost("client", ip="10.0.0.1/24")
        attacker = self.addHost("attacker", ip="10.0.0.2/24")
        ids = self.addHost("ids", ip="10.0.0.3/24")
        server = self.addHost("server", ip="10.0.0.4/24")
        
        # Add switches
        edge = self.addSwitch("s1", protocols="OpenFlow13", failMode="standalone")
        agg = self.addSwitch("s2", protocols="OpenFlow13", failMode="standalone")
        
        # Edge links (100 Mbps, 2ms)
        self.addLink(client, edge, cls=TCLink, bw=100, delay="2ms", loss=0, use_htb=True)
        self.addLink(attacker, edge, cls=TCLink, bw=100, delay="2ms", loss=0, use_htb=True)
        self.addLink(ids, edge, cls=TCLink, bw=100, delay="2ms", loss=0, use_htb=True)
        
        # Bottleneck link (40 Mbps, 5ms) - this is where congestion occurs
        self.addLink(edge, agg, cls=TCLink, bw=40, delay="5ms", loss=0, use_htb=True)
        
        # Server link (100 Mbps, 2ms)
        self.addLink(server, agg, cls=TCLink, bw=100, delay="2ms", loss=0, use_htb=True)


# Port mapping:
# s1 (edge switch):
#   s1-eth1 -> client (10.0.0.1)
#   s1-eth2 -> attacker (10.0.0.2)
#   s1-eth3 -> ids (10.0.0.3)
#   s1-eth4 -> s2 (aggregation switch)
#
# s2 (aggregation switch):
#   s2-eth1 -> s1 (edge switch)
#   s2-eth2 -> server (10.0.0.4)
#
# The bottleneck is the s1-s2 link (40 Mbps)
# When both client and attacker send traffic, they share this 40 Mbps pipe
