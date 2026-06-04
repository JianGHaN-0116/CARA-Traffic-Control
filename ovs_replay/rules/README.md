# OpenFlow Rules for Controller Actions
# 
# These files contain the OVS flow rules used in the Mininet replay.
# Each file corresponds to one controller action.

# Default forwarding rule (Forward/Inspect)
# ovs-ofctl -O OpenFlow13 add-flow s1 "priority=0,actions=NORMAL"
