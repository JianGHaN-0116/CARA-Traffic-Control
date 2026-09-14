import os, sys, pickle
import numpy as np
BASE="/home/penglizhi/jh/IoT/CARA-Traffic-Control"
sys.path.insert(0, BASE)
from src.envs.edge_network_env import EdgeTrafficSecurityEnv
from src.experiments.resource_aware_threshold_baseline import CARATCPolicy
D=os.path.join(BASE,"data/processed/cicids2017_cap10000_ws25_thr07")
EDGE=dict(conf_strict=0.85,isolate_confidence=0.90,detector_ratio_strict=0.80,queue_threshold=0.65,link_threshold=0.65)
RETU=dict(conf_strict=0.5,isolate_confidence=0.7,detector_ratio_strict=0.7,queue_threshold=0.55,link_threshold=0.55)
NAMES=["Fwd","Insp","Mirr","Throt","Reroute","Drop","Isol"]
def dist(pol, path, thr=0.7):
    env=EdgeTrafficSecurityEnv(window_path=path,state_dim=80,max_steps=20000,reward_config={"attack_threshold":thr},shuffle_on_reset=False)
    obs,_=env.reset(); counts=np.zeros(7,int); done=False
    while not done:
        a=int(pol.predict(obs)); counts[a]+=1
        obs,_,t,tr,info=env.step(a); done=t or tr
    env.close(); return counts
tp=os.path.join(D,"test_windows.pkl")
for name,g in [("CARA-TC (Edge-frozen)",EDGE),("CARA-TC (CIC-retuned)",RETU)]:
    c=dist(CARATCPolicy(**g), tp)
    print(name, " ".join(f"{n}={v}" for n,v in zip(NAMES,c)))
