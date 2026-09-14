import os, sys, pickle
import numpy as np
BASE="/home/penglizhi/jh/IoT/CARA-Traffic-Control"
sys.path.insert(0, BASE)
from src.envs.edge_network_env import EdgeTrafficSecurityEnv
from src.experiments.resource_aware_threshold_baseline import CARATCPolicy
D=os.path.join(BASE,"data/processed/cicids2017_cap10000_ws25_thr07")
EDGE=dict(conf_strict=0.85,isolate_confidence=0.90,detector_ratio_strict=0.80,queue_threshold=0.65,link_threshold=0.65)
tp=os.path.join(D,"test_windows.pkl")
def ev(thr):
    env=EdgeTrafficSecurityEnv(window_path=tp,state_dim=80,max_steps=20000,reward_config={"attack_threshold":thr},shuffle_on_reset=False)
    pol=CARATCPolicy(**EDGE); obs,_=env.reset()
    tb=ta=bs=am=bd=0; done=False
    while not done:
        a=int(pol.predict(obs)); obs,_,t,tr,info=env.step(a); done=t or tr
        if info["attack_ratio"]>thr: ta+=1; am+= a in {5,6}
        else: tb+=1; bs+= a in {0,1,2}; bd+= a in {5,6}
    env.close(); return bs/max(tb,1), am/max(ta,1), bd/max(tb,1), tb, ta
for thr in [0.5,0.6,0.7,0.8,0.84]:
    print(f"thr={thr}: BenSafe={ev(thr)[0]:.4f} AtkMit={ev(thr)[1]:.4f} BenDrop={ev(thr)[2]:.4f} nb={ev(thr)[3]} na={ev(thr)[4]}")
