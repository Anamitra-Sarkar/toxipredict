#!/usr/bin/env python
"""
ToxiPredict - Multi-Task GNN Training on Kaggle GPU
This script trains an uncertainty-aware Multi-Task GNN on the Tox21 dataset,
using 5-fold Bemis-Murcko scaffold cross-validation, with early stopping.

Results are uploaded to HuggingFace: Arko007/toxipredict-gnn-models
"""

import os, sys, json, subprocess, warnings, time, math
warnings.filterwarnings("ignore", category=UserWarning, module="torch")

subprocess.run(["pip", "install", "-q",
    "torch-geometric", "rdkit",
    "matplotlib", "safetensors", "huggingface_hub"], check=True)

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GATConv, global_mean_pool
from torch_geometric.data import Data
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem.Scaffolds import MurckoScaffold
from safetensors.torch import save_file
from huggingface_hub import HfApi, create_repo
from collections import defaultdict

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE} | CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")

TASK_NAMES = ["NR-AR","NR-AhR","NR-Aromatase","NR-ER",
              "NR-PPAR-gamma","SR-ARE","SR-ATAD5","SR-HSE",
              "SR-MMP","SR-p53"]
NUM_TASKS = len(TASK_NAMES)
NODE_DIM, EDGE_DIM, HIDDEN_DIM, DROPOUT = 15, 6, 128, 0.15
LR, WEIGHT_DECAY, BATCH_SIZE = 1e-3, 1e-5, 64
EPOCHS, PATIENCE = 200, 20

ATOM_ENCODER = {6:[1,0,0,0,0,0,0],7:[0,1,0,0,0,0,0],8:[0,0,1,0,0,0,0],
                16:[0,0,0,1,0,0,0],15:[0,0,0,0,1,0,0],17:[0,0,0,0,0,1,0],
                35:[0,0,0,0,0,1,0],9:[0,0,0,0,0,1,0],53:[0,0,0,0,0,1,0]}

BOND_ENCODER = {Chem.rdchem.BondType.SINGLE:[1,0,0,0],
                Chem.rdchem.BondType.DOUBLE:[0,1,0,0],
                Chem.rdchem.BondType.TRIPLE:[0,0,1,0],
                Chem.rdchem.BondType.AROMATIC:[0,0,0,1]}

def get_atom_features(atom):
    at = ATOM_ENCODER.get(atom.GetAtomicNum(), [0,0,0,0,0,0,0])
    h = atom.GetHybridization()
    hm = {Chem.rdchem.HybridizationType.SP:[1,0,0,0,0],
          Chem.rdchem.HybridizationType.SP2:[0,1,0,0,0],
          Chem.rdchem.HybridizationType.SP3:[0,0,1,0,0],
          Chem.rdchem.HybridizationType.SP3D:[0,0,0,1,0],
          Chem.rdchem.HybridizationType.SP3D2:[0,0,0,0,1]}
    hy = hm.get(h, [0,0,0,0,0])
    d = min(atom.GetDegree(),6)/6.0
    v = min(atom.GetTotalValence(),8)/8.0
    fc = np.clip(atom.GetFormalCharge(),-3,3)
    ch = (fc+3)/6.0
    ar = 1.0 if atom.GetIsAromatic() else 0.0
    hc = min(atom.GetTotalNumHs(),4)/4.0
    return at+hy+[d,v,ch,ar,hc]

def smiles_to_graph(smi):
    mol = Chem.MolFromSmiles(smi)
    if mol is None: return None
    try:
        mol = Chem.AddHs(mol); AllChem.EmbedMolecule(mol,randomSeed=42)
        AllChem.MMFFOptimizeMolecule(mol); mol = Chem.RemoveHs(mol)
    except: mol = Chem.RemoveHs(mol)
    x = torch.tensor([get_atom_features(a) for a in mol.GetAtoms()], dtype=torch.float32)
    ei, ef = [], []
    for b in mol.GetBonds():
        i,j = b.GetBeginAtomIdx(),b.GetEndAtomIdx()
        bt = BOND_ENCODER.get(b.GetBondType(),[0,0,0,0])
        bf = bt + [1.0 if b.GetIsConjugated() else 0.0]
        ei.extend([[i,j],[j,i]]); ef.extend([bf,bf])
    if ei:
        edge_index = torch.tensor(ei,dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(ef,dtype=torch.float32)
    else:
        edge_index = torch.zeros((2,0),dtype=torch.long)
        edge_attr = torch.zeros((0,6),dtype=torch.float32)
    return Data(x=x,edge_index=edge_index,edge_attr=edge_attr)

class MultiTaskGNN(nn.Module):
    def __init__(self,ic,ed,hd,nt,dp=0.15):
        super().__init__()
        self.nt = nt
        self.c1 = GATConv(ic,hd,heads=4,concat=True,edge_dim=ed,dropout=dp)
        self.b1 = nn.BatchNorm1d(hd*4)
        self.c2 = GATConv(hd*4,hd,heads=4,concat=False,edge_dim=ed,dropout=dp)
        self.b2 = nn.BatchNorm1d(hd)
        self.fc = nn.Sequential(nn.Linear(hd,hd),nn.ReLU(),nn.Dropout(dp))
        self.heads = nn.ModuleList([nn.Linear(hd,1) for _ in range(nt)])
    def forward(self,x,ei,ea,b):
        h = F.relu(self.b1(self.c1(x,ei,ea)))
        h = F.relu(self.b2(self.c2(h,ei,ea)))
        h = global_mean_pool(h,b)
        h = self.fc(h)
        return torch.cat([hd(h) for hd in self.heads],dim=1)

class HomoLoss(nn.Module):
    def __init__(self,nt): super().__init__(); self.s = nn.Parameter(torch.zeros(nt))
    def forward(self,logits,y,mask):
        total = 0.0; w = torch.exp(-self.s)
        for t in range(self.s.shape[0]):
            v = torch.where(mask[:,t]==1)[0]
            if len(v)<1: continue
            total += w[t]*F.binary_cross_entropy_with_logits(logits[v,t],y[v,t].float(),reduction='mean')+0.5*self.s[t]
        return total, w.detach()

def get_scaffold(smi):
    mol = Chem.MolFromSmiles(smi)
    if mol is None: return ""
    try: return Chem.MolToSmiles(MurckoScaffold.GetScaffoldForMol(mol),canonical=True)
    except: return ""

def scaffold_split(smiles,n_splits=5):
    sc = [get_scaffold(s) for s in smiles]
    g = defaultdict(list)
    for i,s in enumerate(sc): g[s].append(i)
    sg = sorted(g.items(),key=lambda x:len(x[1]),reverse=True)
    folds = [[] for _ in range(n_splits)]
    sizes = np.zeros(n_splits,dtype=int)
    for _,ix in sg:
        f = np.argmin(sizes); folds[f].extend(ix); sizes[f]+=len(ix)
    a = np.zeros(len(smiles),dtype=int)
    for f,ix in enumerate(folds):
        for i in ix: a[i]=f
    return a

def metrics(logits,y,mask):
    p = torch.sigmoid(logits).cpu().numpy()
    yn, mn = y.cpu().numpy(), mask.cpu().numpy()
    aucs,f1s,accs = [],[],[]
    for t in range(NUM_TASKS):
        m = mn[:,t]==1
        if m.sum()<5: continue
        yt, yp = yn[m,t], p[m,t]
        yd = (yp>0.5).astype(int)
        if len(np.unique(yt))>1:
            try: aucs.append(roc_auc_score(yt,yp))
            except: pass
        f1s.append(f1_score(yt,yd,zero_division=0))
        accs.append(accuracy_score(yt,yd))
    return {"auc":float(np.mean(aucs)) if aucs else 0.0,
            "f1":float(np.mean(f1s)) if f1s else 0.0}

def main():
    print("="*60); print("ToxiPredict - Kaggle GPU Training"); print("="*60)
    print("\nLoading Tox21...")
    df = pd.read_csv("https://github.com/deepchem/deepchem/raw/master/datasets/tox21.csv.gz")
    print(f"Loaded {len(df)} compounds")
    smiles = df["smiles"].tolist()
    ys, ms = [], []
    for _,r in df.iterrows():
        yl, ml = [], []
        for t in TASK_NAMES:
            v=r.get(t)
            if pd.isna(v): yl.append(0); ml.append(0)
            else: yl.append(int(v)); ml.append(1)
        ys.append(yl); ms.append(ml)
    yt = torch.tensor(ys,dtype=torch.float32)
    mt = torch.tensor(ms,dtype=torch.float32)
    for t in range(NUM_TASKS):
        v = mt[:,t]==1
        nv = v.sum().item()
        print(f"  {TASK_NAMES[t]:15s} n={nv:5d} pos={yt[v,t].mean().item():.3f} miss={1-nv/len(df):.3f}")

    print("\nScaffold splitting...")
    folds = scaffold_split(smiles,5)
    for f in range(5): print(f"  Fold {f}: {(folds==f).sum()}")

    all_auc = []
    for fold in range(5):
        print(f"\n{'='*45}\nFOLD {fold+1}/5\n{'='*45}")
        train_g, val_g = [], []
        for i in np.where(folds!=fold)[0]:
            g = smiles_to_graph(smiles[i])
            if g is not None: g.y=yt[i].unsqueeze(0); g.mask=mt[i].unsqueeze(0); train_g.append(g)
        for i in np.where(folds==fold)[0]:
            g = smiles_to_graph(smiles[i])
            if g is not None: g.y=yt[i].unsqueeze(0); g.mask=mt[i].unsqueeze(0); val_g.append(g)
        print(f"Train: {len(train_g)}, Val: {len(val_g)}")
        tl = DataLoader(train_g,BATCH_SIZE,shuffle=True)
        vl = DataLoader(val_g,BATCH_SIZE,shuffle=False)
        model = MultiTaskGNN(NODE_DIM,EDGE_DIM,HIDDEN_DIM,NUM_TASKS,DROPOUT).to(DEVICE)
        crit = HomoLoss(NUM_TASKS).to(DEVICE)
        opt = Adam(list(model.parameters())+list(crit.parameters()),lr=LR,weight_decay=WEIGHT_DECAY)
        best_auc, best_sd, pc = 0.0, None, 0
        for ep in range(EPOCHS):
            model.train(); tl_loss=0.0
            for b in tl:
                b=b.to(DEVICE); opt.zero_grad()
                lo = model(b.x,b.edge_index,b.edge_attr,b.batch)
                l,_ = crit(lo,b.y,b.mask); l.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(),5.0)
                opt.step(); tl_loss+=l.item()
            model.eval(); vlo,vla,vma=[],[],[]
            with torch.no_grad():
                for b in vl: b=b.to(DEVICE); lo=model(b.x,b.edge_index,b.edge_attr,b.batch); vlo.append(lo.cpu()); vla.append(b.y.cpu()); vma.append(b.mask.cpu())
            vlo=torch.cat(vlo); vla=torch.cat(vla); vma=torch.cat(vma)
            m = metrics(vlo,vla,vma)
            st = crit.s.detach().cpu().numpy()
            if (ep+1)%5==0 or ep==0:
                print(f"  Ep {ep+1:3d} | Train: {tl_loss/len(tl):.4f} | Val AUC: {m['auc']:.4f} F1: {m['f1']:.4f} | s_t: [{st.min():.2f},{st.max():.2f}]")
            if m["auc"]>best_auc:
                best_auc=m["auc"]; best_sd=model.state_dict(); pc=0
            else: pc+=1
            if pc>=PATIENCE: print(f"  Early stop at ep {ep+1}"); break
        print(f"  Fold {fold+1} best Val AUC: {best_auc:.4f}")
        all_auc.append(best_auc)

    print(f"\n{'='*60}\n5-Fold CV AUC: {np.mean(all_auc):.4f} +/- {np.std(all_auc):.4f}\n{'='*60}")
    print("\nFinal training on full dataset...")
    all_g = []
    for i in range(len(smiles)):
        g = smiles_to_graph(smiles[i])
        if g is not None: g.y=yt[i].unsqueeze(0); g.mask=mt[i].unsqueeze(0); all_g.append(g)
    fl = DataLoader(all_g,BATCH_SIZE,shuffle=True)
    fm = MultiTaskGNN(NODE_DIM,EDGE_DIM,HIDDEN_DIM,NUM_TASKS,DROPOUT).to(DEVICE)
    fc = HomoLoss(NUM_TASKS).to(DEVICE)
    fo = Adam(list(fm.parameters())+list(fc.parameters()),lr=LR,weight_decay=WEIGHT_DECAY)
    best_loss, best_sd, pc = float("inf"), None, 0
    for ep in range(EPOCHS):
        fm.train(); tl=0.0
        for b in fl:
            b=b.to(DEVICE); fo.zero_grad()
            lo = fm(b.x,b.edge_index,b.edge_attr,b.batch)
            l,w = fc(lo,b.y,b.mask); l.backward()
            torch.nn.utils.clip_grad_norm_(fm.parameters(),5.0)
            fo.step(); tl+=l.item()
        al = tl/len(fl)
        st = fc.s.detach().cpu().numpy()
        if (ep+1)%10==0:
            print(f"  Ep {ep+1:3d} | Loss: {al:.4f} | w: [{st.min():.2f},{st.max():.2f}] | {w.cpu().numpy().round(3)}")
        if al<best_loss: best_loss=al; best_sd=fm.state_dict(); pc=0
        else: pc+=1
        if pc>=PATIENCE: print(f"  Early stop at ep {ep+1}"); break
    print("\nUploading to HuggingFace...")
    os.makedirs("/kaggle/working/checkpoints",exist_ok=True)
    torch.save(best_sd,"/kaggle/working/checkpoints/model_final.pt")
    save_file(best_sd,"/kaggle/working/checkpoints/model.safetensors")
    cfg = {"node_dim":NODE_DIM,"edge_dim":EDGE_DIM,"hidden_dim":HIDDEN_DIM,
           "num_tasks":NUM_TASKS,"dropout":DROPOUT,"task_names":TASK_NAMES,
           "architecture":"MultiTaskGNN","version":"1.0.0",
           "cv_auc_mean":float(np.mean(all_auc)),"cv_auc_std":float(np.std(all_auc))}
    with open("/kaggle/working/checkpoints/model_config.json","w") as f: json.dump(cfg,f,indent=2)
    HF_TOKEN = os.environ.get("HF_TOKEN","")
    api = HfApi(token=HF_TOKEN)
    create_repo("Arko007/toxipredict-gnn-models",private=False,exist_ok=True,token=HF_TOKEN)
    for fn in ["model.safetensors","model_config.json"]:
        api.upload_file(path_or_fileobj=f"/kaggle/working/checkpoints/{fn}",
                        path_in_repo=fn,repo_id="Arko007/toxipredict-gnn-models",token=HF_TOKEN)
    print(f"\nUploaded to: https://huggingface.co/Arko007/toxipredict-gnn-models")
    print(f"CV AUC: {cfg['cv_auc_mean']:.4f} +/- {cfg['cv_auc_std']:.4f}")
    print("Training complete!")

if __name__ == "__main__":
    main()
