import json,hashlib,sys
from pathlib import Path
import numpy as np,pandas as pd,torch,torch.nn as nn
ROOT=Path(__file__).resolve().parents[2]; SEEDS=[20260811,20260812,20260813,20260814,20260815]
class FCN(nn.Module):
 def __init__(self):
  super().__init__();self.c=nn.Sequential(nn.Conv1d(4,128,51,padding=25),nn.BatchNorm1d(128),nn.ReLU(),nn.Dropout(0),nn.Conv1d(128,256,25,padding=12),nn.BatchNorm1d(256),nn.ReLU(),nn.Dropout(0),nn.Conv1d(256,128,13,padding=6),nn.BatchNorm1d(128),nn.ReLU(),nn.Dropout(0),nn.AdaptiveMaxPool1d(1));self.f=nn.Sequential(nn.Linear(128,64),nn.BatchNorm1d(64),nn.ReLU(),nn.Dropout(0));self.o=nn.Sequential(nn.Dropout(0),nn.Linear(64,1))
 def forward(self,x):return self.o(self.f(self.c(x).squeeze(-1))).squeeze(1)
def sh(p):
 h=hashlib.sha256();
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
def main():
 cache=ROOT/'02_data/interim/challenge2015_external_v1';out=ROOT/'04_analysis/runs/stage1f_v1';out.mkdir(parents=True,exist_ok=True); X=np.load(cache/'X.npy');idx=pd.read_csv(cache/'index.csv');
 if X.shape!=(341,4,2500) or not np.isfinite(X).all() or len(idx)!=341 or int(idx.y_true.sum())!=89:raise RuntimeError('external_input_gate')
 dev='cuda' if torch.cuda.is_available() else 'cpu';ps=[];modeldir=ROOT/'04_analysis/runs/stage1c_v1/models'
 for s in SEEDS:
  m=FCN().to(dev);m.load_state_dict(torch.load(modeldir/f'fcn_realtime_seed_{s}.pt',map_location=dev));m.eval()
  with torch.no_grad():v=torch.sigmoid(m(torch.from_numpy(X).to(dev))).cpu().numpy()
  if not np.isfinite(v).all() or ((v<0)|(v>1)).any():raise RuntimeError('invalid_probability')
  ps.append(v)
 df=idx.copy();df.insert(0,'split','external')
 for s,v in zip(SEEDS,ps):df[f'p_seed_{s}']=v
 df['score']=np.mean(np.stack(ps),axis=0)
 if df.event.duplicated().any() or len(df)!=341 or int(df.y_true.sum())!=89 or not np.isfinite(df.score).all():raise RuntimeError('external_score_gate')
 sp=out/'external_scores.csv';df[['split','record','event','y_true','score']].to_csv(sp,index=False);df.to_csv(out/'external_scores_with_seeds.csv',index=False);(out/'external_scores.sha256').write_text(sh(sp)+'  external_scores.csv\n',encoding='ascii');(out/'external_inference_manifest.json').write_text(json.dumps({'rows':341,'true_vt':89,'false_vt':252,'models':{str(s):sh(modeldir/f'fcn_realtime_seed_{s}.pt') for s in SEEDS},'score_hash':sh(sp)},indent=2)+'\n',encoding='ascii');print(json.dumps({'rows':341,'true_vt':89,'score_hash':sh(sp)},indent=2))
if __name__=='__main__':main()
