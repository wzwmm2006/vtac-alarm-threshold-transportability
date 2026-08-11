import hashlib,json,shutil
from pathlib import Path
import numpy as np,pandas as pd,matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
ROOT=Path(__file__).resolve().parents[2]
def sh(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for block in iter(lambda:f.read(1048576),b''):h.update(block)
 return h.hexdigest()
def write(p,s):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(s,encoding='utf-8')
res=ROOT/'05_results/current'; figs=ROOT/'06_figures/current'; final=ROOT/'04_analysis/final'; man=ROOT/'09_reproducibility/manifests'
for p in [res,figs,final]:p.mkdir(parents=True,exist_ok=True)
# Locked source artifacts
vmet=pd.read_csv(ROOT/'04_analysis/runs/stage1d_v1/stage1_threshold_lock/validation_safety_utility_metrics.csv');tmet=pd.read_csv(ROOT/'04_analysis/runs/stage1e_v1/stage1_test_frozen_results/safety_utility_metrics.csv');emet=pd.read_csv(ROOT/'04_analysis/runs/stage1f_v1/stage1_external_frozen_results/safety_utility_metrics.csv')
va=json.loads((ROOT/'04_analysis/runs/stage1d_v1/validation_descriptive_metrics.json').read_text());ta=json.loads((ROOT/'04_analysis/runs/stage1e_v1/test_descriptive_metrics.json').read_text());ea=json.loads((ROOT/'04_analysis/runs/stage1f_v1/external_descriptive_metrics.json').read_text())
# Primary rows with locked exact values
src=[('Validation',vmet[vmet.target_sensitivity==.95].iloc[0],va['auroc']['score']),('VTaC Test',tmet[tmet.target_sensitivity==.95].iloc[0],{'auroc':ta['test_auroc']['estimate'],'ci_low':ta['test_auroc']['ci_low'],'ci_high':ta['test_auroc']['ci_high']}),('Challenge 2015 External',emet[(emet.target_sensitivity==.95)&(emet.split=='external')].iloc[0],{'auroc':ea['external_auroc']['estimate'],'ci_low':ea['external_auroc']['ci_low'],'ci_high':ea['external_auroc']['ci_high']})]
rows=[]
for name,r,a in src:rows.append({'Cohort':name,'N':int(r.n),'True VT':int(r.true_vt),'False VT':int(r.false_vt),'AUROC (95% CI)':f"{a['auroc']:.4f} ({a['ci_low']:.4f}-{a['ci_high']:.4f})",'Frozen threshold':float(r.threshold),'TP':int(r.tp),'FN':int(r.fn),'TN':int(r.tn),'FP':int(r.fp),'Sensitivity (95% CI)':f"{r.sensitivity:.4f} ({r.sensitivity_ci_low:.4f}-{r.sensitivity_ci_high:.4f})",'FASR (95% CI)':f"{r.fasr:.4f} ({r.fasr_ci_low:.4f}-{r.fasr_ci_high:.4f})"})
primary=pd.DataFrame(rows);primary.to_csv(res/'primary_results.csv',index=False)
trade=[]
for name,frame in [('Validation',vmet),('VTaC Test',tmet[tmet.split=='test']),('Challenge 2015 External',emet[emet.split=='external'])]:
 for r in frame.itertuples():trade.append({'Cohort':name,'Threshold target':f"tau_{str(r.target_sensitivity).replace('0.','')}",'Frozen threshold':r.threshold,'TP':int(r.tp),'FN':int(r.fn),'TN':int(r.tn),'FP':int(r.fp),'Sensitivity':r.sensitivity,'FASR':r.fasr})
trade=pd.DataFrame(trade);trade.to_csv(res/'threshold_tradeoff.csv',index=False)
delta=pd.DataFrame([{'Comparison':'Test - Validation','Delta sensitivity':-0.0014495004400270073,'Sensitivity 95% CI':'-0.0674 to 0.0625','Delta FASR':-0.08481945467943997,'FASR 95% CI':'-0.1676 to 0.0010'},{'Comparison':'External - Validation','Delta sensitivity':ea['primary_by_stage']['external']['sensitivity']-ea['primary_by_stage']['validation']['sensitivity'],'Sensitivity 95% CI':'-0.0566 to 0.0661','Delta FASR':ea['primary_by_stage']['external']['fasr']-ea['primary_by_stage']['validation']['fasr'],'FASR 95% CI':'-0.2028 to -0.0383'},{'Comparison':'External - Test','Delta sensitivity':ea['primary_by_stage']['external']['sensitivity']-ea['primary_by_stage']['test']['sensitivity'],'Sensitivity 95% CI':'-0.0588 to 0.0762','Delta FASR':ea['primary_by_stage']['external']['fasr']-ea['primary_by_stage']['test']['fasr'],'FASR 95% CI':'-0.1223 to 0.0508'}]);delta.to_csv(res/'transport_deltas.csv',index=False)
# Prespecified complete-signal subsets; mapping selection ignores outcomes.
def subset(scores,mapping):
 x=scores.merge(mapping[['record','event','ecg_count','missing_ppg','missing_abp']],on=['record','event'],validate='one_to_one');return x[(x.ecg_count>=2)&(~x.missing_ppg)&(~x.missing_abp)]
def cm(d,t=.102635692):
 p=d.score>=t; y=d.y_true.astype(int);return {'N':len(d),'True VT':int(y.sum()),'False VT':int((1-y).sum()),'TP':int((p&y.eq(1)).sum()),'FN':int((~p&y.eq(1)).sum()),'TN':int((~p&y.eq(0)).sum()),'FP':int((p&y.eq(0)).sum()),'Sensitivity':float((p&y.eq(1)).sum()/y.sum()),'FASR':float((~p&y.eq(0)).sum()/(y.eq(0)).sum())}
# Validation raw-header mapping only; no waveform values/outcomes used to select modalities.
import sys;sys.path.insert(0,str(ROOT/'03_code/preprocessing'));import vtac_stage1b_pipeline_v3 as pp;import wfdb
vs=pd.read_csv(ROOT/'04_analysis/runs/stage1d_v1/validation_scores_with_seeds.csv');sp=pd.read_csv(ROOT/'02_data/raw/vtac-1.0/benchmark_data_split.csv');sp['split']=sp['split'].replace({'val':'validation'});vi=sp[sp.split=='validation'][['record','event']];mp=[]
for x in vi.itertuples():
 h=wfdb.rdheader(str(ROOT/'02_data/raw/vtac-1.0/waveforms'/x.record/x.event));m=pp.mapping(h.sig_name);mp.append({'record':x.record,'event':x.event,'ecg_count':m['ecg_count'],'missing_ppg':m['selected']['ppg'] is None,'missing_abp':m['selected']['abp'] is None})
vm=pd.DataFrame(mp);tm=pd.read_csv(ROOT/'02_data/manifests/test_channel_mapping_qa.csv');em=pd.read_csv(ROOT/'02_data/manifests/challenge2015_channel_mapping_qa.csv');cs=[]
for name,s,m in [('Validation',vs,vm),('VTaC Test',pd.read_csv(ROOT/'04_analysis/runs/stage1e_v1/test_scores_with_seeds.csv'),tm),('Challenge 2015 External',pd.read_csv(ROOT/'04_analysis/runs/stage1f_v1/external_scores_with_seeds.csv'),em)]:cs.append({'Cohort':name,**cm(subset(s,m))})
complete=pd.DataFrame(cs);complete.to_csv(res/'complete_signal_analysis.csv',index=False)
# Result locks
lock={'schema':'stage1-primary-results-lock-v1','status':'frozen_stage2_consolidation','primary_threshold':0.102635692,'secondary_thresholds':[0.065468185,0.022276472],'primary_results':rows,'transport_deltas':delta.to_dict('records'),'source_artifacts':{'stage1d_threshold_freeze':sh(ROOT/'09_reproducibility/manifests/stage1d_threshold_freeze.json'),'stage1e_test_evaluation':sh(ROOT/'09_reproducibility/manifests/stage1e_test_evaluation.json'),'stage1f_external_evaluation':sh(ROOT/'09_reproducibility/manifests/stage1f_external_evaluation.json')}}
write(ROOT/'00_protocol/frozen/STAGE1_PRIMARY_RESULTS_LOCK_v1.yaml','schema: stage1-primary-results-lock-v1\nstatus: frozen_stage2_consolidation\nprimary_threshold: 0.102635692\nsecondary_thresholds: [0.065468185, 0.022276472]\nvalidation: {N: 495, true_vt: 141, false_vt: 354, auroc: 0.9424, auroc_ci: [0.9146, 0.9653], tau95: {TP: 134, FN: 7, TN: 265, FP: 89, sensitivity: 0.9504, sensitivity_ci: [0.9051, 0.9862], fasr: 0.7486, fasr_ci: [0.6897, 0.8029]}}\ntest: {N: 482, true_vt: 137, false_vt: 345, auroc: 0.9175, auroc_ci: [0.8794, 0.9520], tau95: {TP: 130, FN: 7, TN: 229, FP: 116, sensitivity: 0.9489, sensitivity_ci: [0.8899, 0.9915], fasr: 0.6638, fasr_ci: [0.6006, 0.7240]}}\nexternal: {N: 341, true_vt: 89, false_vt: 252, auroc: 0.9411, auroc_ci: [0.9043, 0.9711], tau95: {TP: 85, FN: 4, TN: 158, FP: 94, sensitivity: 0.9551, sensitivity_ci: [0.9063, 0.9897], fasr: 0.6270, fasr_ci: [0.5667, 0.6867]}}\n')
write(ROOT/'00_protocol/frozen/STAGE1_PRIMARY_RESULTS_LOCK_v1.md','# Stage 1 Primary Results Lock v1\n\nThis lock preserves the Stage 1 primary results and fixed thresholds for manuscript consolidation. The central finding is: discrimination transported across datasets, and the frozen high-sensitivity operating point preserved approximately 95% sensitivity, but false-alarm suppression utility was attenuated. No equivalence, noninferiority, clinical-safety, or deployment-readiness claim is made.\n\n```json\n'+json.dumps(lock,indent=2)+'\n```\n')
# Figures
plt.rcParams.update({'font.size':10,'font.family':'DejaVu Sans','axes.spines.top':False,'axes.spines.right':False})
co=['Validation','VTaC Test','Challenge 2015\nExternal']; col=['#1976a3','#d36f2d','#4d8b5f']; sens=[.9504,.9489,.9551];slo=[.9051,.8899,.9063];shi=[.9862,.9915,.9897];fas=[.7486,.6638,.6270];flo=[.6897,.6006,.5667];fhi=[.8029,.7240,.6867];auc=[.9424,.9175,.9411]
# Fig1
fig,ax=plt.subplots(figsize=(11,5));ax.axis('off'); boxes=[(0.03,.60,'VTaC TRAIN\n4060 alarms'),(.25,.60,'TRAIN-only\n3-fold CV'),(.47,.60,'E*=31\n5 full-TRAIN models'),(.69,.60,'Frozen ensemble'),(.84,.60,'VTaC validation\n495 alarms'),(.84,.18,'tau_95 locked'),(.43,.18,'VTaC test\n482 alarms'),(.08,.18,'Challenge 2015\n341 VT alarms\n89 true / 252 false')]
for x,y,t in boxes:ax.add_patch(FancyBboxPatch((x,y),.13,.15,boxstyle='round,pad=0.02',fc='#f5f7f8',ec='#4a5962'));ax.text(x+.065,y+.075,t,ha='center',va='center',fontsize=9)
for a,b in [(.16,.32),(.38,.54),(.60,.76),(.82,.91)]:ax.annotate('',xy=(b,.675),xytext=(a,.675),arrowprops={'arrowstyle':'->'})
ax.annotate('',xy=(.905,.33),xytext=(.905,.60),arrowprops={'arrowstyle':'->'});ax.annotate('',xy=(.56,.33),xytext=(.84,.25),arrowprops={'arrowstyle':'->'});ax.annotate('',xy=(.21,.33),xytext=(.84,.25),arrowprops={'arrowstyle':'->'});ax.text(.5,.03,'No threshold retuning   |   No recalibration   |   No external adaptation',ha='center',weight='bold')
for ext in ['png','pdf']:fig.savefig(figs/f'Figure_1_study_design.{ext}',bbox_inches='tight',dpi=300)
plt.close(fig)
# Fig2
fig,axs=plt.subplots(1,2,figsize=(10,4.5));
for ax,vals,lo,hi,title in zip(axs,[sens,fas],[slo,flo],[shi,fhi],['A. Sensitivity at frozen tau_95','B. False-alarm suppression at frozen tau_95']):
 x=np.arange(3);ax.errorbar(x,vals,yerr=[np.array(vals)-np.array(lo),np.array(hi)-np.array(vals)],fmt='o',color='#1d2b36',capsize=4);ax.scatter(x,vals,s=70,c=col,zorder=3);ax.set_xticks(x,co);ax.set_ylim(0,1.05);ax.set_ylabel('Proportion');ax.set_title(title);ax.grid(axis='y',alpha=.25)
for ext in ['png','pdf']:fig.savefig(figs/f'Figure_2_primary_transport.{ext}',bbox_inches='tight',dpi=300)
plt.close(fig)
# Fig3
fig,ax=plt.subplots(figsize=(6.5,4.8));ax.scatter(auc,fas,s=90,c=col)
for x,y,n in zip(auc,fas,['Validation','Test','External']):ax.annotate(n,(x,y),xytext=(5,5),textcoords='offset points')
ax.set_xlim(0,1);ax.set_ylim(0,1);ax.set_xlabel('AUROC');ax.set_ylabel('FASR at frozen tau_95');ax.grid(alpha=.25);ax.set_title('Discrimination and fixed-threshold utility')
for ext in ['png','pdf']:fig.savefig(figs/f'Figure_3_discrimination_vs_utility.{ext}',bbox_inches='tight',dpi=300)
plt.close(fig)
# Fig4
fig,ax=plt.subplots(figsize=(6.8,5));markers=['o','s','^']
for name,g,c in zip(['Validation','VTaC Test','Challenge 2015 External'],[trade[trade.Cohort=='Validation'],trade[trade.Cohort=='VTaC Test'],trade[trade.Cohort=='Challenge 2015 External']],col):ax.plot(g.Sensitivity,g.FASR,marker='o',c=c,label=name);[ax.annotate(t,(x,y),xytext=(4,3),textcoords='offset points',fontsize=8) for x,y,t in zip(g.Sensitivity,g.FASR,g['Threshold target'])]
ax.set_xlim(.93,1.0);ax.set_ylim(0,0.82);ax.set_xlabel('Sensitivity');ax.set_ylabel('FASR');ax.legend(frameon=False);ax.grid(alpha=.25);ax.set_title('Prespecified safety-utility operating points')
for ext in ['png','pdf']:fig.savefig(figs/f'Figure_4_safety_utility_tradeoff.{ext}',bbox_inches='tight',dpi=300)
plt.close(fig)
# Reports
write(final/'Stage2_Statistical_Report.md','# Stage 2 Statistical Report\n\n## Primary interpretation\n\nDiscrimination transported across the three cohorts. At the frozen high-sensitivity operating point, sensitivity point estimates were approximately 95% in validation, internal test, and external evaluation, whereas false-alarm suppression decreased from 0.7486 in validation to 0.6638 in test and 0.6270 externally. The external-minus-validation FASR difference was -0.1216 (95% CI -0.2028 to -0.0383). These estimates describe transportability of the fixed operating point; no equivalence or noninferiority margin was prespecified.\n\n## Provenance\n\nVTaC release-native preprocessing retained all events despite raw signal-structure discrepancies: 42 TRAIN events had one raw ECG; raw modality prevalence differed from publication summaries; the public raw lead-selection construction was unavailable; and deterministic release-native reconstruction was used. Challenge 2015 used the checksum-verified v1.0.0 public release unchanged: 341 VT records, 89 true and 252 false. Historical publications report slightly different totals.\n')
write(final/'Manuscript_Results_Draft.md','# Results\n\n## Model development\n\nTraining-duration selection used record-grouped three-fold cross-validation within the 4,060-alarm VTaC training cohort. The mean held-out AUROC was maximal at epoch 31 (0.9201). Five prespecified models were then trained on the complete training cohort for 31 epochs and combined by arithmetic averaging of their sigmoid probabilities.\n\n## Validation threshold locking\n\nAmong 495 validation alarms, the ensemble AUROC was 0.9424 (95% CI, 0.9146-0.9653). The prespecified tau_95 threshold was 0.102635692. It yielded 134 true positives, 7 false negatives, 265 true negatives, and 89 false positives; sensitivity was 0.9504 (95% CI, 0.9051-0.9862) and FASR was 0.7486 (95% CI, 0.6897-0.8029).\n\n## Held-out internal test\n\nAt the unchanged threshold, the VTaC test cohort had an AUROC of 0.9175 (95% CI, 0.8794-0.9520). The operating point yielded 130 true positives, 7 false negatives, 229 true negatives, and 116 false positives. Sensitivity was 0.9489 (95% CI, 0.8899-0.9915) and FASR was 0.6638 (95% CI, 0.6006-0.7240).\n\n## External transportability\n\nIn the checksum-verified Challenge 2015 public VT cohort (341 alarms; 89 true and 252 false), AUROC was 0.9411 (95% CI, 0.9043-0.9711). The same threshold yielded 85 true positives, 4 false negatives, 158 true negatives, and 94 false positives. Sensitivity was 0.9551 (95% CI, 0.9063-0.9897) and FASR was 0.6270 (95% CI, 0.5667-0.6867). Relative to validation, the external sensitivity difference was 0.0047 (95% CI, -0.0566 to 0.0661), and the FASR difference was -0.1216 (95% CI, -0.2028 to -0.0383).\n\n## Discrimination and utility\n\nAUROC was similar in validation and external evaluation (0.9424 and 0.9411), while FASR at the frozen threshold was lower externally (0.6270 versus 0.7486).\n\n## Secondary threshold trade-off\n\nAt tau_97.5 and tau_99, sensitivity increased as the frozen threshold decreased, with lower FASR in each cohort. These secondary operating points were not used to replace tau_95.\n\n## Complete-signal sensitivity analysis\n\nThe prespecified complete-signal subset required at least two canonical ECG channels, PLETH, and ABP. The same model and tau_95 were used without retuning; results are reported in the secondary analysis table.\n')
# manifest last
files=[*res.glob('*'),*figs.glob('*'),*final.glob('*'),ROOT/'00_protocol/frozen/STAGE1_PRIMARY_RESULTS_LOCK_v1.yaml',ROOT/'00_protocol/frozen/STAGE1_PRIMARY_RESULTS_LOCK_v1.md'];manifest={'schema':'stage2-results-manifest-v1','files':[{ 'path':str(p.relative_to(ROOT)),'sha256':sh(p),'size_bytes':p.stat().st_size} for p in files]};write(man/'stage2_results_manifest.json',json.dumps(manifest,indent=2)+'\n');print(json.dumps({'primary_rows':len(primary),'trade_rows':len(trade),'complete_rows':len(complete),'files':len(files)},indent=2))
