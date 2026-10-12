"""Review saved masks as spatial partitions, separately from fixed-hole coverage."""
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import torch

from audit_attention_ablation_live import read, verify_suite
from audit_attention_step_block_sweep import safe_tap


def main():
    torch.set_num_threads(4)
    late=Path(sys.argv[1]).resolve();early=Path(sys.argv[2]).resolve()
    dest=early/"mask-shape-review";dest.mkdir(exist_ok=True)
    selection={
        "woman_front_strong_overlap":[(late,"off",3,18),(late,"on",3,18),(late,"on",3,26),(late,"on",3,23),
                                       (early,"on",3,7),(early,"on",3,10)],
        "park":[(early,"on",3,7),(early,"on",3,10),(late,"on",1,18),(late,"on",2,18),
                (late,"on",3,18),(late,"on",3,23),(late,"on",3,26)]}
    result={"scope":"Saved raw target/protected partitions and existing geometry statistics; no independent human labels or generated reference RGB",
            "cases":{}}
    colors=ListedColormap(["#242b34","#f4c14c","#4598dc","#b374c7"])
    for cid,items in selection.items():
        fig,axes=plt.subplots(3,len(items),figsize=(2.5*len(items),8),constrained_layout=True,squeeze=False)
        records=[]
        for column,(root,branch,step,block) in enumerate(items):
            folder=root/cid/"sweeps"/"mask"/branch;outer=read(folder/"manifest.json")
            item=outer["taps"][f"step_{step}.block_{block}"]
            path=safe_tap(folder,item);manifest,tensors=verify_suite(path)
            r=manifest["report"];status=r["algorithm"]["variants"]["multi_proto_bg"]
            row={"branch":branch,"step":step,"block":block,"source_manifest":str(path),"status":status["status"],
                 "candidate":status,"all_selected_file_tensor_hashes_verified":True}
            axes[0,column].set_title(f"{branch.upper()} s{step} b{block}\nraw target",fontsize=10)
            if status["status"]=="ok":
                target=tensors["multi_proto_bg.masks.target"][0];protected=tensors["multi_proto_bg.masks.protected"][0]
                labels=torch.zeros_like(target,dtype=torch.int64)
                labels[target.bool()]=1;labels[protected.bool()]=2;labels[tensors["multi_proto_bg.uncertain_mask"][0].bool()]=3
                for axis,mask in zip(axes[:,column],(target,protected,labels)):
                    axis.imshow(mask.numpy(),interpolation="nearest",vmin=0,vmax=3 if mask is labels else 1,
                                cmap=colors if mask is labels else "gray")
                axes[1,column].set_title("raw protected",fontsize=10)
                axes[2,column].set_title("yellow T / blue P / purple unknown",fontsize=8)
                row["geometry"]=status["diagnostics"]
            else:
                for axis in axes[:,column]:axis.text(.5,.5,"FAILED\n"+status["error_code"],ha="center",va="center",transform=axis.transAxes,fontsize=8)
            for axis in axes[:,column]:
                axis.set_xticks([0,16,32,48,63]);axis.set_yticks([0,16,32,48,63]);axis.tick_params(labelsize=7)
            records.append(row)
        fig.suptitle(cid+" | Spatial sharpness and person-role assignment require separate evaluation",fontsize=12)
        fig.savefig(dest/(cid+"-shape-comparison.png"),dpi=140);plt.close(fig)
        result["cases"][cid]=records
        print(cid,[(r["step"],r["block"],r["branch"],
                     r.get("geometry",{}).get("target",{}).get("component_count_8"),
                     r.get("geometry",{}).get("protected",{}).get("largest_component_fraction")) for r in records])
    (dest/"selected-mask-evidence.json").write_text(json.dumps(result,indent=2,ensure_ascii=False,allow_nan=False),encoding="utf-8")
    print(dest)


if __name__=="__main__":main()
