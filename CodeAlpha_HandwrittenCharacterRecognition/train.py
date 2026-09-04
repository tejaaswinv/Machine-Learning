import argparse
import json
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import datasets, transforms
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from model import CharacterCNN

class TargetShift(Dataset):
    def __init__(self, base): self.base=base
    def __len__(self): return len(self.base)
    def __getitem__(self,i):
        x,y=self.base[i]; return x,y-1


def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--dataset",choices=["mnist","emnist_letters"],default="mnist"); ap.add_argument("--epochs",type=int,default=8); ap.add_argument("--batch_size",type=int,default=128); args=ap.parse_args()
    torch.manual_seed(42)
    tfm=transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.1307,),(0.3081,))])
    if args.dataset=="mnist":
        full=datasets.MNIST("data",train=True,download=True,transform=tfm); test=datasets.MNIST("data",train=False,download=True,transform=tfm); labels=[str(i) for i in range(10)]
    else:
        full=TargetShift(datasets.EMNIST("data",split="letters",train=True,download=True,transform=tfm)); test=TargetShift(datasets.EMNIST("data",split="letters",train=False,download=True,transform=tfm)); labels=[chr(ord('A')+i) for i in range(26)]
    n_val=int(.1*len(full)); train,val=random_split(full,[len(full)-n_val,n_val],generator=torch.Generator().manual_seed(42))
    train_loader=DataLoader(train,batch_size=args.batch_size,shuffle=True,num_workers=2); val_loader=DataLoader(val,batch_size=args.batch_size,num_workers=2); test_loader=DataLoader(test,batch_size=args.batch_size,num_workers=2)
    device=torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    model=CharacterCNN(len(labels)).to(device); opt=torch.optim.AdamW(model.parameters(),lr=1e-3); loss_fn=torch.nn.CrossEntropyLoss(); out=Path("artifacts"); out.mkdir(exist_ok=True); best=-1; hist=[]

    def eval_loader(loader, collect=False):
        model.eval(); correct=total=0; ys=[]; ps=[]
        with torch.no_grad():
            for x,y in loader:
                x,y=x.to(device),y.to(device); p=model(x).argmax(1); correct+=(p==y).sum().item(); total+=len(y)
                if collect: ys.extend(y.cpu().numpy()); ps.extend(p.cpu().numpy())
        return correct/total, np.array(ys), np.array(ps)

    for epoch in range(1,args.epochs+1):
        model.train(); running=0
        for x,y in train_loader:
            x,y=x.to(device),y.to(device); opt.zero_grad(); logits=model(x); loss=loss_fn(logits,y); loss.backward(); opt.step(); running+=loss.item()*len(y)
        val_acc,_,_=eval_loader(val_loader); hist.append({"epoch":epoch,"train_loss":running/len(train),"val_acc":val_acc}); print(hist[-1])
        if val_acc>best:
            best=val_acc; torch.save({"state":model.state_dict(),"labels":labels,"dataset":args.dataset},out/"best_model.pt")

    ckpt=torch.load(out/"best_model.pt",map_location=device); model.load_state_dict(ckpt["state"]); test_acc,y_true,y_pred=eval_loader(test_loader,True)
    (out/"metrics.json").write_text(json.dumps({"validation_best_accuracy":best,"test_accuracy":test_acc,"history":hist},indent=2))
    cm=confusion_matrix(y_true,y_pred); fig=plt.figure(figsize=(9,8)); plt.imshow(cm); plt.title(f"{args.dataset} Confusion Matrix"); plt.xlabel("Predicted"); plt.ylabel("Actual"); plt.tight_layout(); fig.savefig(out/"confusion_matrix.png",dpi=180); plt.close(fig)
    print(f"Test accuracy: {test_acc:.4f}")

if __name__=="__main__": main()
