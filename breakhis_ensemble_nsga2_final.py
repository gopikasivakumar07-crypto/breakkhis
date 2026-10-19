import argparse
import copy
import json
import random
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from joblib import Parallel, delayed
from PIL import Image
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.core.problem import Problem
from pymoo.operators.crossover.pntx import TwoPointCrossover
from pymoo.operators.mutation.bitflip import BitflipMutation
from pymoo.optimize import minimize
from skimage.color import hed2rgb, rgb2hed
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC, LinearSVC
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms as T

IMNET_MEAN, IMNET_STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
OFFICIAL_COUNTS = {40: 1995, 100: 2081, 200: 2013, 400: 1820}
OFFICIAL_PATIENTS = 82


def build_metadata(root, mag, task, keep_all_mags):
    pat = re.compile(r"SOB_([BM])_([A-Z]+)-(.+)-(\d+)-(\d+)\.png$")
    rows = []
    for p in Path(root).rglob("*.png"):
        m = pat.match(p.name)
        if not m:
            continue
        cls, sub, slide, mg, _ = m.groups()
        rows.append(dict(path=str(p), cls=cls, sub=sub, patient=slide, mag=int(mg)))
    df = pd.DataFrame(rows)
    if df.empty:
        raise RuntimeError(f"No BreakHis images found under {root}")
    n_raw = len(df)
    df["name"] = df.path.map(lambda q: Path(q).name)
    df = df.drop_duplicates("name").reset_index(drop=True)
    counts = df.mag.value_counts().sort_index().to_dict()
    print(f"found {n_raw} image files, {len(df)} unique images after removing duplicate copies")
    print(f"images per magnification: {counts}")
    if counts == OFFICIAL_COUNTS:
        print("counts match the official BreaKHis counts (200x = 2013 images, total 7909)")
    else:
        print(f"WARNING: counts differ from the official BreaKHis counts {OFFICIAL_COUNTS}")
    if df.patient.nunique() != OFFICIAL_PATIENTS:
        print(f"note: {df.patient.nunique()} patient ids parsed from file names, official count is {OFFICIAL_PATIENTS}")
    if mag and not keep_all_mags:
        df = df[df.mag == mag].reset_index(drop=True)
    if task == "binary":
        df["y"] = (df.cls == "M").astype(int)
        names = ["benign", "malignant"]
    else:
        codes = sorted((df.cls + "_" + df["sub"]).unique())
        df["y"] = (df.cls + "_" + df["sub"]).map({c: i for i, c in enumerate(codes)})
        names = codes
    print(f"using {len(df)} images, {df.patient.nunique()} patients, classes={names}")
    return df, len(names)


def split_indices(df, n_splits, args):
    if args.split == "patient":
        sp = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=args.seed)
        return sp.split(df, df.y, df.patient)
    sp = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=args.seed)
    return sp.split(df, df.y)


class DS(Dataset):
    def __init__(self, paths, labels, tf):
        self.paths, self.labels, self.tf = list(paths), list(labels), tf

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        img = Image.open(self.paths[i]).convert("RGB")
        return self.tf(img), int(self.labels[i])


class HEDJitter:

    def __init__(self, alpha=0.1, beta=0.005, p=0.8):
        self.alpha, self.beta, self.p = alpha, beta, p

    def __call__(self, img):
        if random.random() > self.p:
            return img
        hed = rgb2hed(np.asarray(img, dtype=np.float64) / 255.0)
        a = np.random.uniform(1 - self.alpha, 1 + self.alpha, (1, 1, 3))
        b = np.random.uniform(-self.beta, self.beta, (1, 1, 3))
        rgb = np.clip(hed2rgb(hed * a + b), 0, 1)
        return Image.fromarray((rgb * 255).astype(np.uint8))


def img_size(args):
    return (args.img, int(round(args.img * args.aspect)))


def train_tf(args):
    size, a = img_size(args), args.aspect
    t = [T.RandomResizedCrop(size, scale=(0.6, 1.0), ratio=(a * 0.85, a * 1.15)),
         T.RandomHorizontalFlip(), T.RandomVerticalFlip()]
    if args.stain_aug:
        t.append(HEDJitter())
    t += [T.ColorJitter(0.2, 0.2, 0.2, 0.05), T.ToTensor(), T.Normalize(IMNET_MEAN, IMNET_STD)]
    return T.Compose(t)


def eval_tf(args):
    return T.Compose([T.Resize(img_size(args)), T.ToTensor(), T.Normalize(IMNET_MEAN, IMNET_STD)])


def finetune(name, df, n_cls, args, device):
    m = timm.create_model(name, pretrained=bool(args.pretrained), num_classes=0)
    if args.epochs == 0:
        return m.eval()
    model = nn.Sequential(m, nn.Linear(m.num_features, n_cls)).to(device)
    y = df.y.values
    cw = len(y) / (n_cls * np.maximum(np.bincount(y, minlength=n_cls), 1))
    cw = torch.tensor(cw, dtype=torch.float32, device=device)
    loader = DataLoader(DS(df.path, y, train_tf(args)), batch_size=args.bs, shuffle=True,
                        num_workers=args.workers, drop_last=True, pin_memory=device.type == "cuda")
    opt = torch.optim.AdamW([{"params": model[0].parameters(), "lr": args.lr},
                             {"params": model[1].parameters(), "lr": args.lr * 10}], weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs * len(loader))
    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    for _ in range(args.epochs):
        model.train()
        for x, t in loader:
            x, t = x.to(device, non_blocking=True), t.to(device, non_blocking=True)
            with torch.autocast(device_type=device.type, enabled=use_amp):
                loss = F.cross_entropy(model(x), t, weight=cw, label_smoothing=0.05)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
    return model[0].eval()


@torch.no_grad()
def extract(backbone, df, args, device):
    backbone = backbone.to(device).eval()
    loader = DataLoader(DS(df.path, df.y, eval_tf(args)), batch_size=args.bs * 2,
                        shuffle=False, num_workers=args.workers)
    use_amp = device.type == "cuda"
    out = []
    for x, _ in loader:
        x = x.to(device)
        with torch.autocast(device_type=device.type, enabled=use_amp):
            f = backbone(x).float()
            if args.tta:
                f = (f + backbone(torch.flip(x, [3])).float() + backbone(torch.flip(x, [2])).float()) / 3
        out.append(f.cpu())
    return torch.cat(out).numpy()


class Fusion(nn.Module):

    def __init__(self, dims, d=256, mode="concat", n_cls=2, drop=0.3):
        super().__init__()
        self.proj = nn.ModuleList(
            nn.Sequential(nn.Linear(k, d), nn.BatchNorm1d(d), nn.ReLU()) for k in dims)
        self.mode = mode
        self.w = nn.Parameter(torch.zeros(len(dims)))
        self.drop = nn.Dropout(drop)
        self.head = nn.Linear(d * len(dims) if mode == "concat" else d, n_cls)

    def fuse(self, xs):
        z = [p(x) for p, x in zip(self.proj, xs)]
        if self.mode == "concat":
            return torch.cat(z, 1)
        w = torch.softmax(self.w, 0)
        return sum(wi * zi for wi, zi in zip(w, z))

    def forward(self, xs):
        return self.head(self.drop(self.fuse(xs)))


def train_fusion(Xtr, ytr, Xva, yva, mode, n_cls, args, device):
    torch.manual_seed(args.seed)
    model = Fusion([x.shape[1] for x in Xtr], args.proj_dim, mode, n_cls).to(device)
    tt = [torch.tensor(x, dtype=torch.float32, device=device) for x in Xtr]
    tv = [torch.tensor(x, dtype=torch.float32, device=device) for x in Xva]
    y_t = torch.tensor(ytr, device=device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-3)
    n, best_acc, best_state = len(ytr), -1.0, None
    for _ in range(args.fusion_epochs):
        model.train()
        perm = torch.randperm(n, device=device)
        for i in range(0, n, 128):
            idx = perm[i:i + 128]
            if len(idx) < 2:
                continue
            loss = F.cross_entropy(model([x[idx] for x in tt]), y_t[idx], label_smoothing=0.1)
            opt.zero_grad()
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            acc = (model(tv).argmax(1).cpu().numpy() == yva).mean()
        if acc > best_acc:
            best_acc, best_state = acc, copy.deepcopy(model.state_dict())
    model.load_state_dict(best_state)
    return model.eval()


@torch.no_grad()
def fuse_np(model, X_list, device):
    return model.fuse([torch.tensor(x, dtype=torch.float32, device=device) for x in X_list]).cpu().numpy()


def _fit_eval(mask, Xtr, ytr, Xva, yva, C, fitness):
    mask = np.asarray(mask, dtype=bool)
    k = int(mask.sum())
    if k == 0:
        return [1.0, 1.0]
    if fitness == "linear":
        clf = LinearSVC(C=C, dual=False, max_iter=5000)
    else:
        clf = SVC(C=C, kernel="rbf", gamma="scale")
    clf.fit(Xtr[:, mask], ytr)
    err = float(np.mean(clf.predict(Xva[:, mask]) != yva))
    return [err, k / len(mask)]


class FSProblem(Problem):

    def __init__(self, Xtr, ytr, Xva, yva, C, fitness, n_jobs):
        super().__init__(n_var=Xtr.shape[1], n_obj=2, xl=0, xu=1, vtype=bool)
        self.d, self.C, self.fitness, self.n_jobs = (Xtr, ytr, Xva, yva), C, fitness, n_jobs

    def _evaluate(self, X, out, *args, **kwargs):
        Xtr, ytr, Xva, yva = self.d
        res = Parallel(n_jobs=self.n_jobs)(
            delayed(_fit_eval)(m, Xtr, ytr, Xva, yva, self.C, self.fitness) for m in X)
        out["F"] = np.array(res)


def init_population(n_var, pop, rng):
    P = np.zeros((pop, n_var), dtype=bool)
    P[0] = True
    dens = [0.1, 0.2, 0.3, 0.5]
    for i in range(1, pop):
        P[i] = rng.rand(n_var) < dens[i % len(dens)]
        if not P[i].any():
            P[i, rng.randint(n_var)] = True
    return P


def run_nsga(Ztr, ytr, Zva, yva, args):
    rng = np.random.RandomState(args.seed)
    if len(ytr) > args.fit_max_train:
        sel = rng.choice(len(ytr), args.fit_max_train, replace=False)
        Ztr, ytr = Ztr[sel], ytr[sel]
    C = args.lin_C if args.fitness == "linear" else args.svm_C
    algo = NSGA2(pop_size=args.pop, sampling=init_population(Ztr.shape[1], args.pop, rng),
                 crossover=TwoPointCrossover(), mutation=BitflipMutation(), eliminate_duplicates=True)
    res = minimize(FSProblem(Ztr, ytr, Zva, yva, C, args.fitness, args.n_jobs), algo,
                   ("n_gen", args.gen), seed=args.seed, verbose=False)
    X, Fv = np.atleast_2d(res.X).astype(bool), np.atleast_2d(res.F)
    order = np.lexsort((Fv[:, 1], Fv[:, 0]))
    return X, Fv, X[order[0]]


def svm_predict(Xtr, ytr, Xva, yva, Xte, Cs=(1, 10, 100)):
    sc = StandardScaler().fit(Xtr)
    a, b, c = sc.transform(Xtr), sc.transform(Xva), sc.transform(Xte)
    bestC = max(Cs, key=lambda C: SVC(C=C, gamma="scale").fit(a, ytr).score(b, yva))
    clf = SVC(C=bestC, gamma="scale").fit(np.vstack([a, b]), np.concatenate([ytr, yva]))
    return clf.predict(c)


def metrics(y_true, y_pred, patients):
    d = pd.DataFrame({"p": patients, "t": y_true, "y": y_pred})
    pv = d.groupby("p").agg(t=("t", "first"), y=("y", lambda s: s.mode().iloc[0]))
    wrong = sorted(pv.index[pv.t != pv.y])
    return dict(img_acc=accuracy_score(y_true, y_pred),
                img_f1=f1_score(y_true, y_pred, average="macro"),
                patient_acc=float((pv.t == pv.y).mean()),
                misclassified_patients="|".join(wrong), n_misclassified_patients=len(wrong),
                n_test_images=len(y_true), n_test_patients=len(pv))


def run_fold(fold, tr_idx, te_idx, df, args, n_cls, device):
    t0 = time.time()

    def restrict(d):
        if args.train_all_mags and args.mag:
            d = d[d.mag == args.mag]
        return d.reset_index(drop=True)

    tr_all = df.iloc[tr_idx].reset_index(drop=True)
    itr, iva = next(split_indices(tr_all, args.inner_splits, args))
    tr = tr_all.iloc[itr].reset_index(drop=True)
    va, te = restrict(tr_all.iloc[iva]), restrict(df.iloc[te_idx])
    if args.split == "patient":
        assert not set(tr.patient) & set(va.patient) and not set(tr_all.patient) & set(te.patient)
    print(f"  [fold {fold}] train={len(tr)} val={len(va)} test={len(te)} images")

    feats, val_f1 = {}, {}
    for name in args.candidates:
        bb = finetune(name, tr, n_cls, args, device)
        Ftr, Fva, Fte = (extract(bb, d, args, device) for d in (tr, va, te))
        sc = StandardScaler().fit(Ftr)
        clf = SVC(C=10, gamma="scale").fit(sc.transform(Ftr), tr.y)
        val_f1[name] = f1_score(va.y, clf.predict(sc.transform(Fva)), average="macro")
        feats[name] = (Ftr, Fva, Fte)
        del bb
        torch.cuda.empty_cache()
        print(f"  [fold {fold}] {name:24s} dim={Ftr.shape[1]:5d} val-F1={val_f1[name]:.4f}")
    top = sorted(val_f1, key=val_f1.get, reverse=True)[:3]
    print(f"  [fold {fold}] top-3 CNNs: {top}")

    rows = []

    def log(method, pred, extra=None):
        r = dict(fold=fold, method=method, **metrics(te.y.values, pred, te.patient.values))
        r.update(extra or {})
        rows.append(r)
        print(f"  [fold {fold}] {method:34s} img-acc={r['img_acc']:.4f} patient-acc={r['patient_acc']:.4f}"
              f" wrong-patients={r['misclassified_patients'] or '-'}")

    Ftr, Fva, Fte = feats[top[0]]
    log("single-best-CNN+SVM", svm_predict(Ftr, tr.y.values, Fva, va.y.values, Fte),
        dict(single_model=top[0]))

    scalers = {n: StandardScaler().fit(feats[n][0]) for n in top}
    S = [[scalers[n].transform(feats[n][i]) for n in top] for i in range(3)]
    masks = {}
    for mode in args.modes:
        fm = train_fusion(S[0], tr.y.values, S[1], va.y.values, mode, n_cls, args, device)
        Ztr, Zva, Zte = (fuse_np(fm, S[i], device) for i in range(3))
        log(f"ens-{mode}+SVM(all {Ztr.shape[1]})",
            svm_predict(Ztr, tr.y.values, Zva, va.y.values, Zte))

        zs = StandardScaler().fit(Ztr)
        _, pareto_F, mask = run_nsga(zs.transform(Ztr), tr.y.values, zs.transform(Zva), va.y.values, args)
        pred = svm_predict(Ztr[:, mask], tr.y.values, Zva[:, mask], va.y.values, Zte[:, mask])
        log(f"ens-{mode}+NSGA2+SVM", pred,
            dict(n_feat=int(mask.sum()), n_total=int(len(mask)), pareto_size=int(len(pareto_F)),
                 top3="|".join(top)))
        masks[mode] = mask.astype(int).tolist()
    print(f"  [fold {fold}] done in {(time.time() - t0) / 60:.1f} min")
    return rows, masks


def summarize(res):
    res = res.copy()
    res["method"] = res.method.str.replace(r"\(all \d+\)", "(all)", regex=True)
    res["_correct"] = res.img_acc * res.n_test_images
    res["_pcorrect"] = res.patient_acc * res.n_test_patients
    g = res.groupby("method")
    summ = g[["img_acc", "img_f1", "patient_acc"]].agg(["mean", "std"])
    summ.columns = ["_".join(c) for c in summ.columns]
    summ["img_acc_pooled"] = g._correct.sum() / g.n_test_images.sum()
    summ["patient_acc_pooled"] = g._pcorrect.sum() / g.n_test_patients.sum()
    return summ.round(4)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--mag", type=int, default=0)
    ap.add_argument("--train_all_mags", action="store_true")
    ap.add_argument("--task", choices=["binary", "multi"], default="binary")
    ap.add_argument("--candidates", nargs="+", default=["densenet121", "efficientnet_b0", "resnet50"])
    ap.add_argument("--modes", nargs="+", default=["concat", "weighted"])
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--split", choices=["patient", "image"], default="patient")
    ap.add_argument("--inner_splits", type=int, default=3)
    ap.add_argument("--img", type=int, default=320)
    ap.add_argument("--aspect", type=float, default=1.5)
    ap.add_argument("--stain_aug", type=int, default=1)
    ap.add_argument("--tta", type=int, default=1)
    ap.add_argument("--pretrained", type=int, default=1)
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--proj_dim", type=int, default=256)
    ap.add_argument("--fusion_epochs", type=int, default=60)
    ap.add_argument("--pop", type=int, default=100)
    ap.add_argument("--gen", type=int, default=40)
    ap.add_argument("--fitness", choices=["rbf", "linear"], default="rbf")
    ap.add_argument("--svm_C", type=float, default=10)
    ap.add_argument("--lin_C", type=float, default=0.1)
    ap.add_argument("--fit_max_train", type=int, default=2000)
    ap.add_argument("--n_jobs", type=int, default=-1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="results_breakhis_v2")
    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.benchmark = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    json.dump(vars(args), open(out / "args.json", "w"), indent=1)

    df, n_cls = build_metadata(args.root, args.mag, args.task, args.train_all_mags)
    if args.split == "image":
        print("WARNING: --split image is a random image-level split. Images of the same patient appear in both "
              "train and test, so accuracy is inflated and not comparable with patient-independent results. "
              "Report it only next to the patient-level result and label it clearly.")
    all_rows, all_masks = [], {}
    for fold, (tr_idx, te_idx) in enumerate(split_indices(df, args.folds, args)):
        print(f"=== outer fold {fold}: train={len(tr_idx)} test={len(te_idx)} images (before mag restriction) ===")
        rows, masks = run_fold(fold, tr_idx, te_idx, df, args, n_cls, device)
        all_rows += rows
        all_masks[fold] = masks
        pd.DataFrame(all_rows).to_csv(out / "fold_results.csv", index=False)
        json.dump(all_masks, open(out / "selected_masks.json", "w"))

    summ = summarize(pd.DataFrame(all_rows))
    print("\n================ SUMMARY (mean/std over outer folds; pooled = over all test images/patients) "
          "================")
    print(summ.to_string())
    summ.to_csv(out / "summary.csv")


if __name__ == "__main__":
    main()
