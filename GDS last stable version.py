import tkinter as tk
from tkinter import ttk, messagebox, filedialog
from collections import deque, defaultdict, Counter
from math import ceil, sqrt, floor
from typing import Optional, Dict, Tuple, List, Set
import time, json, random, os, itertools, re
import numpy as np
import warnings
from copy import deepcopy
from ui.overlays import ColumnGridOverlay, LegendPopover

BED_RULES_FILE = os.path.join(os.path.dirname(__file__), "rules.bedroom.json")
BATH_RULES_FILE = os.path.join(os.path.dirname(__file__), "rules.bathroom.json")
LIV_RULES_FILE = os.path.join(os.path.dirname(__file__), "rules.livingroom.json")
KITCH_RULES_FILE = os.path.join(os.path.dirname(__file__), "rules.kitchen.json")

def load_rules(path: str) -> Dict:
    """
    Load rule configuration from ``path``. If the file is missing or invalid
    an empty dictionary is returned so that built-in defaults remain in effect.
    """
    try:
        with open(path, "r") as f:
            text = f.read()
        text = re.sub(r"//.*?$|/\*.*?\*/", "", text, flags=re.MULTILINE | re.DOTALL)
        return json.loads(text)
    except Exception:
        return {}

RULES = load_rules(BED_RULES_FILE)
BATH_RULES = load_rules(BATH_RULES_FILE)
LIV_RULES = load_rules(LIV_RULES_FILE)
KITCH_RULES = load_rules(KITCH_RULES_FILE)

"""
VASTU – Sketch + Generate (Bedroom) – ALL-IN-ONE ADVANCED – FINAL (Aug-2025)

What’s new in this build
- Clearances: zero-gap from furniture/element edges (no empty-cell gap).
- Bold visibility: walls/door/windows stroke ≈3× thicker.
- Human simulation: visible “block” that sprints the full scribble path (every empty cell + access targets).
  Logs collisions/unreachables; animates in hyperspeed.
- Access faces:
  • BED: 3 faces accessible (all except wall face)
  • BST/WRD/DRS/DESK/TVU: face opposite the wall accessible
- Learning:
  • Rehydrate from all logs; per-feature & overall feedback; sim outcomes.
  • Deep MLP preference model + TransformerPreference (self-attention over feature tokens) – NumPy only.
  • Q-learning for rapid policy nudges (state from sim metrics → actions adjust solver weights).
- Furniture size restrictions vs room area.
- Batch sensitivities applied before Generate.
 - Window controls: first window defaults to the top wall in both bedroom and bathroom.
- Strategic data capture in feedback/simulation JSONL log.

Files (in working dir)
- solver_weights.json
- solver_feedback.jsonl
- solver_simulations.jsonl
- solver_nn.npz
- solver_transformer.npz
- solver_rl.json
"""

# -----------------------
# Constants / units
# -----------------------

IN_TO_M = 0.0254
FT_TO_M = 0.3048
# Use a single cell size for bedroom, bathroom, and living room grids so each
# cell represents the same physical dimension regardless of which plan
# it belongs to.  Bedroom rules take precedence, falling back to the
# bathroom then living room rules (or a 0.25 m default) if unspecified.
BED_CELL_M = RULES.get("units", {}).get("CELL_M")
BATH_CELL_M = BATH_RULES.get("units", {}).get("CELL_M")
LIV_CELL_M = LIV_RULES.get("units", {}).get("CELL_M")
KITCH_CELL_M = KITCH_RULES.get("units", {}).get("CELL_M")
CELL_M = BED_CELL_M or BATH_CELL_M or LIV_CELL_M or KITCH_CELL_M or 0.25
BATH_RULES.setdefault("units", {})["CELL_M"] = CELL_M
LIV_RULES.setdefault("units", {})["CELL_M"] = CELL_M
KITCH_RULES.setdefault("units", {})["CELL_M"] = CELL_M
PATH_WIDTH_CELLS = RULES.get("solver", {}).get("PATH_WIDTH_CELLS", 2)
LEARNING_RATE = RULES.get("learning", {}).get("LEARNING_RATE", 0.06)
WINDOW_CLEARANCE_M = 0.40

SIM_FILE = 'solver_simulations.jsonl'
WEIGHT_FILE='solver_weights.json'
FEEDBACK_FILE='solver_feedback.jsonl'
NN_FILE = 'solver_nn.npz'
TRF_FILE = 'solver_transformer.npz'
RL_FILE = 'solver_rl.json'

# Supervised/Unsupervised ensemble files
SUP_FILE = 'solver_supervised.npz'   # classification + regression head
KM_FILE  = 'solver_kmeans.npz'       # unsupervised clusters
ENS_FILE = 'solver_ensemble.npz'     # light wrapper metadata


# Additional model files (heavy-weight ensemble)
AE_FILE  = 'solver_autoencoder.npz'
CNN_FILE = 'solver_cnn.npz'
RNN_FILE = 'solver_rnn.npz'
GAN_FILE = 'solver_gan.npz'

# --- ultra-light cross-process file lock (works well for many local users) ---
def _acquire_lock(path: str, timeout: float = 1.5, poll: float = 0.05):
    """
    Create <path>.lock atomically. Returns lock file path or None on timeout.
    Safe on macOS/Linux; avoids concurrent writes when multiple app instances run.
    """
    lock = path + '.lock'
    t0 = time.time()
    while True:
        try:
            fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.write(fd, str(os.getpid()).encode())
            os.close(fd)
            return lock
        except FileExistsError:
            if time.time() - t0 > timeout:
                return None
            time.sleep(poll)

def _release_lock(lockpath: Optional[str]):
    if not lockpath: return
    try: os.unlink(lockpath)
    except Exception: pass


AREA_UNIT_TO_M2 = {
    "m²": 1.0,
    "ft²": (FT_TO_M ** 2),
    "yd²": (0.9144 ** 2),
    "cm²": 1e-4,
    "mm²": 1e-6,
    "acre": 4046.8564224,
    "hectare": 10000.0,
}
LENGTH_UNIT_LABELS = ["m", "ft"]

# -----------------------
# Catalog (book sizes only)
# -----------------------

DEFAULT_BEDROOM_BOOK = {
    "BEDS": {
        "SINGLE": {"w": 3 * FT_TO_M, "d": (6 + 6/12) * FT_TO_M},
        "TWIN":   {"w": (3 + 3/12) * FT_TO_M, "d": (6 + 6/12) * FT_TO_M},
        "THREE_Q_SMALL": {"w": 4 * FT_TO_M, "d": (6 + 6/12) * FT_TO_M},
        "DOUBLE": {"w": (4 + 6/12) * FT_TO_M, "d": (6 + 6/12) * FT_TO_M},
    },
    "NIGHT_TABLE": {
        "BST_18": {"w": (1 + 6/12) * FT_TO_M, "d": (1 + 6/12) * FT_TO_M},
        "BST_24": {"w": 2 * FT_TO_M, "d": (1 + 6/12) * FT_TO_M},
    },
    "WARDROBE": {
        "WRD_H_180": {"w": 1.80, "d": 0.60, "front_rec": 0.60, "front_min": 0.50, "type": "hinged"},
        "WRD_S_210": {"w": 2.10, "d": 0.65, "front_rec": 0.80, "front_min": 0.70, "type": "sliding"},
        "WRD_S_180": {"w": 1.80, "d": 0.65, "front_rec": 0.80, "front_min": 0.70, "type": "sliding"},
        "WRD_H_150": {"w": 1.50, "d": 0.60, "front_rec": 0.60, "front_min": 0.50, "type": "hinged"},
    },
    "DRESSER": {
        "CHEST_SM": {"w": 3 * FT_TO_M, "d": (1 + 6/12) * FT_TO_M, "front_rec": 0.90, "front_min": 0.75},
        "DRS_4FT": {"w": 4 * FT_TO_M, "d": 2 * FT_TO_M, "front_rec": 0.90, "front_min": 0.75},
    },
    "DESK": {
        "DESK_120": {"w": 1.20, "d": 0.60, "front_rec": 0.90, "front_min": 0.75},
        "DESK_100": {"w": 1.00, "d": 0.55, "front_rec": 0.90, "front_min": 0.75},
    },
    "TVU": {"TVU_120": {"w": 1.20, "d": 0.45}},
    "CLEAR": {
        "side_rec": (3 + 6/12) * FT_TO_M,
        "side_min": 2 * FT_TO_M,
        "foot_rec": 3 * FT_TO_M,
        "foot_min": (1 + 6/12) * FT_TO_M,
        "unit_gap": 3 * IN_TO_M,
    },
}

BEDROOM_BOOK = RULES.get("bedroom_book") or RULES.get("BEDROOM_BOOK") or DEFAULT_BEDROOM_BOOK


def kitchen_book_from_rules(rules: dict) -> dict:
    """Construct a small kitchen catalog from a ``rules`` dictionary.

    Only a subset of modules is currently included.  Width, depth and
    clearance values are pulled directly from ``rules['furniture']`` so that
    any custom rule file can override the defaults without editing the code.
    """

    furn = rules.get("furniture", {})

    def _variants(module: str) -> dict:
        mod = furn.get(module, {})
        variants = mod.get("variants", {})
        out = {}
        for name, dims in variants.items():
            out[name.upper()] = {k: dims.get(k) for k in ("w", "d") if k in dims}
        return out

    clear = furn.get("CLEAR", {})
    return {
        "SINK": _variants("SINK"),
        "COOK": _variants("COOK"),
        "REF": _variants("REF"),
        "CLEAR": {
            "side_rec": clear.get("side_rec"),
            "side_min": clear.get("side_min"),
            "front_rec": clear.get("front_rec"),
            "front_min": clear.get("front_min"),
            "unit_gap": clear.get("unit_gap_m"),
        },
    }


DEFAULT_KITCHEN_BOOK = kitchen_book_from_rules(KITCH_RULES)
KITCHEN_BOOK = KITCH_RULES.get("kitchen_book") or DEFAULT_KITCHEN_BOOK

# -----------------------
# Theme
# -----------------------

def apply_modern_theme(root: tk.Misc) -> None:
    style = ttk.Style(root)
    try:
        if 'aqua' in style.theme_names():
            style.theme_use('aqua')
        else:
            style.theme_use('clam')
    except Exception:
        pass
    style.configure('Primary.TButton', padding=(14, 7), font=('SF Pro Text', 12, 'bold'))
    style.configure('Secondary.TButton', padding=(14, 7), font=('SF Pro Text', 12))
    style.configure('TLabel', font=('SF Pro Text', 12))
    style.configure('TEntry', font=('SF Pro Text', 12))
    style.configure('TCombobox', font=('SF Pro Text', 12))
    style.configure('TScale', troughcolor='#ddd')

def clamp(v, lo, hi):
    return lo if v < lo else hi if v > hi else v

# -----------------------
# Weights / learning
# -----------------------

def load_weights()->Dict[str,float]:
    try:
        with open(WEIGHT_FILE,'r') as f: return json.load(f)
    except Exception:
        return {
            'bst_pair': 3.0,
            'has_wr': 0.9,
            'has_dr': 0.7,
            'privacy': 0.7,
            'symmetry': 0.5,
            'paths_ok': 1.2,
            'use_rec_clear': 0.5,
            'bed_not_bottom': 0.2,
            'coverage': 1.0,
            'reach_windows': 1.0,
            'door_align': 0.6,
            'longedge_wall': 0.4,
            'near_window_desk': 0.3,
            'adjacency': 0.6,
            'short_time': 0.0,
            'sim_len': -30.0
        }

def save_weights(w:Dict[str,float]):
    lock = _acquire_lock(WEIGHT_FILE)
    try:
        open(WEIGHT_FILE,'w').write(json.dumps(w))
    except Exception:
        pass
    finally:
        _release_lock(lock)

# -----------------------
# Robust file I/O (atomic + cross-process locks)
# -----------------------
try:
    import fcntl  # POSIX lock
    def _lock(f):  fcntl.flock(f.fileno(), fcntl.LOCK_EX)
    def _unlock(f): fcntl.flock(f.fileno(), fcntl.LOCK_UN)
except Exception:
    def _lock(f):  ...
    def _unlock(f): ...

def append_jsonl_locked(path:str, obj:dict)->None:
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    with open(path, 'a') as f:
        try: _lock(f)
        except Exception: pass
        try:
            f.write(json.dumps(obj) + "\n")
        finally:
            try: _unlock(f)
            except Exception: pass

def np_savez_atomic(path:str, **named)->None:
    tmp = path + ".tmp"
    np.savez(tmp, **named)
    try: os.replace(tmp, path)
    except Exception:
        try: os.remove(tmp)
        except Exception: pass


# Safe global dot product between weight dict and feature dict
def dot_score(w: Dict[str, float], f: Dict[str, float]) -> float:
    try:
        return float(sum(w.get(k, 0.0) * float(v) for k, v in f.items()))
    except Exception:
        # Be robust to any odd types in f
        total = 0.0
        for k, v in f.items():
            try:
                total += w.get(k, 0.0) * float(v)
            except Exception:
                continue
        return float(total)


def update_weights(weights:Dict[str,float], features:Dict[str,float], sign:int, lr:float=LEARNING_RATE)->Dict[str,float]:
    for k, v in features.items():
        weights[k] = weights.get(k, 0.0) + lr * sign * v
    save_weights(weights)
    return weights


# -----------------------
# Deep models (NumPy): MLP + Transformer
# -----------------------

class NeuralPreference:
    """Deeper MLP with two hidden layers for non-linear preference modeling."""
    def __init__(self, input_keys: List[str], h1=24, h2=16, lr=0.01, seed=42):
        self.keys = input_keys
        self.lr = lr
        rng = np.random.default_rng(seed)
        self.W1 = rng.normal(0, 0.2, (len(self.keys), h1))
        self.b1 = np.zeros((h1,))
        self.W2 = rng.normal(0, 0.2, (h1, h2))
        self.b2 = np.zeros((h2,))
        self.W3 = rng.normal(0, 0.2, (h2, 1))
        self.b3 = np.zeros((1,))

    def _vec(self, feat: Dict[str,float]):
        return np.array([float(feat.get(k,0.0)) for k in self.keys], dtype=np.float32)

    def predict(self, feat: Dict[str,float]) -> float:
        x = self._vec(feat)
        h1 = np.tanh(x @ self.W1 + self.b1)
        h2 = np.tanh(h1 @ self.W2 + self.b2)
        y_arr = (h2 @ self.W3 + self.b3).reshape(-1)
        return float(y_arr.item())


    def fit_batch(self, X: List[Dict[str,float]], y: List[float], epochs=8):
        if not X: return
        Xv = np.stack([self._vec(f) for f in X])
        yv = np.array(y, dtype=np.float32).reshape(-1,1)
        for _ in range(epochs):
            h1 = np.tanh(Xv @ self.W1 + self.b1)
            h2 = np.tanh(h1 @ self.W2 + self.b2)
            yhat = h2 @ self.W3 + self.b3
            err = yhat - yv
            # backprop
            dW3 = h2.T @ err / len(Xv)
            db3 = np.mean(err, axis=0)
            dh2 = err @ self.W3.T
            dh2_raw = (1 - h2**2) * dh2
            dW2 = h1.T @ dh2_raw / len(Xv)
            db2 = np.mean(dh2_raw, axis=0)
            dh1 = dh2_raw @ self.W2.T
            dh1_raw = (1 - h1**2) * dh1
            dW1 = Xv.T @ dh1_raw / len(Xv)
            db1 = np.mean(dh1_raw, axis=0)
            # update
            self.W3 -= self.lr * dW3; self.b3 -= self.lr * db3
            self.W2 -= self.lr * dW2; self.b2 -= self.lr * db2
            self.W1 -= self.lr * dW1; self.b1 -= self.lr * db1

    def save(self, path):
        np.savez(path, W1=self.W1, b1=self.b1, W2=self.W2, b2=self.b2, W3=self.W3, b3=self.b3, keys=np.array(self.keys, dtype=object))

    @staticmethod
    def load(path):
        """
        Safe loader:
        - Returns None if file missing, old schema, or unreadable.
        - Renames corrupt file to *.bad to avoid repeated crashes.
        """
        try:
            if not os.path.exists(path):
                return None
            d = np.load(path, allow_pickle=True)
            required = {'keys', 'W1', 'b1', 'W2', 'b2', 'W3', 'b3'}
            if not required.issubset(set(d.files)):
                # old or partial file -> ignore and quarantine
                try: os.replace(path, path + ".bad")
                except Exception: pass
                return None
            keys = list(d['keys'])
            mdl = NeuralPreference(keys)
            mdl.W1, mdl.b1 = d['W1'], d['b1']
            mdl.W2, mdl.b2 = d['W2'], d['b2']
            mdl.W3, mdl.b3 = d['W3'], d['b3']
            return mdl
        except Exception:
            # unreadable -> quarantine
            try: os.replace(path, path + ".bad")
            except Exception: pass
            return None


class TransformerPreference:
    """
    Minimal self-attention over feature tokens to capture correlations.
    Sequence = sorted feature keys; token embedding = value * learned key vector.
    Two encoder layers -> pooled vector -> linear score.
    """
    def __init__(self, keys: List[str], d_model=32, n_heads=4, lr=0.01, seed=7):
        self.keys = keys
        self.lr = lr
        self.d = d_model
        self.h = n_heads
        rng = np.random.default_rng(seed)
        # token embeddings (for keys)
        self.E = rng.normal(0, 0.2, (len(keys), d_model))
        # two layers of MHSA + FFN
        self.Wq1 = rng.normal(0, 0.2, (d_model, d_model))
        self.Wk1 = rng.normal(0, 0.2, (d_model, d_model))
        self.Wv1 = rng.normal(0, 0.2, (d_model, d_model))
        self.Wo1 = rng.normal(0, 0.2, (d_model, d_model))
        self.F1_1 = rng.normal(0, 0.2, (d_model, d_model*2))
        self.F1_2 = rng.normal(0, 0.2, (d_model*2, d_model))

        self.Wq2 = rng.normal(0, 0.2, (d_model, d_model))
        self.Wk2 = rng.normal(0, 0.2, (d_model, d_model))
        self.Wv2 = rng.normal(0, 0.2, (d_model, d_model))
        self.Wo2 = rng.normal(0, 0.2, (d_model, d_model))
        self.F2_1 = rng.normal(0, 0.2, (d_model, d_model*2))
        self.F2_2 = rng.normal(0, 0.2, (d_model*2, d_model))

        self.W_out = rng.normal(0, 0.2, (d_model, 1))
        self.b_out = 0.0

    def _seq(self, feat: Dict[str,float]):
        # sequence of token vectors: value * embedding
        vals = np.array([float(feat.get(k,0.0)) for k in self.keys]).reshape(-1,1)
        X = vals * self.E
        return X  # (T,d)

    def _mhsa(self, X, Wq, Wk, Wv, Wo):
        # simple single-head per attention head and concat
        T, d = X.shape
        H = self.h
        dh = d // H
        Q = X @ Wq; K = X @ Wk; V = X @ Wv
        heads=[]
        for i in range(H):
            q = Q[:, i*dh:(i+1)*dh]
            k = K[:, i*dh:(i+1)*dh]
            v = V[:, i*dh:(i+1)*dh]
            att = (q @ k.T) / np.sqrt(dh)
            att = np.exp(att - att.max(axis=1, keepdims=True))
            att = att / (att.sum(axis=1, keepdims=True)+1e-8)
            h = att @ v
            heads.append(h)
        Hcat = np.concatenate(heads, axis=1)
        return Hcat @ Wo

    def _encoder(self, X, Wq, Wk, Wv, Wo, F1, F2):
        # MHSA + residual + FFN + residual
        A = self._mhsa(X, Wq, Wk, Wv, Wo)
        X1 = X + A
        Z = np.tanh(X1 @ F1)
        Z = Z @ F2
        return X1 + Z

    def predict(self, feat: Dict[str,float]) -> float:
        X = self._seq(feat)
        H1 = self._encoder(X, self.Wq1,self.Wk1,self.Wv1,self.Wo1,self.F1_1,self.F1_2)
        H2 = self._encoder(H1, self.Wq2,self.Wk2,self.Wv2,self.Wo2,self.F2_1,self.F2_2)
        pooled = H2.mean(axis=0)
        y_arr = (pooled @ self.W_out).reshape(-1)
        return float(y_arr.item() + float(self.b_out))

    def fit_batch(self, X_list: List[Dict[str,float]], y: List[float], epochs=5):
        # lightweight training by finite-difference gradient approximation (for stability)
        if not X_list: return
        eps = 1e-3
        lr = self.lr
        for _ in range(epochs):
            # random sample a minibatch
            idx = np.random.default_rng().choice(len(X_list), size=min(16, len(X_list)), replace=False)
            loss = 0.0
            for k in idx:
                yhat = self.predict(X_list[k])
                loss += 0.5*(yhat - y[k])**2
            # update only output layer (cheap) + a tiny nudge to embeddings
            # gradient wrt output approx
            for _j in range(8):
                j = int(np.random.randint(0, len(self.keys)))
                base = self.E[j].copy()
                self.E[j] = base + eps
                y_plus = np.mean([self.predict(X_list[k]) for k in idx])
                self.E[j] = base - eps
                y_minus = np.mean([self.predict(X_list[k]) for k in idx])
                self.E[j] = base
                gE = (y_plus - y_minus)/(2*eps)
                self.E[j] -= lr * gE
            # update output layer by simple regression on pooled features
            pooled = []
            for k in idx:
                X = self._seq(X_list[k])
                H1 = self._encoder(X, self.Wq1,self.Wk1,self.Wv1,self.Wo1,self.F1_1,self.F1_2)
                H2 = self._encoder(H1, self.Wq2,self.Wk2,self.Wv2,self.Wo2,self.F2_1,self.F2_2)
                pooled.append(H2.mean(axis=0))
            P = np.vstack(pooled)                          # (B,d)
            yv = np.array([y[k] for k in idx]).reshape(-1,1)
            yhat = P @ self.W_out + self.b_out
            err = yhat - yv
            dW = P.T @ err / len(idx)
            db = float(np.mean(err))
            self.W_out -= lr * dW
            self.b_out  -= lr * db

    def save(self, path):
        np.savez(path,
                 E=self.E, Wq1=self.Wq1, Wk1=self.Wk1, Wv1=self.Wv1, Wo1=self.Wo1,
                 F11=self.F1_1, F12=self.F1_2,
                 Wq2=self.Wq2, Wk2=self.Wk2, Wv2=self.Wv2, Wo2=self.Wo2,
                 F21=self.F2_1, F22=self.F2_2,
                 W_out=self.W_out, b_out=self.b_out, keys=np.array(self.keys, dtype=object))

    @staticmethod
    def load(path):
        """
        Safe loader for transformer:
        - Returns None if file missing/old schema.
        - Quarantines unreadable files to *.bad.
        """
        try:
            if not os.path.exists(path):
                return None
            d = np.load(path, allow_pickle=True)
            required = {'keys','E',
                        'Wq1','Wk1','Wv1','Wo1','F11','F12',
                        'Wq2','Wk2','Wv2','Wo2','F21','F22',
                        'W_out','b_out'}
            if not required.issubset(set(d.files)):
                try: os.replace(path, path + ".bad")
                except Exception: pass
                return None
            keys = list(d['keys'])
            mdl = TransformerPreference(keys)
            mdl.E   = d['E']
            mdl.Wq1 = d['Wq1']; mdl.Wk1 = d['Wk1']; mdl.Wv1 = d['Wv1']; mdl.Wo1 = d['Wo1']
            mdl.F1_1= d['F11']; mdl.F1_2= d['F12']
            mdl.Wq2 = d['Wq2']; mdl.Wk2 = d['Wk2']; mdl.Wv2 = d['Wv2']; mdl.Wo2 = d['Wo2']
            mdl.F2_1= d['F21']; mdl.F2_2= d['F22']
            mdl.W_out = d['W_out']
            mdl.b_out = float(d['b_out'])
            return mdl
        except Exception:
            try: os.replace(path, path + ".bad")
            except Exception: pass
            return None

# -----------------------
# Supervised (cls + reg), Unsupervised (KMeans), and an Ensemble
# -----------------------

class SupervisedHead:
    """Logistic (classification) + Ridge (regression) on feature vectors."""
    def __init__(self, keys: List[str], lr=0.05, reg=1e-3):
        self.keys = keys; self.lr = lr; self.reg = reg
        d = len(keys)
        self.W  = np.zeros((d, 1), dtype=np.float32); self.b  = 0.0   # logistic
        self.Wr = np.zeros((d, 1), dtype=np.float32); self.br = 0.0   # ridge

    def _vec(self, feat: Dict[str,float]):
        return np.array([float(feat.get(k,0.0)) for k in self.keys], dtype=np.float32).reshape(-1,1)

    def fit(self, X_list: List[Dict[str,float]], y_cls: List[float], y_reg: List[float], epochs=40):
        if not X_list: return
        y_cls = [float(x) for x in y_cls]; y_reg = [float(x) for x in y_reg]
        for _ in range(epochs):
            for f, yc, yr in zip(X_list, y_cls, y_reg):
                x = self._vec(f)
                # logistic step
                z = (x.T @ self.W + self.b).item(); p = 1.0/(1.0 + np.exp(-z)); e = (p - yc)
                self.W  -= self.lr * (e * x + self.reg * self.W)
                self.b  -= self.lr * e
                # ridge step
                pred = (x.T @ self.Wr + self.br).item(); err = (pred - yr)
                self.Wr -= self.lr * (err * x + self.reg * self.Wr)
                self.br -= self.lr * err

    def predict_proba(self, feat: Dict[str,float]) -> float:
        x=self._vec(feat); z=(x.T @ self.W + self.b).item()
        return 1.0/(1.0 + np.exp(-z))

    def predict_reg(self, feat: Dict[str,float]) -> float:
        x=self._vec(feat); return (x.T @ self.Wr + self.br).item()

    def score(self, feat: Dict[str,float]) -> float:
        p = self.predict_proba(feat)             # [0,1]
        r = self.predict_reg(feat)               # normalized-ish reward
        return 0.6*(2.0*p - 1.0) + 0.4*r

    def save(self, path:str):
        np_savez_atomic(path, keys=np.array(self.keys, dtype=object),
                        W=self.W, b=np.array([self.b]),
                        Wr=self.Wr, br=np.array([self.br]))

    @staticmethod
    def load(path:str):
        if not os.path.exists(path): return None
        try:
            d = np.load(path, allow_pickle=True)
            m = SupervisedHead(list(d['keys']))
            m.W  = d['W'];  m.b  = float(d['b'][0])
            m.Wr = d['Wr']; m.br = float(d['br'][0])
            return m
        except Exception:
            return None


class KMeansLite:
    """Tiny KMeans with z-norm; marks the 'good' centroid by mean target."""
    def __init__(self, keys: List[str], k=4, seed=0):
        self.keys = keys; self.k = k; self.centers=None
        self.mu=None; self.sigma=None; self.good_idx=0; self.seed=seed

    def _vec(self, feat: Dict[str,float]):
        return np.array([float(feat.get(k,0.0)) for k in self.keys], dtype=np.float32)

    def fit(self, X_list: List[Dict[str,float]], y_quality: Optional[List[float]]=None, iters=30):
        if not X_list: return
        X = np.stack([self._vec(f) for f in X_list])
        self.mu = X.mean(axis=0); self.sigma = X.std(axis=0) + 1e-6
        Xn = (X - self.mu) / self.sigma
        rng = np.random.default_rng(self.seed)
        idx = rng.choice(len(Xn), size=min(self.k, len(Xn)), replace=False)
        C = Xn[idx]
        for _ in range(iters):
            D = ((Xn[:,None,:]-C[None,:,:])**2).sum(axis=2)
            labels = D.argmin(axis=1)
            for j in range(C.shape[0]):
                pts = Xn[labels==j]
                if len(pts)>0: C[j] = pts.mean(axis=0)
        self.centers = C
        if y_quality is not None and len(y_quality)==len(Xn):
            D = ((Xn[:,None,:]-C[None,:,:])**2).sum(axis=2)
            labels = D.argmin(axis=1)
            means = []
            for j in range(C.shape[0]):
                ys = [y_quality[i] for i in range(len(Xn)) if labels[i]==j]
                means.append(np.mean(ys) if ys else -1e9)
            self.good_idx = int(np.argmax(means))

    def score(self, feat: Dict[str,float]) -> float:
        if self.centers is None: return 0.0
        x = (self._vec(feat) - self.mu) / self.sigma
        dists = np.sqrt(((self.centers - x)**2).sum(axis=1))
        best = float(np.min(dists)); good = float(dists[self.good_idx])
        return (dists.mean() - good) / (1.0 + best)

    def save(self, path:str):
        if self.centers is None: return
        np_savez_atomic(path, keys=np.array(self.keys, dtype=object),
                        centers=self.centers, mu=self.mu, sigma=self.sigma,
                        good=np.array([self.good_idx]))

    @staticmethod
    def load(path:str):
        if not os.path.exists(path): return None
        try:
            d = np.load(path, allow_pickle=True)
            m = KMeansLite(list(d['keys']), k=int(d['centers'].shape[0]))
            m.centers = d['centers']; m.mu = d['mu']; m.sigma = d['sigma']; m.good_idx = int(d['good'][0])
            return m
        except Exception:
            return None


class MultiLearnerEnsemble:
    """Mix supervised + unsupervised scores."""
    def __init__(self, keys: List[str], sup: Optional[SupervisedHead], km: Optional[KMeansLite]):
        self.keys = keys; self.sup = sup; self.km = km

    def score(self, feat: Dict[str,float]) -> float:
        s = 0.0
        if self.sup: s += 0.7 * self.sup.score(feats=feat) if hasattr(self, 'score') else 0.7 * self.sup.score(feat)
        if self.km:  s += 0.3 * self.km.score(feat)
        return float(s)

    def save(self, path:str):
        np_savez_atomic(path, keys=np.array(self.keys, dtype=object))

    @staticmethod
    def load(path:str):
        if not os.path.exists(path): return None
        try:
            d = np.load(path, allow_pickle=True)
            keys = list(d['keys'])
            sup = SupervisedHead.load(SUP_FILE)
            km  = KMeansLite.load(KM_FILE)
            if sup is None and km is None: return None
            return MultiLearnerEnsemble(keys, sup, km)
        except Exception:
            return None


# -----------------------
# Additional deep models (NumPy-only)
# -----------------------

class AutoencoderPreference:
    """
    Small AE to learn compact feature embeddings.
    Score = -reconstruction_error + linear head on latent.
    """
    def __init__(self, keys: List[str], h=16, z=8, lr=0.02, seed=3):
        self.keys = keys; self.lr = lr
        rng = np.random.default_rng(seed)
        d = len(keys)
        self.W1 = rng.normal(0,0.2,(d,h)); self.b1 = np.zeros((h,))
        self.Wz = rng.normal(0,0.2,(h,z)); self.bz = np.zeros((z,))
        self.Wd1 = rng.normal(0,0.2,(z,h)); self.bd1= np.zeros((h,))
        self.Wout= rng.normal(0,0.2,(h,d)); self.bout=np.zeros((d,))
        self.Whead = rng.normal(0,0.2,(z,1)); self.bhead = np.zeros((1,))
    def _vec(self,f): return np.array([float(f.get(k,0.0)) for k in self.keys], dtype=np.float32)
    def encode(self,x):
        h = np.tanh(x@self.W1 + self.b1)
        z = np.tanh(h@self.Wz + self.bz); return z, h
    def decode(self,z):
        h = np.tanh(z@self.Wd1 + self.bd1)
        xr = h@self.Wout + self.bout; return xr
    def predict(self, feat):  # returns scalar
        x = self._vec(feat)
        z,_ = self.encode(x)
        xr = self.decode(z)
        recon = float(np.mean((xr - x)**2))
        y = float(z @ self.Whead + self.bhead)
        return float(y - recon)
    def fit_batch(self, X, y=None, epochs=8):
        if not X: return
        Xv = np.stack([self._vec(f) for f in X])
        for _ in range(epochs):
            z,h = self.encode(Xv)
            xr = self.decode(z)
            # recon loss
            err = xr - Xv
            dWout = h.T @ err / len(Xv); dbout = err.mean(axis=0)
            dh = err @ self.Wout.T * (1-h**2)
            dWd1 = z.T @ dh / len(Xv); dbd1 = dh.mean(axis=0)
            dz = dh @ self.Wd1.T * (1-z**2)
            dWz = h.T @ dz / len(Xv); dbz = dz.mean(axis=0)
            dW1 = Xv.T @ ((dz @ self.Wz.T)*(1-h**2)) / len(Xv)
            db1 = ((dz @ self.Wz.T)*(1-h**2)).mean(axis=0)
            # head (optional regression on y)
            if y is not None and len(y)==len(X):
                yv = np.array(y, dtype=np.float32).reshape(-1,1)
                pred = z @ self.Whead + self.bhead
                derr = (pred - yv) / len(Xv)
                dWh = z.T @ derr; dbh = derr.mean(axis=0)
                self.Whead -= self.lr * dWh; self.bhead -= self.lr * dbh
            # updates
            self.Wout -= self.lr*dWout; self.bout -= self.lr*dbout
            self.Wd1  -= self.lr*dWd1;  self.bd1  -= self.lr*dbd1
            self.Wz   -= self.lr*dWz;   self.bz   -= self.lr*dbz
            self.W1   -= self.lr*dW1;   self.b1   -= self.lr*db1
    def save(self,path):
        np.savez(path, keys=np.array(self.keys,dtype=object),
                 W1=self.W1,b1=self.b1,Wz=self.Wz,bz=self.bz,
                 Wd1=self.Wd1,bd1=self.bd1,Wout=self.Wout,bout=self.bout,
                 Whead=self.Whead,bhead=self.bhead)
    @staticmethod
    def load(path):
        try:
            if not os.path.exists(path): return None
            d=np.load(path,allow_pickle=True)
            req={'keys','W1','b1','Wz','bz','Wd1','bd1','Wout','bout','Whead','bhead'}
            if not req.issubset(set(d.files)): return None
            m=AutoencoderPreference(list(d['keys']))
            m.W1,dummy = d['W1'],None; m.b1=d['b1']
            m.Wz,m.bz = d['Wz'],d['bz']
            m.Wd1,m.bd1=d['Wd1'],d['bd1']
            m.Wout,m.bout=d['Wout'],d['bout']
            m.Whead,m.bhead=d['Whead'],d['bhead']
            return m
        except Exception: return None

class CNNPreference:
    """
    Tiny 2D conv net for grid snapshots (downsampled occupancy).
    Input: (H,W) ints -> one-hot channels 0..K; Conv3x3 -> ReLU -> GlobalAvgPool -> Linear.
    """
    def __init__(self, n_classes=8, d=16, lr=0.02, seed=11):
        self.K=n_classes; self.lr=lr
        rng=np.random.default_rng(seed)
        self.F = rng.normal(0,0.15,(3,3,self.K,d))  # 3x3xK->d
        self.W = rng.normal(0,0.15,(d,1)); self.b=0.0
    def _one_hot(self, G):
        H,W = G.shape; X = np.zeros((H,W,self.K), dtype=np.float32)
        Gc = np.clip(G,0,self.K-1)
        for k in range(self.K): X[:,:,k]=(Gc==k).astype(np.float32)
        return X
    def _conv(self,X):
        H,W,K = X.shape; d = self.F.shape[-1]
        Y = np.zeros((H-2,W-2,d), dtype=np.float32)
        for i in range(H-2):
            xs = X[i:i+3,:,:]
            for j in range(W-2):
                patch = xs[:,j:j+3,:]
                Y[i,j,:] = np.tensordot(patch, self.F, axes=((0,1,2),(0,1,2)))
        return np.maximum(0.0, Y)  # ReLU
    def predict(self, grid):
        if grid is None: return 0.0
        X=self._one_hot(grid)
        H=self._conv(X)
        v = H.mean(axis=(0,1))
        y = float(v @ self.W[:,0] + self.b)
        return y
    def fit_batch(self, grids, y, epochs=6):
        if not grids: return
        yv = np.array(y, dtype=np.float32).reshape(-1,1)
        for _ in range(epochs):
            idx = np.random.default_rng().choice(len(grids), size=min(8,len(grids)), replace=False)
            dW = np.zeros_like(self.W); db=0.0; dF = np.zeros_like(self.F)
            for k in idx:
                G = grids[k]; X = self._one_hot(G); H = self._conv(X)
                v = H.mean(axis=(0,1)).reshape(-1,1)
                pred = v.T @ self.W + self.b
                err = (pred - yv[k:k+1])
                dW += v * err
                db += float(err)
                # very rough gradient to F via linear backprop on active ReLU positions
                mask = (H>0).astype(np.float32)
                gradH = (err * self.W.reshape(1,1,-1)) * mask / (H.shape[0]*H.shape[1])
                for i in range(H.shape[0]):
                    for j in range(H.shape[1]):
                        patch = X[i:i+3, j:j+3, :]
                        dF += np.tensordot(patch, gradH[i,j,:], axes=0)
            self.W -= self.lr * dW / max(1,len(idx))
            self.b -= self.lr * db / max(1,len(idx))
            self.F -= self.lr * dF / max(1,len(idx))
    def save(self,path): np.savez(path, F=self.F,W=self.W,b=self.b)
    @staticmethod
    def load(path):
        try:
            if not os.path.exists(path): return None
            d=np.load(path,allow_pickle=True)
            m=CNNPreference()
            m.F,m.W,m.b=d['F'],d['W'],float(d['b'])
            return m
        except Exception: return None

class GRUPreference:
    """
    Tiny GRU on movement sequences (from simulation path deltas).
    """
    def __init__(self, d_in=4, d_h=16, lr=0.02, seed=21):
        rng=np.random.default_rng(seed)
        self.dh=d_h; self.lr=lr
        # input embed
        self.E = rng.normal(0,0.2,(d_in,d_h))
        # GRU params (update/reset/new)
        self.Wz=rng.normal(0,0.2,(d_h,d_h)); self.Uz=rng.normal(0,0.2,(d_h,d_h)); self.bz=np.zeros((d_h,))
        self.Wr=rng.normal(0,0.2,(d_h,d_h)); self.Ur=rng.normal(0,0.2,(d_h,d_h)); self.br=np.zeros((d_h,))
        self.Wh=rng.normal(0,0.2,(d_h,d_h)); self.Uh=rng.normal(0,0.2,(d_h,d_h)); self.bh=np.zeros((d_h,))
        self.Wo=rng.normal(0,0.2,(d_h,1)); self.bo=0.0
    def _step(self, x, h):
        z = 1/(1+np.exp(- (x@self.Wz + h@self.Uz + self.bz)))
        r = 1/(1+np.exp(- (x@self.Wr + h@self.Ur + self.br)))
        h_t = np.tanh(x@self.Wh + (r*h)@self.Uh + self.bh)
        h_new = (1-z)*h + z*h_t
        return h_new
    def predict(self, seq):
        if not seq: return 0.0
        h=np.zeros((self.dh,), dtype=np.float32)
        for t in seq:
            x=self.E[t]
            h=self._step(x,h)
        return float(h @ self.Wo[:,0] + self.bo)
    def fit_batch(self, seqs, y, epochs=6):
        if not seqs: return
        yv=np.array(y,dtype=np.float32).reshape(-1,1)
        for _ in range(epochs):
            idx=np.random.default_rng().choice(len(seqs), size=min(12,len(seqs)), replace=False)
            dWo=np.zeros_like(self.Wo); dbo=0.0  # (no full BPTT to keep it light)
            for k in idx:
                seq=seqs[k]
                if not seq: continue
                h=np.zeros((self.dh,), dtype=np.float32)
                for t in seq:
                    h=self._step(self.E[t], h)
                pred = h @ self.Wo[:,0] + self.bo
                err = pred - float(yv[k])
                dWo += h.reshape(-1,1) * err; dbo += err
            self.Wo -= self.lr * dWo / max(1,len(idx)); self.bo -= self.lr * dbo / max(1,len(idx))
    def save(self,path): np.savez(path, E=self.E,Wz=self.Wz,Uz=self.Uz,bz=self.bz,Wr=self.Wr,Ur=self.Ur,br=self.br,Wh=self.Wh,Uh=self.Uh,bh=self.bh,Wo=self.Wo,bo=self.bo)
    @staticmethod
    def load(path):
        try:
            if not os.path.exists(path): return None
            d=np.load(path,allow_pickle=True); m=GRUPreference()
            m.E=d['E']; m.Wz=d['Wz']; m.Uz=d['Uz']; m.bz=d['bz']
            m.Wr=d['Wr']; m.Ur=d['Ur']; m.br=d['br']
            m.Wh=d['Wh']; m.Uh=d['Uh']; m.bh=d['bh']
            m.Wo=d['Wo']; m.bo=float(d['bo']); return m
        except Exception: return None

class TinyFeatureGAN:
    """
    Very small discriminator on features; generator is implicit noise->linear.
    We use it as a learned 'realism' prior: D(x) high for good layouts.
    """
    def __init__(self, keys: List[str], lr=0.02, seed=4):
        self.keys=keys; self.lr=lr
        rng=np.random.default_rng(seed)
        self.W = rng.normal(0,0.2,(len(keys),1)); self.b=0.0
    def _vec(self,f): return np.array([float(f.get(k,0.0)) for k in self.keys], dtype=np.float32).reshape(-1,1)
    def predict(self, f):  # sigmoid score
        z = float(self._vec(f).T @ self.W + self.b)
        return 1.0/(1.0+np.exp(-z))
    def fit_batch(self, X_pos, X_neg, epochs=6):
        if not X_pos or not X_neg: return
        for _ in range(epochs):
            batch = np.random.default_rng().choice(len(X_pos), size=min(16,len(X_pos)), replace=False)
            dW=np.zeros_like(self.W); db=0.0
            for i in batch:
                xp = self._vec(X_pos[i]); xn = self._vec(X_neg[np.random.randint(0,len(X_neg))])
                for x, y in ((xp,1.0),(xn,0.0)):
                    z = float(x.T @ self.W + self.b); p = 1/(1+np.exp(-z)); e = (p - y)
                    dW += x * e; db += e
            self.W -= self.lr * dW / max(1,len(batch)); self.b -= self.lr * db / max(1,len(batch))
    def save(self,path): np.savez(path, keys=np.array(self.keys,dtype=object), W=self.W,b=self.b)
    @staticmethod
    def load(path):
        try:
            if not os.path.exists(path): return None
            d=np.load(path,allow_pickle=True)
            m=TinyFeatureGAN(list(d['keys'])); m.W=d['W']; m.b=float(d['b']); return m
        except Exception: return None

class EnsemblePreference:
    """Blend available models; missing ones are ignored."""
    def __init__(self, mlp=None, trf=None, ae=None, cnn=None, rnn=None, gan=None):
        self.mlp=mlp; self.trf=trf; self.ae=ae; self.cnn=cnn; self.rnn=rnn; self.gan=gan
    def predict(self, feats:Dict[str,float], grid=None, seq=None):
        score=0.0; wsum=0.0
        if self.mlp: score+=0.35*self.mlp.predict(feats); wsum+=0.35
        if self.trf: score+=0.45*self.trf.predict(feats); wsum+=0.45
        if self.ae:  score+=0.25*self.ae.predict(feats);  wsum+=0.25
        if self.cnn and grid is not None: score+=0.20*self.cnn.predict(grid); wsum+=0.20
        if self.rnn and seq: score+=0.15*self.rnn.predict(seq); wsum+=0.15
        if self.gan: score+=0.10*(self.gan.predict(feats)-0.5); wsum+=0.10
        return score if wsum==0.0 else score



# -----------------------
# RL – lightweight Q-learning over weight-nudge actions
# -----------------------

RL_ACTIONS = [
    {"coverage": +0.5}, {"coverage": -0.5},
    {"reach_windows": +0.5}, {"reach_windows": -0.5},
    {"paths_ok": +0.5}, {"paths_ok": -0.5},
    {"adjacency": +0.5}, {"adjacency": -0.5},
]

def rl_load():
    if os.path.exists(RL_FILE):
        try: return json.load(open(RL_FILE,'r'))
        except Exception: pass
    return {}

def rl_save(data):
    try: json.dump(data, open(RL_FILE,'w'))
    except Exception: pass

def rl_state_from_metrics(coverage, adjacency_hits, end_is_window):
    # Discretize to a small state space
    c = 2 if coverage>=0.98 else (1 if coverage>=0.9 else 0)
    a = 1 if adjacency_hits==0 else 0
    w = 1 if end_is_window else 0
    return f"c{c}_a{a}_w{w}"

def rl_choose_action(Q, state, epsilon=0.2):
    if random.random() < epsilon or state not in Q:
        return random.randrange(len(RL_ACTIONS))
    return int(np.argmax(Q[state]))

def rl_update(Q, state, action, reward, next_state, alpha=0.3, gamma=0.85):
    # tabular Q-learning
    if state not in Q: Q[state] = [0.0]*len(RL_ACTIONS)
    if next_state not in Q: Q[next_state] = [0.0]*len(RL_ACTIONS)
    best_next = max(Q[next_state])
    Q[state][action] += alpha * (reward + gamma * best_next - Q[state][action])
    return Q

def rl_apply_action_to_weights(W, action_idx, scale=0.2):
    for k,delta in RL_ACTIONS[action_idx].items():
        W[k] = W.get(k,0.0) + scale*delta
    save_weights(W)

# -----------------------
# Dialogs
# -----------------------

class ModeDialog(tk.Toplevel):
    def __init__(self, parent: tk.Misc):
        super().__init__(parent)
        self.title('Start a new design')
        self.transient(parent); self.grab_set(); self.resizable(False, False)
        self.result=None
        w,h=440,220; self._center(parent,w,h)
        f=ttk.Frame(self, padding=24); f.pack(fill=tk.BOTH, expand=True)
        ttk.Label(f, text='Choose mode', font=('SF Pro Text', 14, 'bold')).pack(anchor='w')
        ttk.Label(f, text='You can still edit after starting.').pack(anchor='w', pady=(0,12))
        b=ttk.Frame(f); b.pack(fill=tk.X, pady=12)
        ttk.Button(b, text='✏️  Sketch Design', style='Primary.TButton', command=lambda:self._ok('sketch')).pack(side=tk.LEFT, expand=True, fill=tk.X, padx=(0,8))
        ttk.Button(b, text='⚙️  Generate Design', style='Secondary.TButton', command=lambda:self._ok('generate')).pack(side=tk.LEFT, expand=True, fill=tk.X)
        ttk.Button(f, text='Cancel', command=self._cancel).pack(anchor='e', pady=(8,0))
        self.wait_visibility(); self.focus_set()
    def _center(self, parent, w, h):
        self.update_idletasks()
        try:
            x = parent.winfo_rootx() + max(0,(parent.winfo_width()-w)//2)
            y = parent.winfo_rooty() + max(0,(parent.winfo_height()-h)//2)
            self.geometry(f"{w}x{h}+{x}+{y}")
        except Exception:
            sw,sh=self.winfo_screenwidth(), self.winfo_screenheight()
            self.geometry(f"{w}x{h}+{(sw-w)//2}+{(sh-h)//2}")
    def _ok(self, mode): self.result=mode; self.destroy()
    def _cancel(self): self.result=None; self.destroy()


class OpeningDialog(tk.Toplevel):
    """Floating dialog to edit a door or window opening."""

    def __init__(self, parent: tk.Misc, info: dict, on_apply=None):
        super().__init__(parent)
        self.title(f"Edit {info.get('type', 'opening').title()}")
        self.transient(parent)
        self.grab_set()
        self.resizable(False, False)
        self.info = info
        self.on_apply = on_apply

        openings = info.get('openings')
        kind = info.get('type')
        idx = info.get('index')
        wall_names = ['Bottom', 'Right', 'Top', 'Left']

        if kind == 'door':
            wall = openings.door_wall
            length = openings.door_width
            center = openings.door_center
            length_label = 'Width'
        else:
            wall, start, length = openings.windows[idx]
            center = start + length / 2
            length_label = 'Length'

        self.wall_var = tk.StringVar(value=wall_names[wall])
        self.length_var = tk.DoubleVar(value=length)
        self.center_var = tk.DoubleVar(value=center)

        f = ttk.Frame(self, padding=12)
        f.pack(fill=tk.BOTH, expand=True)
        ttk.Label(f, text='Wall').grid(row=0, column=0, sticky='w')
        ttk.Combobox(
            f, textvariable=self.wall_var, values=wall_names, state='readonly', width=10
        ).grid(row=0, column=1, sticky='ew')
        ttk.Label(f, text=length_label).grid(row=1, column=0, sticky='w')
        ttk.Entry(f, textvariable=self.length_var, width=8).grid(row=1, column=1, sticky='ew')
        ttk.Label(f, text='Center').grid(row=2, column=0, sticky='w')
        ttk.Entry(f, textvariable=self.center_var, width=8).grid(row=2, column=1, sticky='ew')
        ttk.Button(f, text='OK', command=self._apply).grid(row=3, column=0, columnspan=2, pady=(8, 0))
        f.grid_columnconfigure(1, weight=1)
        self.wait_visibility()
        self.focus_set()

    def _apply(self):
        wall_map = {'Bottom': 0, 'Right': 1, 'Top': 2, 'Left': 3}
        wall = wall_map.get(self.wall_var.get(), 0)
        length = float(self.length_var.get())
        center = float(self.center_var.get())
        openings = self.info.get('openings')

        if self.info.get('type') == 'door':
            openings.door_wall = wall
            openings.door_width = length
            openings.door_center = center
        else:
            start = max(0.0, center - 0.5 * length)
            idx = self.info.get('index', 0)
            if 0 <= idx < len(openings.windows):
                openings.windows[idx] = [wall, start, length]

        if callable(self.on_apply):
            self.on_apply()
        self.destroy()

class AreaDialog(tk.Toplevel):
    UNITS=["m²","ft²","yd²","cm²","mm²","acre","hectare"]
    def __init__(self, parent: tk.Misc, mode_label: str, include_bed: bool = True):
        super().__init__(parent)
        self.title('Room Inputs')
        self.transient(parent); self.grab_set(); self.resizable(False, False)
        self.result=None
        self.include_bed = include_bed
        w,h=640,380; self._center(parent,w,h)
        f=ttk.Frame(self, padding=24); f.pack(fill=tk.BOTH, expand=True)
        ttk.Label(f, text=f'{mode_label}: set room inputs', font=('SF Pro Text', 14, 'bold')).pack(anchor='w')
        body=ttk.Frame(f); body.pack(fill=tk.X, pady=(10,0))
        self.method=tk.StringVar(value='area')
        ttk.Radiobutton(body, text='Area', variable=self.method, value='area').grid(row=0, column=0, sticky='w')
        ttk.Radiobutton(body, text='W × H', variable=self.method, value='dims').grid(row=0, column=1, sticky='w', padx=10)
        ttk.Label(body, text='Area').grid(row=1, column=0, sticky='w')
        self.area=tk.StringVar(value='12'); ttk.Entry(body, textvariable=self.area, width=10).grid(row=2, column=0, sticky='we')
        ttk.Label(body, text='Units').grid(row=1, column=1, sticky='w')
        self.area_units=tk.StringVar(value='m²'); ttk.Combobox(body, textvariable=self.area_units, values=self.UNITS, state='readonly', width=10).grid(row=2, column=1)
        ttk.Label(body, text='W (for W×H)').grid(row=1, column=2, sticky='w')
        self.W=tk.StringVar(value='4.2'); ttk.Entry(body, textvariable=self.W, width=10).grid(row=2, column=2)
        ttk.Label(body, text='H (for W×H)').grid(row=1, column=3, sticky='w')
        self.H=tk.StringVar(value='3.0'); ttk.Entry(body, textvariable=self.H, width=10).grid(row=2, column=3)
        ttk.Label(body, text='Len units').grid(row=1, column=4, sticky='w')
        self.len_units=tk.StringVar(value='m'); ttk.Combobox(body, textvariable=self.len_units, values=LENGTH_UNIT_LABELS, state='readonly', width=6).grid(row=2, column=4)
        for i in range(5): body.grid_columnconfigure(i, weight=1)

        if include_bed:
            opts=ttk.Frame(f); opts.pack(fill=tk.X, pady=(10,0))
            ttk.Label(opts, text='Bed Size').grid(row=0, column=0, sticky='w')
            self.bed=tk.StringVar(value='Auto')
            ttk.Combobox(opts, textvariable=self.bed,
                         values=['Auto','SINGLE','TWIN','THREE_Q_SMALL','DOUBLE'], state='readonly', width=16).grid(row=1, column=0, sticky='w')
        else:
            self.bed=None
        a=ttk.Frame(f); a.pack(fill=tk.X, pady=(12,0))
        ttk.Button(a, text='Continue', style='Primary.TButton', command=self._ok).pack(side=tk.RIGHT)
        ttk.Button(a, text='Cancel', command=self._cancel).pack(side=tk.RIGHT, padx=(0,8))
        self.wait_visibility(); self.focus_set()
    def _center(self, parent, w, h):
        self.update_idletasks()
        try:
            x = parent.winfo_rootx() + max(0,(parent.winfo_width()-w)//2)
            y = parent.winfo_rooty() + max(0,(parent.winfo_height()-h)//2)
            self.geometry(f"{w}x{h}+{x}+{y}")
        except Exception:
            sw,sh=self.winfo_screenwidth(), self.winfo_screenheight()
            self.geometry(f"{w}x{h}+{(sw-w)//2}+{(sh-h)//2}")
    def _ok(self):
        try:
            if self.method.get()=='area':
                A=float(self.area.get()); assert A>0
                bed_val = self.bed.get() if self.bed is not None else 'Auto'
                self.result={"mode":"area","area":A,"area_units":self.area_units.get(),"bed":bed_val}
            else:
                W=float(self.W.get()); H=float(self.H.get()); assert W>0 and H>0
                bed_val = self.bed.get() if self.bed is not None else 'Auto'
                self.result={"mode":"dims","W":W,"H":H,"len_units":self.len_units.get(),"bed":bed_val}
        except Exception:
            self.bell(); self.title('Room Inputs – enter valid numbers'); return
        self.destroy()
    def _cancel(self): self.result=None; self.destroy()

class AreaDialogCombined(tk.Toplevel):
    """Dialog for capturing bedroom, bathroom, living room and kitchen dimensions."""
    UNITS = AreaDialog.UNITS
    def __init__(self, parent: tk.Misc, mode_label: str):
        super().__init__(parent)
        self.title('Room Inputs')
        self.transient(parent); self.grab_set(); self.resizable(True, True)
        self.result=None
        w,h=640,1020; self._center(parent,w,h)
        f=ttk.Frame(self, padding=24); f.pack(fill=tk.BOTH, expand=True)
        ttk.Label(f, text=f'{mode_label}: set room inputs', font=('SF Pro Text', 14, 'bold')).pack(anchor='w')

        ttk.Label(f, text='Bedroom', font=('SF Pro Text', 12, 'bold')).pack(anchor='w', pady=(10,0))
        bed_body=ttk.Frame(f); bed_body.pack(fill=tk.X, pady=(5,0))
        self.bed_method=tk.StringVar(value='area')
        ttk.Radiobutton(bed_body, text='Area', variable=self.bed_method, value='area').grid(row=0, column=0, sticky='w')
        ttk.Radiobutton(bed_body, text='W × H', variable=self.bed_method, value='dims').grid(row=0, column=1, sticky='w', padx=10)
        ttk.Label(bed_body, text='Area').grid(row=1, column=0, sticky='w')
        self.bed_area=tk.StringVar(value='12'); ttk.Entry(bed_body, textvariable=self.bed_area, width=10).grid(row=2, column=0, sticky='we')
        ttk.Label(bed_body, text='Units').grid(row=1, column=1, sticky='w')
        self.bed_area_units=tk.StringVar(value='m²'); ttk.Combobox(bed_body, textvariable=self.bed_area_units, values=self.UNITS, state='readonly', width=10).grid(row=2, column=1)
        ttk.Label(bed_body, text='W (for W×H)').grid(row=1, column=2, sticky='w')
        self.bed_W=tk.StringVar(value='4.2'); ttk.Entry(bed_body, textvariable=self.bed_W, width=10).grid(row=2, column=2)
        ttk.Label(bed_body, text='H (for W×H)').grid(row=1, column=3, sticky='w')
        self.bed_H=tk.StringVar(value='3.0'); ttk.Entry(bed_body, textvariable=self.bed_H, width=10).grid(row=2, column=3)
        ttk.Label(bed_body, text='Len units').grid(row=1, column=4, sticky='w')
        self.bed_len_units=tk.StringVar(value='m'); ttk.Combobox(bed_body, textvariable=self.bed_len_units, values=LENGTH_UNIT_LABELS, state='readonly', width=6).grid(row=2, column=4)
        for i in range(5): bed_body.grid_columnconfigure(i, weight=1)
        bed_opts=ttk.Frame(f); bed_opts.pack(fill=tk.X, pady=(5,0))
        ttk.Label(bed_opts, text='Bed Size').grid(row=0, column=0, sticky='w')
        self.bed_size=tk.StringVar(value='Auto')
        ttk.Combobox(
            bed_opts,
            textvariable=self.bed_size,
            values=['Auto','SINGLE','TWIN','THREE_Q_SMALL','DOUBLE'],
            state='readonly',
            width=16,
        ).grid(row=1, column=0, sticky='w')
        ttk.Label(bed_opts, text='Door Wall').grid(row=0, column=1, sticky='w')
        self.bed_door_wall = tk.StringVar(value='Right')
        ttk.Combobox(
            bed_opts,
            textvariable=self.bed_door_wall,
            values=['Right'],
            state='readonly',
            width=16,
        ).grid(row=1, column=1, sticky='w')
        for i in range(2):
            bed_opts.grid_columnconfigure(i, weight=1)

        ttk.Label(f, text='Bathroom', font=('SF Pro Text', 12, 'bold')).pack(anchor='w', pady=(20,0))
        bath_body=ttk.Frame(f); bath_body.pack(fill=tk.X, pady=(5,0))
        self.bath_method=tk.StringVar(value='area')
        ttk.Radiobutton(bath_body, text='Area', variable=self.bath_method, value='area').grid(row=0, column=0, sticky='w')
        ttk.Radiobutton(bath_body, text='W × H', variable=self.bath_method, value='dims').grid(row=0, column=1, sticky='w', padx=10)
        ttk.Label(bath_body, text='Area').grid(row=1, column=0, sticky='w')
        self.bath_area=tk.StringVar(value='4'); ttk.Entry(bath_body, textvariable=self.bath_area, width=10).grid(row=2, column=0, sticky='we')
        ttk.Label(bath_body, text='Units').grid(row=1, column=1, sticky='w')
        self.bath_area_units=tk.StringVar(value='m²'); ttk.Combobox(bath_body, textvariable=self.bath_area_units, values=self.UNITS, state='readonly', width=10).grid(row=2, column=1)
        ttk.Label(bath_body, text='W (for W×H)').grid(row=1, column=2, sticky='w')
        self.bath_W=tk.StringVar(value='2.4'); ttk.Entry(bath_body, textvariable=self.bath_W, width=10).grid(row=2, column=2)
        ttk.Label(bath_body, text='H (for W×H)').grid(row=1, column=3, sticky='w')
        self.bath_H=tk.StringVar(value='1.8'); ttk.Entry(bath_body, textvariable=self.bath_H, width=10).grid(row=2, column=3)
        ttk.Label(bath_body, text='Len units').grid(row=1, column=4, sticky='w')
        self.bath_len_units=tk.StringVar(value='m'); ttk.Combobox(bath_body, textvariable=self.bath_len_units, values=LENGTH_UNIT_LABELS, state='readonly', width=6).grid(row=2, column=4)
        for i in range(5): bath_body.grid_columnconfigure(i, weight=1)

        ttk.Label(f, text='Living Room', font=('SF Pro Text', 12, 'bold')).pack(anchor='w', pady=(20,0))
        liv_body=ttk.Frame(f); liv_body.pack(fill=tk.X, pady=(5,0))
        self.liv_method=tk.StringVar(value='area')
        ttk.Radiobutton(liv_body, text='Area', variable=self.liv_method, value='area').grid(row=0, column=0, sticky='w')
        ttk.Radiobutton(liv_body, text='W × H', variable=self.liv_method, value='dims').grid(row=0, column=1, sticky='w', padx=10)
        ttk.Label(liv_body, text='Area').grid(row=1, column=0, sticky='w')
        self.liv_area=tk.StringVar(value='16'); ttk.Entry(liv_body, textvariable=self.liv_area, width=10).grid(row=2, column=0, sticky='we')
        ttk.Label(liv_body, text='Units').grid(row=1, column=1, sticky='w')
        self.liv_area_units=tk.StringVar(value='m²'); ttk.Combobox(liv_body, textvariable=self.liv_area_units, values=self.UNITS, state='readonly', width=10).grid(row=2, column=1)
        ttk.Label(liv_body, text='W (for W×H)').grid(row=1, column=2, sticky='w')
        self.liv_W=tk.StringVar(value='4.0'); ttk.Entry(liv_body, textvariable=self.liv_W, width=10).grid(row=2, column=2)
        ttk.Label(liv_body, text='H (for W×H)').grid(row=1, column=3, sticky='w')
        self.liv_H=tk.StringVar(value='4.0'); ttk.Entry(liv_body, textvariable=self.liv_H, width=10).grid(row=2, column=3)
        ttk.Label(liv_body, text='Len units').grid(row=1, column=4, sticky='w')
        self.liv_len_units=tk.StringVar(value='m'); ttk.Combobox(liv_body, textvariable=self.liv_len_units, values=LENGTH_UNIT_LABELS, state='readonly', width=6).grid(row=2, column=4)
        for i in range(5): liv_body.grid_columnconfigure(i, weight=1)

        ttk.Label(f, text='Kitchen', font=('SF Pro Text', 12, 'bold')).pack(anchor='w', pady=(20,0))
        kitch_body=ttk.Frame(f); kitch_body.pack(fill=tk.X, pady=(5,0))
        self.kitch_method=tk.StringVar(value='area')
        ttk.Radiobutton(kitch_body, text='Area', variable=self.kitch_method, value='area').grid(row=0, column=0, sticky='w')
        ttk.Radiobutton(kitch_body, text='W × H', variable=self.kitch_method, value='dims').grid(row=0, column=1, sticky='w', padx=10)
        ttk.Label(kitch_body, text='Area').grid(row=1, column=0, sticky='w')
        self.kitch_area=tk.StringVar(value='9'); ttk.Entry(kitch_body, textvariable=self.kitch_area, width=10).grid(row=2, column=0, sticky='we')
        ttk.Label(kitch_body, text='Units').grid(row=1, column=1, sticky='w')
        self.kitch_area_units=tk.StringVar(value='m²'); ttk.Combobox(kitch_body, textvariable=self.kitch_area_units, values=self.UNITS, state='readonly', width=10).grid(row=2, column=1)
        ttk.Label(kitch_body, text='W (for W×H)').grid(row=1, column=2, sticky='w')
        self.kitch_W=tk.StringVar(value='3.0'); ttk.Entry(kitch_body, textvariable=self.kitch_W, width=10).grid(row=2, column=2)
        ttk.Label(kitch_body, text='H (for W×H)').grid(row=1, column=3, sticky='w')
        self.kitch_H=tk.StringVar(value='3.0'); ttk.Entry(kitch_body, textvariable=self.kitch_H, width=10).grid(row=2, column=3)
        ttk.Label(kitch_body, text='Len units').grid(row=1, column=4, sticky='w')
        self.kitch_len_units=tk.StringVar(value='m'); ttk.Combobox(kitch_body, textvariable=self.kitch_len_units, values=LENGTH_UNIT_LABELS, state='readonly', width=6).grid(row=2, column=4)
        for i in range(5): kitch_body.grid_columnconfigure(i, weight=1)

        a=ttk.Frame(f); a.pack(fill=tk.X, pady=(12,0))
        ttk.Button(a, text='Continue', style='Primary.TButton', command=self._ok).pack(side=tk.RIGHT)
        ttk.Button(a, text='Cancel', command=self._cancel).pack(side=tk.RIGHT, padx=(0,8))
        self.wait_visibility(); self.focus_set()

    def _center(self, parent, w, h):
        self.update_idletasks()
        try:
            x = parent.winfo_rootx() + max(0,(parent.winfo_width()-w)//2)
            y = parent.winfo_rooty() + max(0,(parent.winfo_height()-h)//2)
            self.geometry(f"{w}x{h}+{x}+{y}")
        except Exception:
            sw,sh=self.winfo_screenwidth(), self.winfo_screenheight()
            self.geometry(f"{w}x{h}+{(sw-w)//2}+{(sh-h)//2}")

    def _ok(self):
        try:
            if self.bed_method.get()=='area':
                A=float(self.bed_area.get()); assert A>0
                bed_res={"mode":"area","area":A,"area_units":self.bed_area_units.get(),"bed":self.bed_size.get(),"door_wall":self.bed_door_wall.get()}
            else:
                W=float(self.bed_W.get()); H=float(self.bed_H.get()); assert W>0 and H>0
                bed_res={"mode":"dims","W":W,"H":H,"len_units":self.bed_len_units.get(),"bed":self.bed_size.get(),"door_wall":self.bed_door_wall.get()}
            if self.bath_method.get()=='area':
                A=float(self.bath_area.get()); assert A>0
                bath_res={"mode":"area","area":A,"area_units":self.bath_area_units.get(),"bed":"Auto"}
            else:
                W=float(self.bath_W.get()); H=float(self.bath_H.get()); assert W>0 and H>0
                bath_res={"mode":"dims","W":W,"H":H,"len_units":self.bath_len_units.get(),"bed":"Auto"}
            if self.liv_method.get()=='area':
                A=float(self.liv_area.get()); assert A>0
                liv_res={"mode":"area","area":A,"area_units":self.liv_area_units.get(),"bed":"Auto"}
            else:
                W=float(self.liv_W.get()); H=float(self.liv_H.get()); assert W>0 and H>0
                liv_res={"mode":"dims","W":W,"H":H,"len_units":self.liv_len_units.get(),"bed":"Auto"}
            if self.kitch_method.get()=='area':
                A=float(self.kitch_area.get()); assert A>0
                kitch_res={"mode":"area","area":A,"area_units":self.kitch_area_units.get(),"bed":"Auto"}
            else:
                W=float(self.kitch_W.get()); H=float(self.kitch_H.get()); assert W>0 and H>0
                kitch_res={"mode":"dims","W":W,"H":H,"len_units":self.kitch_len_units.get(),"bed":"Auto"}
            self.result={"bedroom":bed_res,"bathroom":bath_res,"livingroom":liv_res,"kitchen":kitch_res}
        except Exception:
            self.bell(); self.title('Room Inputs – enter valid numbers'); return
        self.destroy()

    def _cancel(self): self.result=None; self.destroy()

# -----------------------
# Sketch grid (quick painter)
# -----------------------

class SketchGrid:
    DEFAULT_FILL = "#404040"
    ROOM_TYPES = [("Living","#f2b632"),("Bedroom","#4aa3a2"),("Kitchen","#ef6f6c"),("Dining","#7f86ff"),("Bath","#6cc24a"),("Other","#a77dc2")]
    def __init__(self, master: tk.Misc, n_cells: int, unit_label: str, on_back=None):
        self.on_back=on_back
        self.master=master; self.unit_label=unit_label
        self.n_cells=max(1,int(n_cells)); self.W,self.H=self._best_grid_dims(self.n_cells)
        self.margin=40; self.box=24; self.sx=self.sy=0
        self.colors=[[self.DEFAULT_FILL for _ in range(self.W)] for _ in range(self.H)]
        self.current_label,self.current_color=self.ROOM_TYPES[0]
        self.wrap=ttk.Frame(master); self.wrap.pack(fill=tk.BOTH, expand=True)
        tb=ttk.Frame(self.wrap, style='Toolbar.TFrame'); tb.pack(fill=tk.X)
        ttk.Button(tb, text='← Back', command=self._back).pack(side=tk.LEFT, padx=6, pady=4)
        self.canvas=tk.Canvas(self.wrap, bg="#1A1A1A", highlightthickness=0); self.canvas.pack(fill=tk.BOTH, expand=True)
        self.status=ttk.Frame(self.wrap); self.status.pack(fill=tk.X, side=tk.BOTTOM)
        self.count_labels={}; self._build_statusbar()
        self.sel_start=None; self.sel_rect_id=None
        self.menu=tk.Menu(self.canvas, tearoff=0)
        for idx,(name,color) in enumerate(self.ROOM_TYPES, start=1):
            self.menu.add_command(label=f"{idx}. {name}", command=lambda n=name,c=color:self.choose_color(n,c))
        self.menu.add_separator(); self.menu.add_command(label="Erase (empty)", command=lambda:self.choose_color("Empty", self.DEFAULT_FILL))
        self._bind_events(); self._reflow()
    def _back(self):
        try: self.wrap.destroy()
        except: pass
        if callable(self.on_back): self.on_back()
    def _best_grid_dims(self, n):
        if n<=0: return (1,1)
        r=int(sqrt(n)); best=None
        for h in range(1,r+1):
            if n%h==0:
                w=n//h
                if best is None or abs(w-h)<abs(best[0]-best[1]): best=(w,h)
        if best: return best
        w=int(ceil(sqrt(n))); h=int(ceil(n/w)); return (w,h)
    def _bind_events(self):
        self.canvas.bind('<Configure>', lambda e:self._reflow())
        self.canvas.bind('<Button-1>', self._on_drag_start)
        self.canvas.bind('<B1-Motion>', self._on_drag_move)
        self.canvas.bind('<ButtonRelease-1>', self._on_drag_end)
        for seq in ('<Button-3>','<Button-2>','<Control-Button-1>'): self.canvas.bind(seq, self._on_right_click)
        self.master.bind('<Escape>', lambda e:self._clear_selection())
    def _build_statusbar(self):
        self.current_var=tk.StringVar(value=f"Current: {self.current_label}")
        ttk.Label(self.status, textvariable=self.current_var).pack(side=tk.LEFT, padx=(8,12))
        self.count_labels['Empty']=ttk.Label(self.status, text="Empty: 0"); self.count_labels['Empty'].pack(side=tk.RIGHT, padx=6)
        for name,_ in reversed(self.ROOM_TYPES):
            lbl=ttk.Label(self.status, text=f"{name}: 0"); lbl.pack(side=tk.RIGHT, padx=6); self.count_labels[name]=lbl
        self._refresh_counts()
    def _reflow(self):
        cw=self.canvas.winfo_width() or 1; ch=self.canvas.winfo_height() or 1
        avail_w=max(1,cw-2*self.margin); avail_h=max(1,ch-2*self.margin)
        self.box=max(1,int(min(avail_w/self.W, avail_h/self.H)))
        grid_w=self.box*self.W; grid_h=self.box*self.H
        self.sx=(cw-grid_w)//2; self.sy=(ch-grid_h)//2
        self._redraw()
    def _redraw(self):
        c=self.canvas; c.delete('all')
        for i in range(self.H):
            y0=self.sy+i*self.box; y1=y0+self.box
            for j in range(self.W):
                x0=self.sx+j*self.box; x1=x0+self.box
                fill=self.colors[i][j]
                c.create_rectangle(x0,y0,x1,y1, fill=fill, outline="#2a2a2a")
        if self.sel_start and self.sel_rect_id: self.canvas.tag_raise(self.sel_rect_id)
    def _xy_to_ij(self, x,y):
        if not (self.sx<=x<self.sx+self.W*self.box and self.sy<=y<self.sy+self.H*self.box): return (None,None)
        j=int((x-self.sx)//self.box); i=int((y-self.sy)//self.box)
        return (clamp(i,0,self.H-1), clamp(j,0,self.W-1))
    def _on_drag_start(self, e):
        i,j=self._xy_to_ij(e.x,e.y);
        if i is None: return
        self.sel_start=(i,j); self._update_sel_rect(i,j,i,j)
    def _on_drag_move(self, e):
        if not self.sel_start: return
        i0,j0=self.sel_start; i1,j1=self._xy_to_ij(e.x,e.y)
        if i1 is None: return
        self._update_sel_rect(i0,j0,i1,j1)
    def _on_drag_end(self, e):
        if not self.sel_start: return
        i0,j0=self.sel_start; i1,j1=self._xy_to_ij(e.x,e.y)
        if i1 is None: self._clear_selection(); return
        imin,imax=sorted((i0,i1)); jmin,jmax=sorted((j0,j1))
        for i in range(imin,imax+1):
            for j in range(jmin,jmax+1): self.colors[i][j]=self.current_color
        self._clear_selection(); self._refresh_counts(); self._redraw()
    def _on_right_click(self, e):
        try: self.menu.tk_popup(e.x_root,e.y_root)
        finally: self.menu.grab_release()
    def choose_color(self,label,color):
        self.current_label,self.current_color=label,color
        self.current_var.set(f"Current: {label}")
    def _update_sel_rect(self,i0,j0,i1,j1):
        imin,imax=sorted((i0,i1)); jmin,jmax=sorted((j0,j1))
        x0=self.sx+jmin*self.box; y0=self.sy+imin*self.box
        x1=self.sx+(jmax+1)*self.box; y1=self.sy+(imax+1)*self.box
        if self.sel_rect_id is None:
            self.sel_rect_id=self.canvas.create_rectangle(x0,y0,x1,y1, outline="#fff", dash=(4,3), width=1)
        else:
            self.canvas.coords(self.sel_rect_id, x0,y0,x1,y1)
    def _clear_selection(self):
        if self.sel_rect_id is not None: self.canvas.delete(self.sel_rect_id); self.sel_rect_id=None
        self.sel_start=None
    def _refresh_counts(self):
        counts={name:0 for name,_ in self.ROOM_TYPES}; empty=0
        for i in range(self.H):
            for j in range(self.W):
                c=self.colors[i][j]; hit=False
                for name,col in self.ROOM_TYPES:
                    if c==col: counts[name]+=1; hit=True; break
                if not hit: empty+=1
        for name,_ in self.ROOM_TYPES: pass

# -----------------------
# Generate model
# -----------------------


class ColumnGrid:
    """Helper to map grid coordinates to spreadsheet-style labels.

    Vertical grid lines are labelled alphabetically (A, B, ...), while
    horizontal lines use numbers (1, 2, ...).  Labels reference the intersection
    at the lower-left corner of a cell, allowing cells to be addressed via a
    combined label such as ``B3``.
    """

    def __init__(self, gw: int, gh: int):
        self.gw = gw
        self.gh = gh

    @staticmethod
    def col_label(idx: int) -> str:
        label = ""
        n = idx
        while True:
            label = chr(ord("A") + (n % 26)) + label
            n = n // 26 - 1
            if n < 0:
                break
        return label

    @staticmethod
    def _col_index(label: str) -> int:
        n = 0
        for ch in label:
            n = n * 26 + (ord(ch) - ord("A") + 1)
        return n - 1

    @staticmethod
    def row_label(idx: int) -> str:
        return str(idx + 1)

    def coord_to_label(self, i: int, j: int) -> str:
        return f"{self.col_label(i)}{self.row_label(j)}"

    def label_to_coord(self, label: str) -> Tuple[int, int]:
        import re

        m = re.fullmatch(r"([A-Z]+)(\d+)", label)
        if not m:
            raise ValueError(f"Invalid label: {label}")
        col_s, row_s = m.groups()
        i = self._col_index(col_s)
        j = int(row_s) - 1
        return i, j


class GridPlan:
    def __init__(self, Wm: float, Hm: float, column_grid: ColumnGrid = None,
                 x_offset: int = 0, y_offset: int = 0):
        self.Wm = Wm
        self.Hm = Hm
        self.cell = CELL_M
        self.gw = max(1, int(round(Wm / self.cell)))
        self.gh = max(1, int(round(Hm / self.cell)))
        self.occ = [[None for _ in range(self.gw)] for _ in range(self.gh)]
        # clearzones: (x,y,w,h,kind,owner)
        self.clearzones: List[Tuple[int, int, int, int, str, str]] = []
        self.column_grid = column_grid
        self.x_offset = x_offset
        self.y_offset = y_offset

    def coord_to_label(self, i: int, j: int) -> str:
        if not self.column_grid:
            raise ValueError("ColumnGrid not attached")
        return self.column_grid.coord_to_label(i + self.x_offset, j + self.y_offset)

    def label_to_coord(self, label: str) -> Tuple[int, int]:
        if not self.column_grid:
            raise ValueError("ColumnGrid not attached")
        i, j = self.column_grid.label_to_coord(label)
        return i - self.x_offset, j - self.y_offset
    def fits(self, x:int,y:int,w:int,h:int)->bool:
        if x<0 or y<0 or x+w>self.gw or y+h>self.gh: return False
        for j in range(y,y+h):
            for i in range(x,x+w):
                if self.occ[j][i] is not None: return False
        for cx, cy, cw, ch, kind, _ in self.clearzones:
            if kind == 'DOOR_CLEAR' and not (
                x + w <= cx or cx + cw <= x or y + h <= cy or cy + ch <= y
            ):
                return False
        return True
    def place(self, x:int,y:int,w:int,h:int, code:str):
        if not self.fits(x, y, w, h):
            raise ValueError("placement does not fit")
        for j in range(y,y+h):
            for i in range(x,x+w):
                self.occ[j][i]=code
        return (x,y,w,h)
    def clear(self, x:int,y:int,w:int,h:int):
        for j in range(y,y+h):
            for i in range(x,x+w): self.occ[j][i]=None
    def mark_clear(self, x:float, y:float, w:float, h:float, kind:str, owner:str):
        # Zero-gap: start exactly at element edge (caller must pass adjacent coords)
        x0 = max(0, int(floor(x)))
        y0 = max(0, int(floor(y)))
        x1 = min(self.gw, int(ceil(x + w)))
        y1 = min(self.gh, int(ceil(y + h)))
        w_int = max(0, x1 - x0)
        h_int = max(0, y1 - y0)
        if w_int > 0 and h_int > 0:
            cleared = set()
            for j in range(y0, y0 + h_int):
                for i in range(x0, x0 + w_int):
                    code = self.occ[j][i]
                    if code is None or code in cleared:
                        continue
                    cleared.add(code)
                    Q = [(i, j)]
                    cells = []
                    seen = {(i, j)}
                    while Q:
                        cx, cy = Q.pop()
                        cells.append((cx, cy))
                        for dx, dy in ((1,0),(-1,0),(0,1),(0,-1)):
                            nx, ny = cx + dx, cy + dy
                            if 0 <= nx < self.gw and 0 <= ny < self.gh \
                               and (nx, ny) not in seen and self.occ[ny][nx] == code:
                                seen.add((nx, ny))
                                Q.append((nx, ny))
                    xs = [c[0] for c in cells]
                    ys = [c[1] for c in cells]
                    bx0, by0 = min(xs), min(ys)
                    bx1, by1 = max(xs) + 1, max(ys) + 1
                    self.clear(bx0, by0, bx1 - bx0, by1 - by0)
            self.clearzones.append((x, y, w, h, kind, owner))
    def meters_to_cells(self, m:float)->int:
        return max(1, int(ceil(m/self.cell)))

WALL_BOTTOM = 0
WALL_RIGHT = 1
WALL_TOP = 2
WALL_LEFT = 3


def opposite_wall(w: int) -> int:
    """Return the wall opposite to ``w``."""
    return (w + 2) % 4


class Openings:
    def __init__(self, plan:GridPlan):
        self.p=plan
        self.door_wall=WALL_LEFT  # default orientation
        self.door_center=0.25*plan.Hm
        self.door_width=0.90
        # Default door swing depth: 1 grid cell
        self.swing_depth = CELL_M
        # (wall, start_m, length_m)
        self.windows=[[1, plan.Hm*0.40, 1.20], [-1,0.0,0.0]]
        self.ext_rect = None
    def door_rect_cells(self)->Tuple[int,int,int,int]:
        p=self.p; swing=p.meters_to_cells(self.swing_depth); w=p.meters_to_cells(self.door_width)
        if self.door_wall==0:
            x=max(0, min(p.gw-w, int(round(self.door_center/p.cell))-w//2)); return (x, 0, w, swing)
        if self.door_wall==2:
            x=max(0, min(p.gw-w, int(round(self.door_center/p.cell))-w//2)); return (x, p.gh-swing, w, swing)
        if self.door_wall==3:
            y=max(0, min(p.gh-w, int(round(self.door_center/p.cell))-w//2)); return (0, y, swing, w)
        y=max(0, min(p.gh-w, int(round(self.door_center/p.cell))-w//2)); return (p.gw-swing, y, swing, w)
    def door_span_cells(self)->Tuple[int,int,int]:
        p=self.p; w=p.meters_to_cells(self.door_width)
        if self.door_wall in (WALL_BOTTOM, WALL_TOP):
            x=max(0, min(p.gw-w, int(round(self.door_center/p.cell))-w//2))
            return (self.door_wall, x, w)
        y=max(0, min(p.gh-w, int(round(self.door_center/p.cell))-w//2))
        return (self.door_wall, y, w)
    def window_spans_cells(self)->List[Tuple[int,int,int]]:
        res=[]; p=self.p
        for wall,start_m,length_m in self.windows:
            if wall<0 or length_m<=0: continue
            L = p.gw if wall in (0,2) else p.gh
            start_c=max(0,min(L-1,int(round(start_m/p.cell)))); length_c=max(1,int(round(length_m/p.cell)))
            res.append((wall,start_c,length_c))
        return res

def components_by_code(plan, code:str):
    """Return list of (x,y,w,h,wall) rects for contiguous components of given base code."""
    gw, gh = plan.gw, plan.gh
    seen = set()
    comps = []
    def infer_wall(x,y,w,h):
        if y == 0: return 0
        if y + h == gh: return 2
        if x == 0: return 3
        if x + w == gw: return 1
        return -1
    for j in range(gh):
        for i in range(gw):
            if (i,j) in seen: continue
            cell = plan.occ[j][i]
            if cell is None: continue
            base = cell.split(':')[0] if isinstance(cell,str) else cell
            if base != code: continue
            Q=[(i,j)]; seen.add((i,j)); cells=[]
            while Q:
                x,y = Q.pop(); cells.append((x,y))
                for di,dj in ((1,0),(-1,0),(0,1),(0,-1)):
                    nx,ny = x+di, y+dj
                    if 0<=nx<gw and 0<=ny<gh and (nx,ny) not in seen and plan.occ[ny][nx] == plan.occ[j][i]:
                        seen.add((nx,ny)); Q.append((nx,ny))
            xs=[c[0] for c in cells]; ys=[c[1] for c in cells]
            x0=min(xs); y0=min(ys); x1=max(xs); y1=max(ys)
            w=x1-x0+1; h=y1-y0+1
            comps.append((x0,y0,w,h,infer_wall(x0,y0,w,h)))
    return comps

# -----------------------
# Learning utilities (rehydrate + train deep nets)
# -----------------------

def rehydrate_from_feedback():
    """
    Replays historical feedback + simulation runs and trains:
      - Weights (per-feature)
      - Deep MLP + Transformer (regression)
      - SupervisedHead (classification + regression)
      - KMeansLite (unsupervised structure)
      - Semi-supervised: pseudo-labels on unlabeled feature snapshots
    Returns: (W, mlp, trf, None, None, None, None, ensemble)
    """
    W = load_weights()
    co_pos=defaultdict(int); co_neg=defaultdict(int)

    # labeled
    feats_reg: List[Dict[str,float]] = []
    y_reg:   List[float] = []
    feats_cls: List[Dict[str,float]] = []
    y_cls:   List[float] = []

    # unlabeled pool
    unlabeled_feats: List[Dict[str,float]] = []

    # feedback.jsonl
    try:
        with open(FEEDBACK_FILE, 'r') as f:
            for ln in f:
                try: ev = json.loads(ln)
                except Exception: continue
                if ev.get('event') == 'feedback_feature':
                    key = ev.get('feature'); sign = +1 if ev.get('sign') == 'up' else -1
                    val = float(ev.get('value', 1.0))
                    update_weights(W, {key: val}, sign)
                elif ev.get('event') == 'feedback':
                    sign = +1 if ev.get('feedback') in ('up','+','pos','👍') else -1
                    feats = ev.get('features', {})
                    update_weights(W, {'paths_ok': 0.4, 'coverage': 0.4, 'symmetry': 0.2}, sign)
                    if feats:
                        feats_cls.append(feats); y_cls.append(1.0 if sign>0 else 0.0)
                        feats_reg.append(feats); y_reg.append(1.0 if sign>0 else -1.0)
                        for a,b in itertools.combinations(sorted(feats.keys()),2):
                            (co_pos if sign>0 else co_neg)[(a,b)] += 1
    except FileNotFoundError:
        pass

    # simulations.jsonl
    try:
        with open(SIM_FILE, 'r') as f:
            for ln in f:
                try: sim = json.loads(ln)
                except Exception: continue
                if sim.get('event') == 'solve_result':
                    unl = sim.get('features', {})
                    if unl: unlabeled_feats.append(unl)
                    continue
                if sim.get('event') == 'simulate_full_coverage':
                    collisions = int(sim.get('collisions', 0))
                    coverage   = float(sim.get('coverage_free', sim.get('coverage', 0.0)))
                    unmet_adj  = int(sim.get('unmet_adjacency', 0))
                    win_ok     = bool(sim.get('end_is_window', True))
                    reward = (coverage*2.0) + (1.0 if win_ok else -1.0) + (-0.8*unmet_adj) + (-0.2*collisions)
                    feats = {'coverage':coverage,'reach_windows':1.0 if win_ok else 0.0,
                             'paths_ok': 1.0 if collisions==0 else 0.0,
                             'adjacency': 1.0 if unmet_adj==0 else 0.0}
                    feats_reg.append(feats); y_reg.append(reward)
    except FileNotFoundError:
        pass

    # correlation nudges
    for (a,b),cnt in co_pos.items():
        if cnt>0: update_weights(W, {a:0.05,b:0.05}, +1)
    for (a,b),cnt in co_neg.items():
        if cnt>0: update_weights(W, {a:0.05,b:0.05}, -1)

    # Deep models (existing)
    mlp = None; trf = None
    if feats_reg:
        all_keys=set()
        for f in feats_reg: all_keys|=set(f.keys())
        keys=sorted(list(all_keys))
        yv = np.array(y_reg, dtype=np.float32)
        if yv.std()>1e-6:
            yv = (yv - yv.mean()) / (abs(yv).max()+1e-6)
        y_list = list(yv)
        mlp = NeuralPreference.load(NN_FILE) or NeuralPreference(keys, h1=32, h2=20, lr=0.03, seed=12)
        Xexp=[{k:f.get(k,0.0) for k in keys} for f in feats_reg]
        mlp.fit_batch(Xexp, y_list, epochs=12); mlp.save(NN_FILE)
        trf = TransformerPreference.load(TRF_FILE) or TransformerPreference(keys, d_model=48, n_heads=4, lr=0.01, seed=9)
        trf.fit_batch(Xexp, y_list, epochs=8); trf.save(TRF_FILE)

    # Supervised + Unsupervised + Semi-supervised
    ensemble = None
    if feats_reg or feats_cls or unlabeled_feats:
        keys=set()
        for f in feats_reg: keys|=set(f.keys())
        for f in feats_cls: keys|=set(f.keys())
        for f in unlabeled_feats: keys|=set(f.keys())
        keys=sorted(list(keys))
        def align(lst): return [{k:f.get(k,0.0) for k in keys} for f in lst]
        Xr = align(feats_reg); yr = list(np.array(y_reg, dtype=np.float32))
        Xc = align(feats_cls); yc = [float(v) for v in y_cls] if y_cls else []

        sup = SupervisedHead(keys, lr=0.05, reg=1e-3)
        if not Xc: Xc, yc = (Xr, [(1.0 if y>0 else 0.0) for y in yr])
        if not Xr: Xr, yr = (Xc, [1.0 if y>0.5 else -1.0 for y in yc])
        sup.fit(Xc, yc, yr, epochs=40)

        # semi-supervised: confident pseudo labels
        if unlabeled_feats:
            Xu = align(unlabeled_feats)
            probs = [sup.predict_proba(f) for f in Xu]
            pseudo = []
            for f, p in zip(Xu, probs):
                if p>=0.85 or p<=0.15:
                    pseudo.append((f, 1.0 if p>0.5 else 0.0, float(2.0*p-1.0)))
            if pseudo:
                Xp, ycp, yrp = zip(*pseudo)
                Xc2 = list(Xc) + list(Xp)
                yc2 = list(yc) + list(ycp)
                yr2 = list(yr) + list(yrp)
                sup.fit(Xc2, yc2, yr2, epochs=20)

        km = KMeansLite(keys, k=4, seed=7)
        q = yr if yr else [1.0 if y>0.5 else -1.0 for y in yc]
        km.fit(list(Xr or Xc), list(q), iters=30)

        sup.save(SUP_FILE); km.save(KM_FILE)
        ensemble = MultiLearnerEnsemble(keys, sup, km)
        try: ensemble.save(ENS_FILE)
        except Exception: pass

    save_weights(W)
    return W, mlp, trf, None, None, None, None, ensemble

# -----------------------
# Seeds / filters by room area
# -----------------------

def area_based_filters(Wm:float, Hm:float):
    A=Wm*Hm
    allow_large_wr = A>=11.0
    allow_double   = A>=11.0
    allow_twin     = A>=9.0
    return {
        "bed_choices": ['SINGLE','THREE_Q_SMALL','TWIN','DOUBLE'] if allow_double else (['SINGLE','THREE_Q_SMALL','TWIN'] if allow_twin else ['SINGLE','THREE_Q_SMALL']),
        "wardrobe_order": (['WRD_S_210','WRD_S_180','WRD_H_180','WRD_H_150'] if allow_large_wr else ['WRD_H_180','WRD_H_150','WRD_S_180'])
    }

def seed_templates(plan:'GridPlan', bed_key:str)->List[Dict]:
    bw=plan.meters_to_cells(BEDROOM_BOOK['BEDS'][bed_key]['w'])
    bd=plan.meters_to_cells(BEDROOM_BOOK['BEDS'][bed_key]['d'])
    g=plan.gw; h=plan.gh
    seeds=[]
    for wall in (0,2,3,1):
        if wall in (0,2):
            seeds.append({'wall':wall,'beds':[ (g//2-bw//2, 0 if wall==0 else h-bd, bw, bd) ]})
        else:
            seeds.append({'wall':wall,'beds':[ (0 if wall==3 else g-bd, h//2-bw//2, bd, bw) ]})
    offs=[-g//6, g//6]
    for dx in offs:
        seeds.append({'wall':0,'beds':[ (clamp(g//2-bw//2+dx,0,g-bw), 0, bw, bd) ]})
        seeds.append({'wall':2,'beds':[ (clamp(g//2-bw//2+dx,0,g-bw), h-bd, bw, bd) ]})
    offs=[-h//6, h//6]
    for dy in offs:
        seeds.append({'wall':3,'beds':[ (0, clamp(h//2-bw//2+dy,0,h-bw), bd, bw) ]})
        seeds.append({'wall':1,'beds':[ (g-bd, clamp(h//2-bw//2+dy,0,h-bw), bd, bw) ]})
    if bed_key=='TWIN':
        t_bw=bw; t_bd=bd
        seeds.append({'wall':0,'beds':[ (g//2-t_bw-1,0,t_bw,t_bd), (g//2+1,0,t_bw,t_bd) ]})
        seeds.append({'wall':2,'beds':[ (g//2-t_bw-1,h-t_bd,t_bw,t_bd), (g//2+1,h-t_bd,t_bw,t_bd) ]})
    return seeds

# -----------------------
# Solver
# -----------------------

def default_furniture_sets(extra_sets: Optional[List[Tuple[str, ...]]] = None) -> List[Tuple[str, ...]]:
    """Return the default search order for required furniture combinations.

    Each tuple represents a multiset of codes that must be present.  The
    ordering is ascending in complexity so callers may ``reversed`` it to try
    the most demanding combination first.  ``extra_sets`` allows future callers
    to extend the search without modifying the core routine.
    """
    base: List[Tuple[str, ...]] = [
        ("BED",),
        ("BED", "BST"),
        ("BED", "BST", "BST"),
        ("BED", "DRS"),
        ("BED", "BST", "DRS"),
        ("BED", "BST", "BST", "DRS"),
        ("BED", "WRD"),
        ("BED", "BST", "WRD"),
        ("BED", "BST", "BST", "WRD"),
        ("BED", "DRS", "WRD"),
        ("BED", "BST", "DRS", "WRD"),
        ("BED", "BST", "BST", "DRS", "WRD"),
    ]
    if extra_sets:
        base.extend(extra_sets)
    return base


def default_kitchen_sets(extra_sets: Optional[List[Tuple[str, ...]]] = None) -> List[Tuple[str, ...]]:
    """Return the default search order for kitchen appliance combinations.

    The ordering progresses from the simplest requirement (a single
    appliance) to the full work triangle of refrigerator, sink, and
    cooktop.  ``extra_sets`` permits callers to extend the search without
    modifying the core routine.
    """
    base: List[Tuple[str, ...]] = [
        ("SINK",),
        ("COOK",),
        ("REF",),
        ("SINK", "COOK"),
        ("SINK", "REF"),
        ("COOK", "REF"),
        ("SINK", "COOK", "REF"),
    ]
    if extra_sets:
        base.extend(extra_sets)
    return base

class BedroomSolver:
    def __init__(self,
                 plan:GridPlan,
                 openings:Openings,
                 bed_key:Optional[str],
                 rng:random.Random,
                 weights:Dict[str,float],
                 mlp=None,
                 transformer=None,
                 ensemble=None,
                 pref=None,
                 force_bst_pair: bool = False):
        self.p=plan; self.op=openings; self.rng=rng; self.weights=weights
        self.mlp = mlp
        # unify attribute name used by run(); keep alias for any legacy refs
        self.transformer = transformer
        self.trf = transformer
        self.ensemble = ensemble
        self.c = BEDROOM_BOOK['CLEAR']
        filters=area_based_filters(plan.Wm, plan.Hm)
        self.allowed_beds=filters["bed_choices"]
        self.wr_order=filters["wardrobe_order"]
        self.bed_key = (bed_key if (bed_key and bed_key in self.allowed_beds) else self.autopick_bed())
        self.force_bst_pair = bool(force_bst_pair)

    def autopick_bed(self)->str:
        Wm=self.p.Wm; Hm=self.p.Hm; A=Wm*Hm
        options=self.allowed_beds
        if 'DOUBLE' in options and (min(Wm,Hm)>=3.1 and A>=12): return 'DOUBLE'
        if 'THREE_Q_SMALL' in options: return 'THREE_Q_SMALL'
        return options[0]

    def run(self,
            iters: int = 900,
            time_budget_ms: int = 520,
            max_attempts: int = 3,
            furniture_sets: Optional[List[Tuple[str, ...]]] = None
            ) -> Tuple[Optional[GridPlan], Dict]:
        sets = furniture_sets or default_furniture_sets()
        # search from most demanding combination to least
        for req in reversed(sets):
            needed = Counter(req)
            for _ in range(max_attempts):
                t0 = time.time()
                best = None; best_meta = {}; best_score = -1e9
                seeds = seed_templates(self.p, self.bed_key)
                beam = []
                tries = 0
                while (time.time() - t0) * 1000 < time_budget_ms and tries < iters:
                    tries += 1
                    seed = self.rng.choice(seeds)
                    res, meta, feats = self._attempt(seed)
                    if not res:
                        continue
                    base_score = dot_score(self.weights, feats)
                    adj = self._adjacency_score(res)
                    feats['adjacency'] = adj
                    base_score += self.weights.get('adjacency', 0.6) * adj
                    if self.mlp:
                        base_score += 0.35 * self.mlp.predict(feats)
                    if self.transformer:
                        base_score += 0.45 * self.transformer.predict(feats)
                    if getattr(self, 'ensemble', None):
                        try:
                            base_score += 0.40 * self.ensemble.score(feats)
                        except Exception:
                            pass
                    beam.append((base_score, res, {**meta, 'features': feats, 'score': base_score}))
                    beam.sort(key=lambda x: -x[0])
                    beam = beam[:6]
                if beam:
                    best_score, best, best_meta = beam[0]
                if best and all(len(list(components_by_code(best, code))) >= count
                                 for code, count in needed.items()):
                    return best, best_meta
        return None, {'status': 'no_bed'}

    def _attempt(self, seed:Dict):
        res,meta,feats=self._try_seed(seed)
        if res: return res,meta,feats
        for relax in [0.0, 0.4, 0.8]:
            for wall in (seed['wall'], 0,2,3,1):
                for shift in (-3,-2,-1,0,1,2,3):
                    s2=self._shifted_seed(seed, wall, shift)
                    res,meta,feats=self._try_seed(s2, relax)
                    if res: return res,meta,feats
        return None,{},{}

    def _grid_snapshot(self, plan:'GridPlan', max_hw: int = 16):
        """Downsample occupancy to a small int grid for CNN/analytics."""
        mapping = {'BED':1,'BST':2,'WRD':3,'DRS':4,'DESK':5,'TVU':6}
        H = min(max_hw, plan.gh); W = min(max_hw, plan.gw)
        sx = max(1, plan.gw // W); sy = max(1, plan.gh // H)
        G = np.zeros((H,W), dtype=np.int8)
        jj=0
        for y in range(0, plan.gh, sy):
            ii=0
            for x in range(0, plan.gw, sx):
                c = plan.occ[y][x]
                if c:
                    base = c.split(':')[0]
                    G[jj,ii] = mapping.get(base,7)
                ii+=1
                if ii>=W: break
            jj+=1
            if jj>=H: break
        return G


    def _shifted_seed(self, seed:Dict, wall:int, shift:int)->Dict:
        beds=[]
        for (x,y,w,h) in seed['beds']:
            if wall in (0,2):
                beds.append((clamp(x+shift,0,self.p.gw-w), 0 if wall==0 else self.p.gh-h, w,h))
            elif wall==3:
                beds.append((0, clamp(y+shift,0,self.p.gh-h), w,h))
            else:
                beds.append((self.p.gw-w, clamp(y+shift,0,self.p.gh-h), w,h))
        return {'wall':wall,'beds':beds}

    def _try_seed(self, seed:Dict, relax_factor:float=0.0):
        p=GridPlan(self.p.Wm,self.p.Hm)
        # Door block
        dx,dy,dw,dh=self.op.door_rect_cells()
        for j in range(dy, dy+dh):
            for i in range(dx, dx+dw): p.occ[j][i]='DOOR'
        self._add_door_clearance(p, owner='DOOR')

        # Place bed(s)
        beds=[]
        for (x,y,w,h) in seed['beds']:
            jx=max(0,min(p.gw-w, x + self.rng.randint(-1,1)))
            jy=max(0,min(p.gh-h, y + self.rng.randint(-1,1)))
            if not p.fits(jx,jy,w,h): return None,{},{}
            if not self._bed_touches_wall(jx,jy,w,h,seed['wall']): return None,{},{}
            side=p.meters_to_cells(self.c['side_rec']*(1.0-relax_factor) + self.c['side_min']*relax_factor)
            side=max(0, side-2)
            foot=p.meters_to_cells(self.c['foot_rec']*(1.0-relax_factor) + self.c['foot_min']*relax_factor)
            if not self._clear_ok(p,jx,jy,w,h,seed['wall'],side,foot): return None,{},{}
            p.place(jx,jy,w,h,f'BED:{self.bed_key}')
            # zero-gap clearances at edges
            if seed['wall'] in (0,2):
                p.mark_clear(jx-side, jy, side, h, 'SIDE', 'BED')
                p.mark_clear(jx+w, jy, side, h, 'SIDE', 'BED')
                if seed['wall']==0: p.mark_clear(jx, jy+h, w, foot, 'FOOT', 'BED')
                else:               p.mark_clear(jx, jy-foot, w, foot, 'FOOT', 'BED')
            else:
                p.mark_clear(jx, jy-side, w, side, 'SIDE', 'BED')
                p.mark_clear(jx, jy+h, w, side, 'SIDE', 'BED')
                if seed['wall']==3:
                    p.mark_clear(jx+w, jy, foot, h, 'FOOT', 'BED')
                else:
                    p.mark_clear(jx-foot, jy, foot, h, 'FOOT', 'BED')
            beds.append((jx,jy,w,h,seed['wall']))

        # Discard candidates that failed to place any bed
        if not beds:
            return None, {}, {}

        # Night tables (strict)
        bst=BEDROOM_BOOK['NIGHT_TABLE']['BST_18']
        tw=p.meters_to_cells(bst['w']); td=p.meters_to_cells(bst['d'])
        placed_bst=0
        for (bx,by,bw,bd,bedwall) in beds:
            placed_bst += self._place_bst_pair_strict(p, bedwall, bx,by,bw,bd, tw,td)

        # If user requires a pair for each bed, enforce it hard
        if getattr(self, 'force_bst_pair', False) and placed_bst < 2*len(beds):
            return None, {}, {}


        # Windows clearances
        self._add_window_clearances(p)

        # Wardrobe + Dresser
        wr = self._place_wall_unit(p, 'WRD')
        dr = self._place_wall_unit(p, 'DRS')
        if self.force_bst_pair and placed_bst < 2*len(beds):
            return None, {}, {}

        # Path feasibility & coverage (coarse)
        ok, coverage, reach_windows = self._paths_and_coverage(p)
        if not ok: return None,{},{}

        # Ensure at least one bed remains after all placements
        final_beds = components_by_code(p, 'BED')
        if not final_beds:
            return None, {}, {}
        expected = sorted((bw, bh) for (_, _, bw, bh, _) in beds)
        actual = sorted((w, h) for (_, _, w, h, _) in final_beds)
        if actual != expected:
            return None, {}, {}

        # features
        def desk_near_window():
            desks = components_by_code(p, 'DESK')
            if not desks:
                return 0.0
            win_spans = self.op.window_spans_cells()
            for (x,y,w,h,wall) in desks:
                if wall < 0: continue
                if wall in (0,2):
                    a0,a1=x,x+w
                    for ww,s,lenv in win_spans:
                        if ww==wall:
                            b0,b1=s,s+lenv
                            if not (a1<=b0 or b1<=a0): return 1.0
                else:
                    a0,a1=y,y+h
                    for ww,s,lenv in win_spans:
                        if ww==wall:
                            b0,b1=s,s+lenv
                            if not (a1<=b0 or b1<=a0): return 1.0
            return 0.0

        feats = {
            'bst_pair': 1.0 if placed_bst >= 2*len(beds) else 0.5 if placed_bst > 0 else 0.0,
            'has_wr': 1.0 if wr else 0.0,
            'has_dr': 1.0 if dr else 0.0,
            'privacy': 1.0 if seed['wall'] != 0 else 0.5,
            'symmetry': 1.0 if placed_bst in (0,2,4) else 0.5,
            'paths_ok': 1.0,
            'use_rec_clear': 1.0 if relax_factor == 0.0 else 0.0,
            'bed_not_bottom': 1.0 if seed['wall'] != 0 else 0.0,
            'coverage': coverage,
            'reach_windows': 1.0 if reach_windows else 0.0,
            'door_align': 1.0,
            'longedge_wall': 1.0,
            'near_window_desk': desk_near_window()
        }

        meta={'placed_bst':placed_bst,'has_wr':bool(wr),'has_dr':bool(dr),'wall':seed['wall'],
              'coverage':coverage,'reach_windows':reach_windows}
        p.clearzones = merge_clearances(p.clearzones)
        return p, meta, feats

    # --- helpers

    def _bed_touches_wall(self,x,y,w,h,wall)->bool:
        if wall==0: return y==0
        if wall==2: return (y+h)==self.p.gh
        if wall==3: return x==0
        return (x+w)==self.p.gw

    def _clear_ok(self, p:GridPlan,x:int,y:int,w:int,h:int, wall:int, side:int, foot:int)->bool:
        if wall in (0,2):
            if x-side<0 or x+w+side>p.gw: return False
            if wall==0 and y+h+foot>p.gh: return False
            if wall==2 and y-foot<0: return False
        else:
            if y-side<0 or y+h+side>p.gh: return False
            if wall==3 and x+w+foot>p.gw: return False
            if wall==1 and x-foot<0: return False
        return True

    def _span_blocks_opening(self, wall:int, start:int, width:int, x:int,y:int,w:int,h:int)->bool:
        if wall in (0,2) and y==(0 if wall==0 else self.p.gh-h):
            a0=x; a1=x+w; b0=start; b1=start+width
            return not (a1<=b0 or b1<=a0)
        if wall in (3,1) and x==(0 if wall==3 else self.p.gw-w):
            a0=y; a1=y+h; b0=start; b1=start+width
            return not (a1<=b0 or b1<=a0)
        return False

    def _place_bst_pair_strict(self, p:GridPlan, bed_wall:int, bx:int,by:int,bw:int,bd:int, tw:int,td:int)->int:
        """
        Place up to two bedside tables adjacent to the bed's accessible sides.

        Works for all bed-wall orientations:
          - Bed on top/bottom wall → place tables to the left & right, flush to that wall.
          - Bed on left/right wall → place tables above & below, flush to that wall.

        Pass 1: strict (avoid door + avoid windows)
        Pass 2: relaxed (avoid door, ignore windows)
        Pass 3: inset by 1 cell from the wall (still avoid door)
        """
        # Treat bw/bd as the ACTUAL placed rect size (bw = bed width in cells, bd = bed height in cells).
        bed_w, bed_h = bw, bd

        dwall, dstart, dwidth = self.op.door_span_cells()
        win_spans = self.op.window_spans_cells()

        # Compute the door clearance rectangle (swing area) to avoid blocking access
        swing = max(1, p.meters_to_cells(self.op.swing_depth))
        if dwall == 0:
            door_clear = (dstart, 0, dwidth, swing)
        elif dwall == 2:
            door_clear = (dstart, p.gh - swing, dwidth, swing)
        elif dwall == 3:
            door_clear = (0, dstart, swing, dwidth)
        else:
            door_clear = (p.gw - swing, dstart, swing, dwidth)

        bath_door_span = None
        if getattr(self, 'bath_openings', None):
            bwall, bstart, bwidth = self.bath_openings.door_span_cells()
            if bwall == 3:   # shared left wall
                bath_door_span = (bstart, bwidth)

        def touches_opening(x, y, w, h, avoid_windows: bool) -> bool:
            # identify which wall the candidate touches
            if y == 0: wall = 0
            elif y + h == p.gh: wall = 2
            elif x == 0: wall = 3
            elif x + w == p.gw: wall = 1
            else: wall = -1

            if wall == dwall and self._span_blocks_opening(wall, max(0, dstart-1), max(1, dwidth+2), x, y, w, h):
                return True
            if avoid_windows and wall >= 0:
                for ww, start, L in win_spans:
                    if ww == wall and self._span_blocks_opening(wall, start, L, x, y, w, h):
                        return True
            return False

        def try_pass(avoid_windows: bool, inset: int) -> int:
            placed = 0
            if bed_wall in (0, 2):
                # bedside tables along same top/bottom wall → y anchored to the wall
                y = (0 if bed_wall == 0 else (p.gh - td))
                y = y + (inset if bed_wall == 0 else -inset)
                candidates = [
                    (bx - tw,      y, tw, td),               # left of bed
                    (bx + bed_w,   y, tw, td),               # right of bed
                ]
            else:
                # bedside tables along same left/right wall → x anchored to the wall
                x = (0 if bed_wall == 3 else (p.gw - tw))
                x = x + (inset if bed_wall == 3 else -inset)
                candidates = [
                    (x, by - td,        tw, td),             # above bed
                    (x, by + bed_h,     tw, td),             # below bed
                ]

            for (x, y, w, h) in candidates:
                # keep inside bounds and empty
                if x < 0 or y < 0 or x + w > p.gw or y + h > p.gh:
                    continue
                # Skip if door clearance area is on the same wall and overlaps
                if bed_wall == dwall:
                    dcx, dcy, dcw, dch = door_clear
                    if not (x + w <= dcx or dcx + dcw <= x or y + h <= dcy or dcy + dch <= y):
                        continue
                if bath_door_span and x == 0:
                    bstart, bwidth = bath_door_span
                    if not (y + h <= bstart or bstart + bwidth <= y):
                        continue
                if not p.fits(x, y, w, h):
                    continue

                # demand true side adjacency to the bed rectangle
                # (rectangles share a side and overlap along the other axis)
                side_adjacent = (
                    # left of bed
                    (x + w == bx and not (y + h <= by or by + bed_h <= y)) or
                    # right of bed
                    (x == bx + bed_w and not (y + h <= by or by + bed_h <= y)) or
                    # above bed
                    (y + h == by and not (x + w <= bx or bx + bed_w <= x)) or
                    # below bed
                    (y == by + bed_h and not (x + w <= bx or bx + bed_w <= x))
                )
                if not side_adjacent:
                    continue

                if touches_opening(x, y, w, h, avoid_windows):
                    continue

                p.place(x, y, w, h, 'BST')
                placed += 1
                if placed == 2:
                    break
            return placed

        total = 0
        total += try_pass(avoid_windows=True,  inset=0)
        if total < 2:
            total += try_pass(avoid_windows=False, inset=0)
        if total < 2:
            total += try_pass(avoid_windows=False, inset=1)

        return min(total, 2)

    def _place_wall_unit(self, p:GridPlan, kind:str):
        if kind=='WRD': specs = self.wr_order
        else: specs = ['DRS_4FT','CHEST_SM']
        dwall,dstart,dwidth=self.op.door_span_cells()
        win_spans=self.op.window_spans_cells()
        bath_door_span = None
        if getattr(self, "bath_openings", None):
            bwall, bstart, bwidth = self.bath_openings.door_span_cells()
            if bwall == 3:
                bath_door_span = (bstart, bwidth)
        swing = max(1, p.meters_to_cells(self.op.swing_depth))
        if dwall == 0:
            door_clear = (dstart, 0, dwidth, swing)
        elif dwall == 2:
            door_clear = (dstart, p.gh - swing, dwidth, swing)
        elif dwall == 3:
            door_clear = (0, dstart, swing, dwidth)
        else:
            door_clear = (p.gw - swing, dstart, swing, dwidth)
        for name in specs:
            spec = BEDROOM_BOOK['WARDROBE'][name] if kind=='WRD' else BEDROOM_BOOK['DRESSER'][name]
            W=p.meters_to_cells(spec['w']); D=p.meters_to_cells(spec['d'])
            orient_sizes = []
            ww,hh=(max(W,D), min(W,D))
            orient_sizes += [(0, ww, hh), (2, ww, hh)]
            orient_sizes += [(3, hh, ww), (1, hh, ww)]
            self.rng.shuffle(orient_sizes)
            for wall, ww, hh in orient_sizes:
                options=[]
                if wall==0: y=0; xs=range(0, p.gw-ww+1); options += [(x,y,ww,hh) for x in xs]
                elif wall==2: y=p.gh-hh; xs=range(0, p.gw-ww+1); options += [(x,y,ww,hh) for x in xs]
                elif wall==3: x=0; ys=range(0, p.gh-hh+1); options += [(x,y,ww,hh) for y in ys]
                else: x=p.gw-ww; ys=range(0, p.gh-hh+1); options += [(x,y,ww,hh) for y in ys]
                self.rng.shuffle(options)
                for x,y,w,h in options:
                    if wall==dwall:
                        if self._span_blocks_opening(wall,max(0,dstart-1),max(1,dwidth+2),x,y,w,h): continue
                        dcx,dcy,dcw,dch = door_clear
                        if not (x + w <= dcx or dcx + dcw <= x or y + h <= dcy or dcy + dch <= y):
                            continue
                    bad=False
                    for wwspan,start,lenv in win_spans:
                        if wall==wwspan and self._span_blocks_opening(wall,start,lenv,x,y,w,h): bad=True; break
                    if bad: continue
                    if bath_door_span and wall == 3:
                        bstart, bwidth = bath_door_span
                        if self._span_blocks_opening(3, max(0, bstart - 1),
                                                     max(1, bwidth + 2), x, y, w, h):
                            continue
                    if not p.fits(x,y,w,h): continue
                    p.place(x,y,w,h, kind)
                    if 'front_rec' in spec:
                        fc = p.meters_to_cells(spec['front_rec'])
                        if kind == 'WRD' and fc == 4:
                            fc = 2
                        clear_w, clear_x = w, x
                        clear_h, clear_y = h, y
                        if kind == 'WRD' and wall in (0, 2) and w > 1:
                            clear_w = w - 1
                            clear_x = x + 0.5
                        elif kind == 'DRS':
                            fc = 2 if fc == 4 else 5 if fc == 3 else fc
                            rect_w = 5
                            rect_d = 2
                            clear_w = rect_w
                            clear_h = rect_w
                            clear_x = x + (w - rect_w) / 2
                            clear_y = y + (h - rect_w) / 2
                            if wall == 0:
                                p.mark_clear(clear_x, y + h, rect_w, rect_d, 'FRONT', kind)
                            elif wall == 2:
                                p.mark_clear(clear_x, y - rect_d, rect_w, rect_d, 'FRONT', kind)
                            elif wall == 3:
                                p.mark_clear(x + w, clear_y, rect_d, rect_w, 'FRONT', kind)
                            else:
                                p.mark_clear(x - rect_d, clear_y, rect_d, rect_w, 'FRONT', kind)
                            return (x, y, w, h)
                        if wall == 0:
                            p.mark_clear(clear_x, y + h, clear_w, fc, 'FRONT', kind)
                        elif wall == 2:
                            p.mark_clear(clear_x, y - fc, clear_w, fc, 'FRONT', kind)
                        elif wall == 3:
                            p.mark_clear(x + w, clear_y, fc, clear_h, 'FRONT', kind)
                        else:
                            p.mark_clear(x - fc, clear_y, fc, clear_h, 'FRONT', kind)
                    return (x,y,w,h)
        return None

    def _add_door_clearance(self, p: GridPlan, owner: str):
        """Mark door clearance on both sides of the doorway.

        The interior clearance rectangle is computed as before. A mirrored
        rectangle on the opposite side of the door is then derived by
        reflecting across the door line (``y`` for walls 0/2 and ``x`` for
        walls 1/3). Both rectangles are recorded via ``p.mark_clear`` using
        ``DOOR_CLEAR`` with distinct owners (``INSIDE`` and ``OUTSIDE``).

        The mirrored rectangle is returned in ``p``'s cell units so that callers
        (e.g. a bathroom planner) may convert it to metres and apply it to an
        adjacent room if needed.
        """
        wall, start, width = self.op.door_span_cells()
        depth = p.meters_to_cells(self.op.swing_depth)
        pw = max(1, PATH_WIDTH_CELLS)

        if wall == WALL_BOTTOM:  # door along bottom wall
            inside = (start, depth, width, pw)
            line = 0
            outside = (start, 2 * line - (depth + pw), width, pw)
        elif wall == WALL_TOP:  # door along top wall
            inside = (start, p.gh - depth - pw, width, pw)
            line = p.gh
            outside = (start, 2 * line - (inside[1] + pw), width, pw)
        elif wall == WALL_LEFT:  # door along left wall
            inside = (depth, start, pw, width)
            line = 0
            outside = (2 * line - (depth + pw), start, pw, width)
        else:  # WALL_RIGHT
            inside = (p.gw - depth - pw, start, pw, width)
            line = p.gw
            outside = (2 * line - (inside[0] + pw), start, pw, width)

        p.mark_clear(*inside, 'DOOR_CLEAR', 'INSIDE')
        p.mark_clear(*outside, 'DOOR_CLEAR', 'OUTSIDE')

        return outside

    def _add_window_clearances(self, p:GridPlan):
        depth = max(1, p.meters_to_cells(0.40))
        for wall,start,length in self.op.window_spans_cells():
            if wall==0:   p.mark_clear(start, depth, length, 1, 'WIN_CLEAR','WINDOW')
            elif wall==2: p.mark_clear(start, p.gh-1-depth, length, 1, 'WIN_CLEAR','WINDOW')
            elif wall==3: p.mark_clear(1, start, 1, length, 'WIN_CLEAR','WINDOW')
            else:         p.mark_clear(p.gw-2, start, 1, length, 'WIN_CLEAR','WINDOW')

    def _paths_and_coverage(self, p:GridPlan)->Tuple[bool,float,bool]:
        gw,gh=p.gw,p.gh
        blocked=[[p.occ[j][i] is not None and p.occ[j][i] != 'DOOR' for i in range(gw)] for j in range(gh)]
        free=sum(1 for j in range(gh) for i in range(gw) if not blocked[j][i])
        dx,dy,dw,dh=self.op.door_rect_cells()
        if self.op.door_wall==0: start=(dx+dw//2, dh)
        elif self.op.door_wall==2: start=(dx+dw//2, gh-dh-1)
        elif self.op.door_wall==3: start=(dw, dy+dh//2)
        else: start=(gw-dw-1, dy+dh//2)
        if not (0<=start[0]<gw and 0<=start[1]<gh): return False,0.0,False
        blocked[start[1]][start[0]]=False

        Q=deque([start]); seen={start}
        while Q:
            i,j=Q.popleft()
            for di,dj in ((1,0),(-1,0),(0,1),(0,-1)):
                ni=i+di; nj=j+dj
                if 0<=ni<gw and 0<=nj<gh and not blocked[nj][ni] and (ni,nj) not in seen:
                    seen.add((ni,nj)); Q.append((ni,nj))
        coverage=len(seen)/max(1,free)

        targets=[]
        for wall,start,lenv in self.op.window_spans_cells():
            mid=start+max(0,lenv//2)
            if wall==0: targets.append((mid, 1))
            elif wall==2: targets.append((mid, gh-2))
            elif wall==3: targets.append((1, mid))
            else: targets.append((gw-2, mid))
        reach_windows=all(t in seen for t in targets) if targets else True
        ok = (coverage>0.60 and reach_windows)
        return ok, coverage, reach_windows

    def _adjacency_score(self, plan:GridPlan) -> float:
        """
        Coarse adjacency score using simple matrix; higher is better.
        """
        A = {
            'BED': {'BST': +2.0, 'WRD': +0.8, 'DRS': +0.6, 'DESK': +0.4},
            'DESK': {'WIN': +1.5},  # desk near windows
        }
        def boxes(code):
            comps = components_by_code(plan, code)
            if comps:
                x,y,w,h,_ = comps[0]
                return (x,y,x+w-1,y+h-1)
            return None
        score = 0.0
        bbed = boxes('BED')
        if bbed:
            bx0, by0, bx1, by1 = bbed
            for other, wt in A['BED'].items():
                b = boxes(other)
                if not b: continue
                ox0, oy0, ox1, oy1 = b
                dx = max(0, max(ox0 - bx1, bx0 - ox1))
                dy = max(0, max(oy0 - by1, by0 - oy1))
                score += wt * (1.0 / (1.0 + dx + dy))
        # desk near window
        bdesk = boxes('DESK')
        if bdesk:
            dx0, dy0, dx1, dy1 = bdesk
            for wall, s, L in self.op.window_spans_cells():
                if wall in (0,2):
                    a0, a1 = dx0, dx1
                    b0, b1 = s, s+L
                    if not (a1<b0 or b1<a0):
                        score += A['DESK']['WIN']; break
        return score


class KitchenSolver:
    def __init__(self,
                 plan: GridPlan,
                 openings: Openings,
                 rng: random.Random,
                 weights: Dict[str, float],
                 book: Dict[str, Dict] = KITCHEN_BOOK):
        self.p = plan
        self.op = openings
        self.rng = rng
        self.weights = weights
        self.book = book
        # Allow callers to override clearance distances by supplying a custom
        # ``book`` mapping with a ``CLEAR`` section.  Falling back to an empty
        # mapping keeps behavior consistent with the default catalog.
        self.c = self.book.get('CLEAR', {})

    def run(self,
            appliance_sets: Optional[List[Tuple[str, ...]]] = None,
            iters: int = 200,
            time_budget_ms: int = 520,
            max_attempts: int = 3,
            min_adjacency: float = 0.0) -> Tuple[Optional[GridPlan], Dict]:
        sets = appliance_sets or default_kitchen_sets()
        best_overall = None; meta_overall = None
        for req in reversed(sets):
            needed = Counter(req)
            for _ in range(max_attempts):
                t0 = time.time()
                tries = 0
                best = None; best_meta = None; best_adj = -1.0
                while (time.time() - t0) * 1000 < time_budget_ms and tries < iters:
                    tries += 1
                    plan = deepcopy(self.p)
                    success = True
                    for code, cnt in needed.items():
                        existing = list(components_by_code(plan, code))
                        missing = cnt - len(existing)
                        for _ in range(missing):
                            spot = self._find_placement(plan, code)
                            if spot is None:
                                success = False
                                break
                            x, y, w, h = spot
                            plan.place(x, y, w, h, code)
                        if not success:
                            break
                    if not success:
                        continue
                    feats = {'work_triangle_bonus': self._work_triangle_bonus(plan)}
                    adj, ok = self._adjacency_score(plan)
                    if not ok:
                        continue
                    feats['adjacency'] = adj
                    score = dot_score(self.weights, feats)
                    cand_meta = {'features': feats, 'score': score, 'adjacency': adj}
                    if adj > best_adj or (adj == best_adj and score > (best_meta or {}).get('score', -1e9)):
                        best_adj = adj
                        best = plan
                        best_meta = cand_meta
                    if adj >= 1.0:
                        break
                if best and best_meta['features'].get('adjacency', 0.0) >= min_adjacency:
                    return best, best_meta
                if best and (meta_overall is None or best_meta['features']['adjacency'] > meta_overall['features']['adjacency']):
                    best_overall, meta_overall = best, best_meta
        if meta_overall:
            return None, {**meta_overall, 'status': 'adjacency_below_threshold'}
        return None, {'status': 'missing_appliance'}

    def _find_placement(self, plan: GridPlan, code: str) -> Optional[Tuple[int, int, int, int]]:
        """Return a valid (x,y,w,h) placement for ``code`` or ``None``."""
        variants = self.book.get(code, {})
        if not variants:
            defaults = {'SINK': (0.6, 0.6), 'COOK': (0.6, 0.6), 'REF': (0.9, 0.76)}
            w_m, d_m = defaults.get(code, (0.6, 0.6))
            variants = {'_': {'w': w_m, 'd': d_m}}
        for dims in variants.values():
            w = plan.meters_to_cells(dims.get('w', 0.6))
            h = plan.meters_to_cells(dims.get('d', 0.6))
            for w0, h0 in ((w, h), (h, w)):
                for y in range(plan.gh - h0 + 1):
                    for x in range(plan.gw - w0 + 1):
                        if plan.fits(x, y, w0, h0):
                            return x, y, w0, h0
        return None

    def _work_triangle_bonus(self, plan: GridPlan) -> float:
        """Return a bonus based on the kitchen work triangle.

        The function measures the distances between the centers of the sink
        (``SINK``), cooktop (``COOK``) and refrigerator (``REF``).  If any of
        these appliances are missing, the bonus is ``0.0``.  Otherwise the three
        pairwise distances are summed to obtain the triangle's perimeter.  A
        perimeter in the range of 4–7 meters (inclusive) is considered optimal
        and yields a bonus of ``1.0``; values outside this range return ``0.0``.
        """

        nodes = ['SINK', 'COOK', 'REF']
        centers = []
        for code in nodes:
            comps = list(components_by_code(plan, code))
            if not comps:
                return 0.0
            x, y, w, h, _ = comps[0]
            # convert appliance center to meters
            centers.append(((x + w / 2.0) * plan.cell,
                            (y + h / 2.0) * plan.cell))

        # compute side lengths in meters
        d1 = sqrt((centers[0][0] - centers[1][0]) ** 2 +
                  (centers[0][1] - centers[1][1]) ** 2)
        d2 = sqrt((centers[1][0] - centers[2][0]) ** 2 +
                  (centers[1][1] - centers[2][1]) ** 2)
        d3 = sqrt((centers[2][0] - centers[0][0]) ** 2 +
                  (centers[2][1] - centers[0][1]) ** 2)
        perimeter = d1 + d2 + d3
        return 1.0 if 4.0 <= perimeter <= 7.0 else 0.0

    def _adjacency_score(self, plan: GridPlan) -> Tuple[float, bool]:
        """Return adjacency score and validity flag for kitchen layouts."""
        A = {
            'SINK': {'REF': (+1.0, 5), 'COOK': (+1.0, 4)},
            'COOK': {'REF': (+0.5, 5)},
        }

        def boxes(code):
            comps = components_by_code(plan, code)
            if comps:
                x, y, w, h, _ = comps[0]
                return (x, y, x + w - 1, y + h - 1)
            return None

        score = 0.0
        bsink = boxes('SINK')
        if bsink:
            sx0, sy0, sx1, sy1 = bsink
            for other, (wt, thresh) in A['SINK'].items():
                b = boxes(other)
                if not b:
                    continue
                ox0, oy0, ox1, oy1 = b
                dx = max(0, max(ox0 - sx1, sx0 - ox1))
                dy = max(0, max(oy0 - sy1, sy0 - oy1))
                dist = dx + dy
                if dist > thresh:
                    return 0.0, False
                score += wt * (1.0 / (1.0 + dist))

        bcook = boxes('COOK')
        if bcook:
            cx0, cy0, cx1, cy1 = bcook
            for other, (wt, thresh) in A.get('COOK', {}).items():
                b = boxes(other)
                if not b:
                    continue
                ox0, oy0, ox1, oy1 = b
                dx = max(0, max(ox0 - cx1, cx0 - ox1))
                dy = max(0, max(oy0 - cy1, cy0 - oy1))
                dist = dx + dy
                if dist > thresh:
                    return 0.0, False
                score += wt * (1.0 / (1.0 + dist))

        return score, True


# -----------------------
# Clearance merging
# -----------------------

def merge_clearances(rects:List[Tuple[int,int,int,int,str,str]])->List[Tuple[int,int,int,int,str,str]]:
    by_kind=defaultdict(list)
    for r in rects: by_kind[(r[4], r[5])].append(r)
    out=[]
    for key, group in by_kind.items():
        changed=True
        L=[list(r[:4]) for r in group]  # [x,y,w,h]
        while changed:
            changed=False
            i=0
            while i<len(L):
                x1,y1,w1,h1=L[i]; merged=False
                j=i+1
                while j<len(L):
                    x2,y2,w2,h2=L[j]
                    if rects_touch_or_overlap((x1,y1,w1,h1),(x2,y2,w2,h2)):
                        x=min(x1,x2); y=min(y1,y2)
                        X=max(x1+w1,x2+w2); Y=max(y1+h1,y2+h2)
                        x1,y1,w1,h1=x,y,X-x,Y-y
                        L[i]=[x1,y1,w1,h1]; L.pop(j); merged=True; changed=True
                    else:
                        j+=1
                if not merged: i+=1
        for x,y,w,h in L:
            out.append((x,y,w,h,key[0],key[1]))
    return out

def rects_touch_or_overlap(a,b)->bool:
    ax,ay,aw,ah=a; bx,by,bw,bh=b
    if ax+aw < bx or bx+bw < ax: return False
    if ay+ah < by or by+bh < ay: return False
    return True

def add_door_clearance(p: GridPlan, op: Openings, owner: str):
    """Mark clearance for a door defined by ``op`` onto ``p`` and return the
    mirrored rectangle on the opposite side of the doorway.

    The interior clearance is applied directly to ``p``.  The exterior
    rectangle is returned (but not recorded) in ``p``'s cell units so callers
    may convert it to metres and map it onto an adjacent plan if needed.
    """
    wall, start, width = op.door_span_cells()
    depth = p.meters_to_cells(op.swing_depth)
    pw = max(1, PATH_WIDTH_CELLS)
    if wall == WALL_BOTTOM:
        inside = (start, depth, width, pw)
        line = 0
        outside = (start, 2 * line - (depth + pw), width, pw)
    elif wall == WALL_TOP:
        inside = (start, p.gh - depth - pw, width, pw)
        line = p.gh
        outside = (start, 2 * line - (inside[1] + pw), width, pw)
    elif wall == WALL_LEFT:
        inside = (depth, start, pw, width)
        line = 0
        outside = (2 * line - (depth + pw), start, pw, width)
    else:  # WALL_RIGHT
        inside = (p.gw - depth - pw, start, pw, width)
        line = p.gw
        outside = (2 * line - (inside[0] + pw), start, pw, width)

    p.mark_clear(*inside, 'DOOR_CLEAR', owner)
    return outside

# -----------------------
# Bathroom arranger (very light – placeholder)
# -----------------------

def arrange_bathroom(
    Wm: float,
    Hm: float,
    rules: Dict,
    openings: Optional[Openings] = None,
    secondary_openings: Optional[Openings] = None,
    rng: Optional[random.Random] = None,
) -> GridPlan:
    """Generate a bathroom layout honouring clearance rules.

    The function is intentionally lightweight – its goal is simply to place the
    four common fixtures (tub, shower, water closet and lavatory) while
    respecting the minimum clearances encoded in ``rules``.  It is **not** an
    optimiser; if the room is too small to satisfy the hard minimums a partially
    filled plan may be returned.  ``openings`` describes door and window
    positions to honour when reserving door clearances.
    """

    def _intersects_clear(p: GridPlan, x: int, y: int, w: int, h: int) -> bool:
        for cx, cy, cw, ch, *_ in p.clearzones:
            if (x < cx + cw and x + w > cx and y < cy + ch and y + h > cy):
                return True
        return False

    def _place_with_front(p: GridPlan, x: int, y: int, w: int, h: int,
                          code: str, front_m: float, against_top: bool) -> bool:
        """Place an element and reserve the required front clearance."""
        if not p.fits(x, y, w, h) or _intersects_clear(p, x, y, w, h):
            return False
        fc = p.meters_to_cells(front_m)
        if fc > 0:
            if against_top:
                fx, fy, fw, fh = x, y - fc, w, fc
            else:
                fx, fy, fw, fh = x, y + h, w, fc
            if fy < 0 or fy + fh > p.gh:
                return False
            if not p.fits(fx, fy, fw, fh) or _intersects_clear(p, fx, fy, fw, fh):
                return False
        if not p.fits(x, y, w, h):
            return False
        p.place(x, y, w, h, code)
        if fc > 0:
            p.mark_clear(fx, fy, fw, fh, 'FRONT', code)
        return True

    def _force_place(p: GridPlan, w: int, h: int, code: str) -> bool:
        """Place ``code`` somewhere on ``p`` even if rules cannot be satisfied."""
        for sw in range(w, 0, -1):
            for sh in range(h, 0, -1):
                for y in range(0, p.gh - sh + 1):
                    for x in range(0, p.gw - sw + 1):
                        if p.fits(x, y, sw, sh):
                            p.place(x, y, sw, sh, code)
                            return True
        return False

    units = rules.get('units', {})
    in_m = units.get('IN_M', 0.0254)

    fx = rules.get('fixtures', {})
    clear = {
        'lav_side': fx.get('lavatory', {}).get('side_clear_to_wall_m', {}).get('min', 0.508),
        'lav_front': fx.get('lavatory', {}).get('front_clear_to_opposite_m', {}).get('min', 0.610),
        'lav_to_fixture': fx.get('lavatory', {}).get('to_adjacent_fixture_edge_m', {}).get('min', 0.406),
        'wc_side': fx.get('water_closet', {}).get('center_to_side_obstruction_m', {}).get('min', 0.406),
        'wc_front': fx.get('water_closet', {}).get('front_clear_to_opposite_m', {}).get('min', 0.610),
        'tub_front': fx.get('bathtub', {}).get('front_clear_to_opposite_wall_m', {}).get('min', 0.762),
        'shr_front': fx.get('bathtub', {}).get('entry_front_clear_m', 0.762),
    }

    tub_lengths = sorted(fx.get('bathtub', {}).get('common_lengths_m', [1.5]))
    shr_opts = sorted(
        fx.get('shower', {}).get('stall_nominal_sizes_in', [{'w': 36, 'd': 36}]),
        key=lambda s: (s.get('w', 0) * s.get('d', 0), s.get('w', 0), s.get('d', 0))
    )

    p = GridPlan(Wm, Hm)
    for op, owner in ((openings, 'DOOR'), (secondary_openings, 'LIVING_DOOR')):
        if op:
            dx, dy, dw, dh = op.door_rect_cells()
            if p.fits(dx, dy, dw, dh):
                p.place(dx, dy, dw, dh, 'DOOR')
            ext = add_door_clearance(p, op, owner)
            setattr(op, 'ext_rect', ext)

    # Attempt to place each fixture individually; warn and force placement when needed.

    for tub_len in tub_lengths:
        tw = p.meters_to_cells(tub_len)
        td = p.meters_to_cells(0.75)
        if _place_with_front(p, 0, p.gh - td, tw, td, 'TUB', clear['tub_front'], True):
            break

    shr_dims = [
        (p.meters_to_cells(s.get('w', 36) * in_m),
         p.meters_to_cells(s.get('d', 36) * in_m))
        for s in shr_opts
    ]
    shr_placed = False
    for sw, sd in shr_dims:
        if _place_with_front(p, max(0, p.gw - sw), p.gh - sd, sw, sd,
                             'SHR', clear['shr_front'], True):
            shr_placed = True
            break
    if not shr_placed:
        warnings.warn('Shower could not be placed with required clearances', UserWarning)
        sw, sd = shr_dims[0]
        _force_place(p, sw, sd, 'SHR')

    ww = p.meters_to_cells(clear['wc_side'] * 2)
    wd = p.meters_to_cells(0.76)
    wc_placed = _place_with_front(p, 0, 0, ww, wd, 'WC', clear['wc_front'], False)
    if not wc_placed:
        warnings.warn('Water closet could not be placed with required clearances', UserWarning)
        _force_place(p, ww, wd, 'WC')

    lw = p.meters_to_cells(clear['lav_side'] * 2)
    ld = p.meters_to_cells(0.6)
    gap_req = p.meters_to_cells(clear['lav_to_fixture'])
    lav_placed = _place_with_front(p, p.gw - lw, 0, lw, ld, 'LAV', clear['lav_front'], False)
    if not lav_placed:
        warnings.warn('Lavatory could not be placed with required clearances', UserWarning)
        _force_place(p, lw, ld, 'LAV')
    elif wc_placed and (p.gw - ww - lw < gap_req):
        warnings.warn('Lavatory clearance to adjacent fixture not met', UserWarning)

    return p


def arrange_livingroom(
    Wm: float,
    Hm: float,
    rules: Dict,
    openings: Optional[Openings] = None,
    rng: Optional[random.Random] = None,
) -> GridPlan:
    """Generate a simple living room layout honouring size/clearance rules.

    The routine mirrors :func:`arrange_bathroom` in spirit: furniture pieces
    are attempted one at a time and skipped if they do not fit.  Clearances for
    traffic lanes or furniture spacing are reserved using ``GridPlan``'s
    ``mark_clear`` method.  A rug is recorded as a clearzone so that other
    elements may overlap it.
    """

    def _intersects_clear(p: GridPlan, x: int, y: int, w: int, h: int) -> bool:
        for cx, cy, cw, ch, *_ in p.clearzones:
            if (x < cx + cw and x + w > cx and y < cy + ch and y + h > cy):
                return True
        return False

    def _place_with_clear(p: GridPlan, x: int, y: int, w: int, h: int,
                           code: str, front_m: float = 0.0,
                           side_m: float = 0.0, against_top: bool = True) -> bool:
        """Place ``code`` at ``(x,y,w,h)`` and reserve front/side clearances."""
        if not p.fits(x, y, w, h) or _intersects_clear(p, x, y, w, h):
            return False
        fc = p.meters_to_cells(front_m) if front_m > 0 else 0
        sc = p.meters_to_cells(side_m) if side_m > 0 else 0
        # front clearance
        if fc > 0:
            if against_top:
                fx, fy, fw, fh = x, y - fc, w, fc
            else:
                fx, fy, fw, fh = x, y + h, w, fc
            if fy < 0 or fy + fh > p.gh:
                return False
            if not p.fits(fx, fy, fw, fh) or _intersects_clear(p, fx, fy, fw, fh):
                return False
        # side clearances (both sides)
        sides = []
        if sc > 0:
            sides = [(x - sc, y, sc, h), (x + w, y, sc, h)]
            for sx, sy, sw, sh in sides:
                if sx < 0 or sx + sw > p.gw:
                    return False
                if not p.fits(sx, sy, sw, sh) or _intersects_clear(p, sx, sy, sw, sh):
                    return False
        p.place(x, y, w, h, code)
        if fc > 0:
            p.mark_clear(fx, fy, fw, fh, 'FRONT', code)
        if sc > 0:
            for sx, sy, sw, sh in sides:
                p.mark_clear(sx, sy, sw, sh, 'SIDE', code)
        return True

    furn = rules.get('furniture_size_ranges', {})
    tables = rules.get('tables_and_coffee', {})
    clear = rules.get('clearances', {})

    # Basic furniture dimensions
    sofa_len = furn.get('sofas', {}).get('length_m_range', [2.0])[0]
    sofa_dep = furn.get('sofas', {}).get('depth_m_range', [0.9])[0]
    side_w = furn.get('end_tables', {}).get('width_m_range', [0.3])[0]
    side_d = furn.get('end_tables', {}).get('depth_m_range', [0.4])[0]
    chair_w = furn.get('straight_chairs', {}).get('width_m_range', [0.5])[0]
    chair_d = furn.get('straight_chairs', {}).get('depth_m_range', [0.5])[0]

    sofa_to_coffee = tables.get('typical_distance_sofa_front_to_coffee_edge_m', [0.45])[0]
    lane = max(
        clear.get('traffic_lane_min_m', 1.0),
        tables.get('aisle_between_coffee_and_chairs_min_m', 1.016),
    )

    coffee_w = 1.0  # default size if not specified in rules
    coffee_d = 0.5

    p = GridPlan(Wm, Hm)
    if openings:
        # ``Openings`` seeds each plan with a placeholder window.  For the
        # living room we don't want any implicit windows—only those the user
        # explicitly defined or that come from rule configuration.  When the
        # default placeholder is present (two entries with the second marked by
        # a negative wall index) clear the list so no window is assumed.
        if len(openings.windows) == 2 and openings.windows[1][0] < 0:
            openings.windows = []
        dx, dy, dw, dh = openings.door_rect_cells()
        if p.fits(dx, dy, dw, dh):
            p.place(dx, dy, dw, dh, 'DOOR')
        add_door_clearance(p, openings, 'DOOR')

    # Sofa against top wall, centered
    sw = p.meters_to_cells(sofa_len)
    sd = p.meters_to_cells(sofa_dep)
    sx = max(0, (p.gw - sw) // 2)
    sofa_placed = _place_with_clear(p, sx, 0, sw, sd, 'SOFA', against_top=True)

    # Side tables, one on each side of sofa
    stw = p.meters_to_cells(side_w)
    std = p.meters_to_cells(side_d)
    if sofa_placed:
        _place_with_clear(p, sx - stw, 0, stw, std, 'STAB', against_top=True)
        _place_with_clear(p, sx + sw, 0, stw, std, 'STAB', against_top=True)

    # Coffee table in front of sofa
    cw = p.meters_to_cells(coffee_w)
    cd = p.meters_to_cells(coffee_d)
    gap = p.meters_to_cells(sofa_to_coffee)
    cx = max(0, (p.gw - cw) // 2)
    cy = sd + gap
    coffee_placed = _place_with_clear(p, cx, cy, cw, cd, 'CTAB', front_m=lane, against_top=False)

    # Pair of chairs facing the sofa beyond the coffee table
    if coffee_placed:
        chw = p.meters_to_cells(chair_w)
        chd = p.meters_to_cells(chair_d)
        aisle = p.meters_to_cells(lane)
        gap_between = p.meters_to_cells(0.5)
        total_w = 2 * chw + gap_between
        chy = cy + cd + aisle
        if chy + chd <= p.gh and total_w <= p.gw:
            chx = max(0, (p.gw - total_w) // 2)
            _place_with_clear(p, chx, chy, chw, chd, 'CHAR', against_top=False)
            _place_with_clear(p, chx + chw + gap_between, chy, chw, chd, 'CHAR', against_top=False)

    # Rug covering the conversation area (recorded as a clearzone)
    if coffee_placed:
        rx = max(0, min(sx, cx) - p.meters_to_cells(0.1))
        rw = min(p.gw - rx, max(sx + sw, cx + cw) - rx + p.meters_to_cells(0.2))
        ry = max(0, sd)
        rd = min(p.gh - ry, (cy + cd) - ry)
        p.clearzones.append((rx, ry, rw, rd, 'RUG', 'RUG'))

    return p

# -----------------------
# UI – Generate view
# -----------------------

PALETTE = {
    'BED':'#2eea98','BST':'#cfcfcf','WRD':'#ffa54a','DRS':'#ffd84a',
    'DESK':'#8ad1ff','TVU':'#b7b7b7','CLEAR':'#6fb6ff','DOOR':'#8b4513'
}

# Living room elements (complimenting tones)
PALETTE.setdefault('SOFA', '#86e3ce')
PALETTE.setdefault('CTAB', '#d0f4ea')  # coffee table
PALETTE.setdefault('STAB', '#a1c6ea')  # side table
PALETTE.setdefault('RUG',  '#f6d186')
PALETTE.setdefault('CHAR', '#c4b7cb')  # chair

# Dining elements (color family distinct from LR)
PALETTE.setdefault('DTAB',   '#ffb3c1')  # dining table
PALETTE.setdefault('DCHAIR', '#ffc9de')  # dining chair
PALETTE.setdefault('DSIDE',  '#ffd6a5')  # sideboard / buffet

# Bathroom elements
PALETTE.setdefault('WC',  '#ffb4b4')
PALETTE.setdefault('SHR', '#a3d5ff')
PALETTE.setdefault('TUB', '#fff3b0')
PALETTE.setdefault('LAV', '#c5e1a5')

# Kitchen elements
PALETTE.setdefault('SINK',  '#add8e6')
PALETTE.setdefault('COOK',  '#ffcccb')
PALETTE.setdefault('REF',   '#b0e0e6')
PALETTE.setdefault('DW',    '#ffe4b5')
PALETTE.setdefault('ISLN',  '#e6e6fa')
PALETTE.setdefault('BASE',  '#deb887')
PALETTE.setdefault('WALL',  '#f5deb3')
PALETTE.setdefault('HOOD',  '#d8bfd8')
PALETTE.setdefault('OVEN',  '#ffc0cb')
PALETTE.setdefault('MICRO', '#dda0dd')

ITEM_LABELS = {
    'BED': 'Bed',
    'BST': 'Night Table',
    'WRD': 'Wardrobe',
    'DRS': 'Dresser',
    'DESK': 'Desk',
    'TVU': 'TV Unit',
    'WC': 'Toilet',
    'SHR': 'Shower',
    'TUB': 'Tub',
    'LAV': 'Lavatory',
    'CLEAR': 'Clearance',
    'SOFA': 'Sofa',
    'CTAB': 'Coffee Table',
    'STAB': 'Side Table',
    'RUG': 'Rug',
    'CHAR': 'Chair',
    'DTAB': 'Dining Table',
    'DCHAIR': 'Dining Chair',
    'DSIDE': 'Sideboard',
    'SINK': 'Sink',
    'COOK': 'Cooktop',
    'REF': 'Refrigerator',
    'DW': 'Dishwasher',
    'ISLN': 'Island',
    'BASE': 'Base Cabinet',
    'WALL': 'Wall Cabinet',
    'HOOD': 'Range Hood',
    'OVEN': 'Oven',
    'MICRO': 'Microwave',
    'CHEST': 'Chest'
}


WALL_COLOR='#000000'

# Use a brighter fill so doors stand out from black walls and keep window
# fill distinct.  Expose them as module-level constants so tests can refer to
# them directly.
DOOR_FILL='#ff8c00'
WINDOW_FILL='#95c8ff'

HUMAN1_COLOR='#ff6262'
HUMAN2_COLOR='#ffdd55'

class GenerateView:
    BED_CODES = {'WRD', 'DRS', 'DESK', 'TVU', 'BST', 'BED'}
    BATH_CODES = {'WC', 'SHR', 'TUB', 'LAV'}
    LIV_CODES = {'SOFA', 'CTAB', 'STAB', 'RUG', 'CHAR', 'DTAB', 'DCHAIR', 'DSIDE'}
    KITCH_CODES = {
        'SINK', 'COOK', 'REF', 'DW', 'ISLN',
        'BASE', 'WALL', 'HOOD', 'OVEN', 'MICRO'
    }
    ALL_FURN_CODES = BED_CODES | BATH_CODES | LIV_CODES | KITCH_CODES
    REQUIRED_FURNITURE = {
        'bed_plan': {'BED'},
        'kitch_plan': {'SINK'},
    }

    def __init__(
        self,
        root: tk.Misc,
        Wm: float,
        Hm: float,
        bed_key: Optional[str],
        room_label: str = 'Bedroom',
        bath_dims: Optional[Tuple[float, float]] = None,
        liv_dims: Optional[Tuple[float, float]] = None,
        kitch_dims: Optional[Tuple[float, float]] = None,
        pack_side=tk.LEFT,
        on_back=None,
    ):
        self.root=root; self.on_back=on_back
        self.room_label = room_label

        self.bed_Wm = Wm; self.bed_Hm = Hm
        self.bath_dims = bath_dims
        if bath_dims:
            bw, bh = bath_dims
            self.bath_Wm = bw; self.bath_Hm = bh
        else:
            self.bath_Wm = self.bath_Hm = 0.0

        self.liv_dims = liv_dims
        if liv_dims:
            lw, lh = liv_dims
            self.liv_Wm = lw; self.liv_Hm = lh
        else:
            self.liv_Wm = self.liv_Hm = 0.0
        self._validate_living_dims()

        self.kitch_dims = kitch_dims
        if kitch_dims:
            kw, kh = kitch_dims
            self.kitch_Wm = kw; self.kitch_Hm = kh
        else:
            self.kitch_Wm = self.kitch_Hm = 0.0

        # Maintain separate plans for bedroom, bathroom, living room and kitchen.
        self.bed_plan = GridPlan(self.bed_Wm, self.bed_Hm)
        self.bath_plan = GridPlan(self.bath_Wm, self.bath_Hm) if bath_dims else None
        self.liv_plan = GridPlan(self.liv_Wm, self.liv_Hm) if self.liv_dims else None
        self.kitch_plan = GridPlan(self.kitch_Wm, self.kitch_Hm) if self.kitch_dims else None

        # Overall dims remain fixed for combined plan
        self.Wm = max(
            self.bed_Wm + (self.bath_Wm if bath_dims else 0),
            self.liv_Wm if self.liv_dims else 0,
            self.kitch_Wm if self.kitch_dims else 0,
        )
        self.Hm = max(
            self.bed_Hm,
            self.bath_Hm if bath_dims else 0,
        ) + (
            self.liv_Hm if self.liv_dims else 0
        ) + (
            self.kitch_Hm if self.kitch_dims else 0
        )

        # Inform users if dimensions were auto-adjusted
        if getattr(self, "liv_auto_adjusted", []):
            try:
                from tkinter import messagebox
                messagebox.showinfo(
                    "Adjusted dimensions", "\n".join(self.liv_auto_adjusted)
                )
            except Exception:
                print("Adjusted dimensions:", "; ".join(self.liv_auto_adjusted))

        # Combined plan used by legacy helpers (_cell_rect etc.)
        self.plan = GridPlan(self.Wm, self.Hm)
        self._combine_plans()

        self.bed_openings = Openings(GridPlan(self.bed_Wm, self.bed_Hm))
        # Bedroom doors retain the previous swing depth
        self.bed_openings.swing_depth = 0.60
        self.bed_openings.door_wall = WALL_RIGHT
        self.openings = self.bed_openings  # maintain compatibility for bedroom ops
        self.bath_openings = (
            Openings(self.bath_plan) if bath_dims else None
        )
        if self.bath_openings:
            self.bath_openings.swing_depth = CELL_M
        self.bath_liv_openings = (
            Openings(self.bath_plan) if bath_dims and liv_dims else None
        )
        if self.bath_liv_openings:
            self.bath_liv_openings.swing_depth = CELL_M
        self.liv_openings = (
            Openings(self.liv_plan) if liv_dims else None
        )
        if self.liv_openings:
            self.liv_openings.swing_depth = 0.60
        self.bed_key=None if bed_key=='Auto' else bed_key
        self.weights, self.mlp, self.transformer, self.ae, self.cnn, self.rnn, self.gan, self.ensemble = rehydrate_from_feedback()
        self.rng=random.Random()
        self.container=ttk.Frame(root); self.container.pack(side=pack_side, fill=tk.BOTH, expand=True)
        tb=ttk.Frame(self.container); tb.pack(fill=tk.X)
        ttk.Button(tb, text='← Back', command=self._go_back).pack(side=tk.LEFT, padx=6, pady=4)
        ttk.Label(tb, text=self.room_label, font=('SF Pro Text', 12, 'bold')).pack(side=tk.LEFT, padx=6)
        self.canvas=tk.Canvas(self.container, bg='#ffffff', highlightthickness=0, cursor='hand2')
        self.canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.zoom_factor = tk.DoubleVar(value=1.0)
        self.grid_overlay = ColumnGridOverlay(self.canvas)
        # Floating legend popovers for openings
        self.popover = LegendPopover(self.canvas)
        self.opening_item_info = {}

        # Tooltip elements are managed via the 'tooltip' tag

        self.sidebar=ttk.Frame(self.container, width=360, padding=10)
        self.sidebar.pack(side=tk.RIGHT, fill=tk.Y)
        self._build_sidebar()
        self.canvas.bind('<Configure>', lambda e: self._draw())
        
        # (keyboard bindings are set below via canvas.bind + root.bind_all)

        
        # Drag state
        self.drag_item=None
        # Undo/redo stacks
        self.undo_stack = []
        self.redo_stack = []
        # Selection state for keyboard ops and double-click lock
        self.selected=None
        self.selected_locked=False
        self.canvas.bind('<Button-1>', self._on_down)
        self.canvas.bind('<B1-Motion>', self._on_drag)
        self.canvas.bind('<ButtonRelease-1>', self._on_up)
        
        # Double-click toggles sticky selection (lock/unlock)
        self.canvas.bind('<Double-Button-1>', self._on_double_click)

        # Also bind arrows directly on the canvas (widget-level gets priority when canvas has focus)
        for seq in ('<KeyPress-Left>', '<Left>',
                    '<KeyPress-Right>', '<Right>',
                    '<KeyPress-Up>', '<Up>',
                    '<KeyPress-Down>', '<Down>'):
            self.canvas.bind(seq, self._on_arrow_rotate)
        self.canvas.bind('r', self._on_arrow_rotate)


        # --- Keyboard bindings (global) ---
        # Bind to both KeyPress and plain sequences to avoid widget focus swallowing arrows
        for seq in ('<KeyPress-Left>', '<Left>',
                    '<KeyPress-Right>', '<Right>',
                    '<KeyPress-Up>', '<Up>',
                    '<KeyPress-Down>', '<Down>'):
            self.root.bind_all(seq, self._on_arrow_rotate)

        # Ctrl/Cmd + Arrow = nudge by 1 cell
        for seq, cb in (
            ('<Control-Left>',  self._on_ctrl_left),
            ('<Control-Right>', self._on_ctrl_right),
            ('<Control-Up>',    self._on_ctrl_up),
            ('<Control-Down>',  self._on_ctrl_down),
            ('<Command-Left>',  self._on_ctrl_left),
            ('<Command-Right>', self._on_ctrl_right),
            ('<Command-Up>',    self._on_ctrl_up),
            ('<Command-Down>',  self._on_ctrl_down),
        ):
            self.root.bind_all(seq, cb)

        # 'r' also rotates
        self.root.bind_all('r', self._on_arrow_rotate)

        # Sims
        self.sim_timer=None; self.sim2_timer=None
        self.sim_path=[]; self.sim_index=0; self.sim_poly=[]
        self.sim2_path=[]; self.sim2_index=0; self.sim2_poly=[]
        # Human blocks
        self.human_id=None; self.human2_id=None
        # batch feedback vars filled in _build_sidebar
        # Kick off initial solve after all widgets are set up so that
        # variables and geometry are ready before the heavy work runs.
        self.root.after_idle(self._solve_and_draw)

    def _validate_living_dims(self):
        """Ensure living room dimensions allow room connections."""
        if not self.liv_dims:
            return
        self.liv_auto_adjusted = []
        min_width = self.bed_Wm + self.bath_Wm
        if self.liv_dims[0] < min_width:
            self.liv_dims = (min_width, self.liv_dims[1])
            self.liv_Wm = min_width
            self.liv_auto_adjusted.append(
                f"Living room width increased to {min_width:.2f} m to span bedroom and bathroom."
            )
        required = max(0.60, CELL_M)
        if self.liv_dims[1] < required:
            self.liv_dims = (self.liv_dims[0], required)
            self.liv_Hm = required
            self.liv_auto_adjusted.append(
                f"Living room depth increased to {required:.2f} m for door clearance."
            )

    # ----------------- sidebar

    def _build_sidebar(self):
        # Clear any existing widgets so sidebar can be rebuilt cleanly
        for child in self.sidebar.winfo_children():
            child.destroy()

        ttk.Button(
            self.sidebar,
            text='↻ Generate',
            style='Primary.TButton',
            command=self._apply_batch_and_generate,
        ).pack(fill=tk.X)

        if 'bedroom' in self.room_label.lower():
            ttk.Label(self.sidebar, text='Furniture', font=('SF Pro Text', 13, 'bold')).pack(anchor='w', pady=(8, 2))
            fb = ttk.Frame(self.sidebar)
            fb.pack(fill=tk.X, pady=(0, 2))
            self.furn_kind = tk.StringVar(value='TVU')
            self.auto_place = tk.BooleanVar(value=True)
            ttk.Combobox(
                fb,
                textvariable=self.furn_kind,
                values=['TVU', 'DESK', 'DRS_4FT', 'CHEST_SM', 'WRD_S_210', 'WRD_H_180'],
                width=12,
                state='readonly',
            ).pack(side=tk.LEFT)
            ttk.Checkbutton(fb, text='auto', variable=self.auto_place).pack(side=tk.LEFT, padx=6)
            b2 = ttk.Frame(self.sidebar)
            b2.pack(fill=tk.X)
            ttk.Button(b2, text='Add', command=self._add_furniture).pack(side=tk.LEFT, expand=True, fill=tk.X)
            ttk.Button(b2, text='Remove', command=self._remove_furniture).pack(side=tk.LEFT, expand=True, fill=tk.X, padx=6)

            # Rules
            self.force_bst_pair = tk.BooleanVar(value=True)
            ttk.Checkbutton(
                self.sidebar,
                text='Force bedside tables (pair)',
                variable=self.force_bst_pair,
            ).pack(anchor='w', pady=(6, 2))
        else:
            self.force_bst_pair = tk.BooleanVar(value=False)

        ttk.Button(self.sidebar, text='▶ Simulate Circulation', command=self._simulate_one).pack(fill=tk.X, pady=(8, 0))
        ttk.Button(self.sidebar, text='▶▶ Simulate Two Humans', command=self._simulate_two).pack(fill=tk.X, pady=(4, 6))
        ttk.Button(
            self.sidebar,
            text='Run circulation sim (scribble)',
            command=self.simulate_circulation,
        ).pack(fill=tk.X, pady=(0, 8))

        ttk.Label(self.sidebar, text='Zoom').pack(anchor='w', pady=(6, 2))
        ttk.Scale(
            self.sidebar, variable=self.zoom_factor, from_=0.5, to=1.0, orient='horizontal'
        ).pack(fill=tk.X)
        if hasattr(self.zoom_factor, 'trace_add'):
            self.zoom_factor.trace_add('write', lambda *args: self._draw())

        ttk.Button(self.sidebar, text='Export PNG', command=self._export_png).pack(fill=tk.X, pady=(6, 0))
        self.status = tk.StringVar(value='')
        ttk.Label(self.sidebar, textvariable=self.status, wraplength=320).pack(anchor='w', pady=(6, 0))

        if self.force_bst_pair is not None and hasattr(self.force_bst_pair, 'trace_add'):
            self.force_bst_pair.trace_add('write', lambda *args: self._solve_and_draw())
    def _go_back(self):
        try: self.container.destroy()
        except: pass
        if callable(self.on_back): self.on_back()

    # ----------------- solve / openings

    def _apply_openings_from_ui(self):
        """Validate openings and keep inter-room doors aligned."""
        if self.liv_dims and self.bed_openings.door_wall != WALL_RIGHT:
            self.status.set('Bedroom door must be on shared wall.')
            return False

        if self.bath_dims and self.liv_dims:
            if not self.bath_liv_openings:
                base = self.bath_plan or GridPlan(*self.bath_dims)
                self.bath_liv_openings = Openings(base)
                self.bath_liv_openings.swing_depth = CELL_M
            def _shared_wall(b: GridPlan, l: GridPlan) -> int:
                bx0, by0 = b.x_offset, b.y_offset
                bx1, by1 = bx0 + b.gw, by0 + b.gh
                lx0, ly0 = l.x_offset, l.y_offset
                lx1, ly1 = lx0 + l.gw, ly0 + l.gh
                if bx1 == lx0 and max(by0, ly0) < min(by1, ly1):
                    return WALL_RIGHT
                if lx1 == bx0 and max(by0, ly0) < min(by1, ly1):
                    return WALL_LEFT
                if by1 == ly0 and max(bx0, lx0) < min(bx1, lx1):
                    return WALL_BOTTOM
                if ly1 == by0 and max(bx0, lx0) < min(bx1, lx1):
                    return WALL_TOP
                return WALL_BOTTOM
            self.bath_liv_openings.door_wall = _shared_wall(self.bath_plan, self.liv_plan)
            if not getattr(self, 'liv_bath_openings', None):
                base = self.liv_plan or GridPlan(*self.liv_dims)
                self.liv_bath_openings = Openings(base)
                self.liv_bath_openings.swing_depth = CELL_M
            self.liv_bath_openings.door_wall = opposite_wall(self.bath_liv_openings.door_wall)
            self.liv_bath_openings.door_center = self.bath_liv_openings.door_center
            self.liv_bath_openings.door_width = self.bath_liv_openings.door_width
            self.liv_bath_openings.swing_depth = self.bath_liv_openings.swing_depth
        else:
            self.liv_bath_openings = None
        return True
    def _apply_batch_and_generate(self):
        # (1) snapshot only if you want to keep a LOCKED item; otherwise clear
        sticky = []
        bath_sticky = []
        liv_sticky = []
        if getattr(self, 'selected_locked', False) and getattr(self, 'selected', None):
            x, y, w, h = self.selected['rect']
            code = self.selected['code']
            wall = self._infer_wall(x, y, w, h)
            bed_gw = getattr(self.bed_plan, 'gw', 0)
            bath_gw = self.bath_plan.gw if getattr(self, 'bath_plan', None) else 0
            if getattr(self, 'liv_plan', None) and x >= bed_gw + bath_gw:
                liv_sticky.append((code, x - bed_gw - bath_gw, y, w, h, wall))
            elif getattr(self, 'bath_plan', None) and x >= bed_gw:
                bath_sticky.append((code, x - bed_gw, y, w, h, wall))
            else:
                sticky.append((code, x, y, w, h, wall))
        self._sticky_items = sticky  # only keep the locked item (if any)
        self._sticky_bath_items = bath_sticky
        self._sticky_liv_items = liv_sticky
        # (2) RUN THE SOLVER to regenerate
        if not self._apply_openings_from_ui():
            return
        self._solve_and_draw()
        self.status.set('Rooms regenerated.')

    def _add_door_clearance(self, p: GridPlan, owner: str, openings=None):
        """Mark clearance for bedroom or bathroom doors on ``p`` and return the
        exterior rectangle if one is produced.

        The rectangle is expressed in ``p``'s cell units. If ``openings`` is not
        supplied, it is inferred based on whether ``p`` corresponds to
        ``self.bath_plan`` or ``self.bed_plan``.
        """
        if openings is None:
            openings = self.bath_openings if p is self.bath_plan else self.bed_openings
        if openings:
            return add_door_clearance(p, openings, owner)
        return None

    def _solve_and_draw(self):
        if self.sim_timer: self.root.after_cancel(self.sim_timer); self.sim_timer=None
        if self.sim2_timer: self.root.after_cancel(self.sim2_timer); self.sim2_timer=None
        self.sim_path=[]; self.sim_poly=[]; self.sim2_path=[]; self.sim2_poly=[]
        if not self._apply_openings_from_ui():
            self.bed_plan = None
            self.bath_plan = None
            self.plan = GridPlan(self.bed_Wm, self.bed_Hm)
            self._draw()
            return
        bed_wall, _, _ = self.bed_openings.door_span_cells()
        bath_ok = True
        # Preserve previous plan so we can restore it on failure
        prev_plan = getattr(self, 'plan', None)
        if self.bath_dims:
            bath_wall, _, _ = self.bath_openings.door_span_cells()
            if bath_wall != WALL_LEFT:
                self.status.set('Bathroom door must be on shared wall.')
                bath_ok = False
        bed_plan=GridPlan(self.bed_Wm,self.bed_Hm)

        # Pre-mark door clearances on the initial plan so the solver is aware
        # of keep-out zones.  Both the interior clearance and the exterior
        # rectangle (returned) are recorded.
        ext = self._add_door_clearance(bed_plan, 'DOOR', self.bed_openings)
        if ext:
            bed_plan.mark_clear(*ext, 'DOOR_CLEAR', 'DOOR')

        shared_ext = None
        if self.bath_dims and self.bath_openings:
            shared_op = Openings(bed_plan)
            shared_op.door_wall = WALL_RIGHT
            shared_op.door_center = self.bath_openings.door_center
            shared_op.door_width = self.bath_openings.door_width
            shared_op.swing_depth = self.bath_openings.swing_depth
            shared_ext = self._add_door_clearance(bed_plan, 'BATHROOM_DOOR', shared_op)
            if shared_ext:
                bed_plan.mark_clear(*shared_ext, 'DOOR_CLEAR', 'BATHROOM_DOOR')
        solver=BedroomSolver(
            bed_plan,
            self.bed_openings,
            self.bed_key,
            random.Random(),
            load_weights(),
            self.mlp,
            self.transformer,
            ensemble=getattr(self, 'ensemble', None),
            pref=None,
            force_bst_pair=bool(getattr(self, 'force_bst_pair', tk.BooleanVar(value=False)).get())
        )
        solver.bath_openings = getattr(self, 'bath_openings', None)
        best, meta = solver.run()
        if not isinstance(best, GridPlan):
            # If bedroom solver fails, keep the previous plan and inform the user
            if meta.get('status') == 'no_bed':
                self.status.set('No bed placed; adjust parameters.')
                if prev_plan is not None:
                    self.plan = prev_plan
                return
            else:
                self.status.set('No arrangement found (adjust door/windows).')
                if prev_plan is not None:
                    self.plan = prev_plan
                self._draw()
                return

        # overlay sticky items (if any), preserving positions & clearances
        sticky = getattr(self, '_sticky_items', [])
        if sticky:
            FRONT_REC_DEFAULT = {'WRD': 0.80, 'DRS': 0.90, 'DESK': 0.90}
            for (code,x,y,w,h,wall) in sticky:
                # force place: clear any conflicts then place
                best.clear(x,y,w,h)
                best.place(x,y,w,h, code)
                # reapply front clearance if we have a default
                fc_m = FRONT_REC_DEFAULT.get(code, 0.0)
                if fc_m > 0.0:
                    fc = best.meters_to_cells(fc_m)
                    if code == 'WRD' and fc == 4:
                        fc = 2
                    clear_w, clear_x = w, x
                    clear_h, clear_y = h, y
                    if code == 'WRD' and wall in (0, 2) and w > 1:
                        clear_w = w - 1
                        clear_x = x + 0.5
                    elif code == 'DRS':
                        fc = 2 if fc == 4 else 5 if fc == 3 else fc
                        rect_w = 5
                        rect_d = 2
                        clear_x = x + (w - rect_w) / 2
                        clear_y = y + (h - rect_w) / 2
                        if wall == 0:
                            best.mark_clear(clear_x, y + h, rect_w, rect_d, 'FRONT', code)
                        elif wall == 2:
                            best.mark_clear(clear_x, y - rect_d, rect_w, rect_d, 'FRONT', code)
                        elif wall == 3:
                            best.mark_clear(x + w, clear_y, rect_d, rect_w, 'FRONT', code)
                        elif wall == 1:
                            best.mark_clear(x - rect_d, clear_y, rect_d, rect_w, 'FRONT', code)
                        continue
                    if wall == 0:
                        best.mark_clear(clear_x, y + h, clear_w, fc, 'FRONT', code)
                    elif wall == 2:
                        best.mark_clear(clear_x, y - fc, clear_w, fc, 'FRONT', code)
                    elif wall == 3:
                        best.mark_clear(x + w, clear_y, fc, clear_h, 'FRONT', code)
                    elif wall == 1:
                        best.mark_clear(x - fc, clear_y, fc, clear_h, 'FRONT', code)
            best.clearzones = merge_clearances(best.clearzones)

        bed_plan = best

        # Reapply door clearances on the solved bedroom plan and capture the
        # exterior rectangle for the shared bathroom door.
        ext = self._add_door_clearance(bed_plan, 'DOOR', self.bed_openings)
        if ext:
            bed_plan.mark_clear(*ext, 'DOOR_CLEAR', 'DOOR')

        bath_ext = None
        if self.bath_dims and self.bath_openings:
            shared_op = Openings(bed_plan)
            shared_op.door_wall = WALL_RIGHT
            shared_op.door_center = self.bath_openings.door_center
            shared_op.door_width = self.bath_openings.door_width
            shared_op.swing_depth = self.bath_openings.swing_depth
            bath_ext = self._add_door_clearance(bed_plan, 'BATHROOM_DOOR', shared_op)
            # ``bath_ext`` is expressed in the bedroom grid; it will be converted
            # to metres and regridded when applied to the bathroom plan.
            if bath_ext:
                bed_plan.mark_clear(*bath_ext, 'DOOR_CLEAR', 'BATHROOM_DOOR')
        bed_plan.clearzones = merge_clearances(bed_plan.clearzones)

        bath_plan = None
        failure_msg = None
        if self.bath_dims and bath_ok:
            bath_plan = arrange_bathroom(
                self.bath_dims[0], self.bath_dims[1], BATH_RULES,
                openings=self.bath_openings,
                secondary_openings=self.bath_liv_openings,
            )
            if isinstance(bath_plan, GridPlan):
                dx, dy, dw, dh = self.bath_openings.door_rect_cells()
                for j in range(dy, dy + dh):
                    for i in range(dx, dx + dw):
                        bath_plan.occ[j][i] = 'DOOR'
                if self.bath_liv_openings:
                    dx2, dy2, dw2, dh2 = self.bath_liv_openings.door_rect_cells()
                    for j in range(dy2, dy2 + dh2):
                        for i in range(dx2, dx2 + dw2):
                            bath_plan.occ[j][i] = 'DOOR'
                bath_sticky = getattr(self, '_sticky_bath_items', [])
                if bath_sticky:
                    fx = BATH_RULES.get('fixtures', {})
                    clear = {
                        'lav_front': fx.get('lavatory', {}).get('front_clear_to_opposite_m', {}).get('min', 0.610),
                        'wc_front': fx.get('water_closet', {}).get('front_clear_to_opposite_m', {}).get('min', 0.610),
                        'tub_front': fx.get('bathtub', {}).get('front_clear_to_opposite_wall_m', {}).get('min', 0.762),
                        'shr_front': fx.get('bathtub', {}).get('entry_front_clear_m', 0.762),
                    }
                    FRONT_BATH_DEFAULT = {
                        'WC': clear['wc_front'],
                        'LAV': clear['lav_front'],
                        'TUB': clear['tub_front'],
                        'SHR': clear['shr_front'],
                    }
                    for (code, x, y, w, h, wall) in bath_sticky:
                        bath_plan.clear(x, y, w, h)
                        bath_plan.place(x, y, w, h, code)
                        fc_m = FRONT_BATH_DEFAULT.get(code, 0.0)
                        if fc_m > 0.0:
                            fc = bath_plan.meters_to_cells(fc_m)
                            if wall == 0:
                                bath_plan.mark_clear(x, y + h, w, fc, 'FRONT', code)
                            elif wall == 2:
                                bath_plan.mark_clear(x, y - fc, w, fc, 'FRONT', code)
                bath_plan.clearzones = merge_clearances(bath_plan.clearzones)
            else:
                bath_plan = None
                bath_ok = False
                failure_msg = 'Bathroom generation failed; bedroom only.'

        # Living room generation
        liv_plan = None
        if self.liv_dims:
            liv_plan = arrange_livingroom(
                self.liv_dims[0], self.liv_dims[1], LIV_RULES,
                openings=self.liv_openings,
            )
            if isinstance(liv_plan, GridPlan):
                dx, dy, dw, dh = self.liv_openings.door_rect_cells()
                for j in range(dy, dy + dh):
                    for i in range(dx, dx + dw):
                        liv_plan.occ[j][i] = 'DOOR'
                if getattr(self, 'liv_bath_openings', None):
                    self.liv_bath_openings.p = liv_plan
                    dx2, dy2, dw2, dh2 = self.liv_bath_openings.door_rect_cells()
                    for j in range(dy2, dy2 + dh2):
                        for i in range(dx2, dx2 + dw2):
                            liv_plan.occ[j][i] = 'DOOR'
                    self._add_door_clearance(
                        liv_plan, 'LIVING_DOOR', self.liv_bath_openings
                    )
                liv_sticky = getattr(self, '_sticky_liv_items', [])
                for (code, x, y, w, h, _wall) in liv_sticky:
                    liv_plan.clear(x, y, w, h)
                    liv_plan.place(x, y, w, h, code)
                liv_plan.clearzones = merge_clearances(liv_plan.clearzones)
                self._add_door_clearance(liv_plan, 'DOOR', self.liv_openings)
            else:
                liv_plan = None

        top_gw = bed_plan.gw + (bath_plan.gw if bath_plan else 0)
        top_gh = max(bed_plan.gh, bath_plan.gh if bath_plan else 0)
        liv_gw = liv_plan.gw if liv_plan else 0
        liv_gh = liv_plan.gh if liv_plan else 0
        total_gw = max(top_gw, liv_gw)
        total_gh = top_gh + liv_gh
        col_grid = ColumnGrid(total_gw, total_gh)
        bed_plan.column_grid = col_grid
        bed_plan.x_offset = 0
        bed_plan.y_offset = 0
        if bath_plan:
            bath_plan.column_grid = col_grid
            bath_plan.x_offset = bed_plan.gw
            bath_plan.y_offset = 0
        if liv_plan:
            liv_plan.column_grid = col_grid
            liv_plan.x_offset = 0
            liv_plan.y_offset = top_gh

        if bath_plan and bath_ext:
            bx, by, bw, bh = bath_ext
            lbl = bed_plan.coord_to_label(bx, by)
            bx_c, by_c = bath_plan.label_to_coord(lbl)
            bath_plan.mark_clear(bx_c, by_c, bw, bh, 'DOOR_CLEAR', 'BATHROOM_DOOR')
        if bath_plan:
            bath_plan.clearzones = merge_clearances(bath_plan.clearzones)
        if liv_plan:
            liv_plan.clearzones = merge_clearances(liv_plan.clearzones)

        # assign plans and combine with maximal-contact offsets
        self.bed_plan = bed_plan
        self.bath_plan = bath_plan
        self.liv_plan = liv_plan
        self._combine_plans()

        self.meta = meta
        self._log_run(meta)
        self._draw()
        sc = meta.get('score', 0.0)
        if bath_ok:
            self.status.set(
                f"Coverage {meta.get('coverage', 0) * 100:.1f}% · Paths {'ok' if meta.get('paths_ok') else 'blocked'} ·"
                f" Windows {'ok' if meta.get('reach_windows') else 'miss'} · Score {sc:.2f}"
            )
        elif failure_msg:
            self.status.set(failure_msg)

    def _solve_and_draw_bath(self):
        """Recompute bathroom layout without rerunning the bedroom solver."""
        if not self.bath_dims:
            return
        if self.sim_timer: self.root.after_cancel(self.sim_timer); self.sim_timer=None
        if self.sim2_timer: self.root.after_cancel(self.sim2_timer); self.sim2_timer=None
        self.sim_path=[]; self.sim_poly=[]; self.sim2_path=[]; self.sim2_poly=[]

        if not self._apply_openings_from_ui():
            return
        self.bath_plan = arrange_bathroom(
            self.bath_dims[0], self.bath_dims[1], BATH_RULES,
            openings=self.bath_openings,
            secondary_openings=self.bath_liv_openings,
        )
        dx, dy, dw, dh = self.bath_openings.door_rect_cells()
        for j in range(dy, dy + dh):
            for i in range(dx, dx + dw):
                self.bath_plan.occ[j][i] = 'DOOR'
        if self.bath_liv_openings:
            dx2, dy2, dw2, dh2 = self.bath_liv_openings.door_rect_cells()
            for j in range(dy2, dy2 + dh2):
                for i in range(dx2, dx2 + dw2):
                    self.bath_plan.occ[j][i] = 'DOOR'
        bath_sticky = getattr(self, '_sticky_bath_items', [])
        if bath_sticky:
            fx = BATH_RULES.get('fixtures', {})
            clear = {
                'lav_front': fx.get('lavatory', {}).get('front_clear_to_opposite_m', {}).get('min', 0.610),
                'wc_front': fx.get('water_closet', {}).get('front_clear_to_opposite_m', {}).get('min', 0.610),
                'tub_front': fx.get('bathtub', {}).get('front_clear_to_opposite_wall_m', {}).get('min', 0.762),
                'shr_front': fx.get('bathtub', {}).get('entry_front_clear_m', 0.762),
            }
            FRONT_BATH_DEFAULT = {
                'WC': clear['wc_front'],
                'LAV': clear['lav_front'],
                'TUB': clear['tub_front'],
                'SHR': clear['shr_front'],
            }
            for (code, x, y, w, h, wall) in bath_sticky:
                self.bath_plan.clear(x, y, w, h)
                self.bath_plan.place(x, y, w, h, code)
                fc_m = FRONT_BATH_DEFAULT.get(code, 0.0)
                if fc_m > 0.0:
                    fc = self.bath_plan.meters_to_cells(fc_m)
                    if wall == 0:
                        self.bath_plan.mark_clear(x, y + h, w, fc, 'FRONT', code)
                    elif wall == 2:
                        self.bath_plan.mark_clear(x, y - fc, w, fc, 'FRONT', code)
            self.bath_plan.clearzones = merge_clearances(self.bath_plan.clearzones)
        self._add_door_clearance(self.bath_plan, 'DOOR', self.bath_openings)
        if self.bath_liv_openings:
            self._add_door_clearance(self.bath_plan, 'LIVING_DOOR', self.bath_liv_openings)
        bed_wall, _, _ = self.bed_openings.door_span_cells()
        if bed_wall == WALL_RIGHT:
            self._add_door_clearance(self.bed_plan, 'DOOR', self.bed_openings)
        self.bath_plan.clearzones = merge_clearances(self.bath_plan.clearzones)

        if self.bed_plan is None:
            self.bed_plan = GridPlan(self.bed_Wm, self.bed_Hm)
        self._combine_plans()
        self._draw()

    def _solve_and_draw_liv(self):
        """Recompute living room layout without rerunning other solvers."""
        if not self.liv_dims:
            return
        if self.sim_timer:
            self.root.after_cancel(self.sim_timer); self.sim_timer = None
        if self.sim2_timer:
            self.root.after_cancel(self.sim2_timer); self.sim2_timer = None
        self.sim_path = []; self.sim_poly = []; self.sim2_path = []; self.sim2_poly = []

        if not self._apply_openings_from_ui():
            return
        self.liv_plan = arrange_livingroom(
            self.liv_dims[0], self.liv_dims[1], LIV_RULES,
            openings=self.liv_openings
        )
        dx, dy, dw, dh = self.liv_openings.door_rect_cells()
        for j in range(dy, dy + dh):
            for i in range(dx, dx + dw):
                self.liv_plan.occ[j][i] = 'DOOR'
        if getattr(self, 'liv_bath_openings', None):
            self.liv_bath_openings.p = self.liv_plan
            dx2, dy2, dw2, dh2 = self.liv_bath_openings.door_rect_cells()
            for j in range(dy2, dy2 + dh2):
                for i in range(dx2, dx2 + dw2):
                    self.liv_plan.occ[j][i] = 'DOOR'
            self._add_door_clearance(
                self.liv_plan, 'LIVING_DOOR', self.liv_bath_openings
            )
        liv_sticky = getattr(self, '_sticky_liv_items', [])
        for (code, x, y, w, h, _wall) in liv_sticky:
            self.liv_plan.clear(x, y, w, h)
            self.liv_plan.place(x, y, w, h, code)
        self.liv_plan.clearzones = merge_clearances(self.liv_plan.clearzones)
        self._add_door_clearance(self.liv_plan, 'DOOR', self.liv_openings)

        if self.bed_plan is None:
            self.bed_plan = GridPlan(self.bed_Wm, self.bed_Hm)
        self._combine_plans()
        self._draw()

    # ----------------- draw & helpers

    def _draw(self):
        cv = self.canvas
        cv.delete('plan')
        bed_gw, bed_gh = self.bed_plan.gw, self.bed_plan.gh
        bath_gw = self.bath_plan.gw if self.bath_plan else 0
        bath_gh = self.bath_plan.gh if self.bath_plan else 0
        liv_gw = self.liv_plan.gw if getattr(self, 'liv_plan', None) else 0
        liv_gh = self.liv_plan.gh if getattr(self, 'liv_plan', None) else 0
        top_w = bed_gw + bath_gw
        top_h = max(bed_gh, bath_gh)
        total_w = max(top_w, liv_gw)
        total_h = top_h + liv_gh
        cw, ch = cv.winfo_width() or 1, cv.winfo_height() or 1
        margin = 26
        scale = min((cw - 2 * margin) / max(1, total_w),
                    (ch - 2 * margin) / max(1, total_h))
        r = max(8, scale * 0.3)
        label_gap = r * 2.5
        margin = max(26, label_gap + 10)
        scale = min((cw - 2 * margin) / max(1, total_w),
                    (ch - 2 * margin) / max(1, total_h))
        zf = (self.zoom_factor.get()
              if hasattr(self.zoom_factor, 'get') else self.zoom_factor)
        scale *= zf
        self.scale = scale
        ox = (cw - total_w * scale) / 2
        oy = (ch - total_h * scale) / 2
        self.ox = ox
        self.oy = oy
        bed_ox = ox
        bed_oy = oy + (top_h - bed_gh) * scale
        bath_ox = ox + bed_gw * scale
        bath_oy = oy + (top_h - bath_gh) * scale
        liv_ox = ox
        liv_oy = oy + top_h * scale
        wall_width = max(4, int(scale * 0.12)) * 3
        # Make door/window outlines thicker for better visibility
        open_width = max(2, wall_width // 2)
        self.opening_item_info = {}

        self._draw_all_layers(
            self.bed_plan,
            self.bed_openings,
            bed_ox,
            bed_oy,
            scale,
            wall_width,
            open_width,
            True,
            'bed',
        )
        if self.bath_plan:
            self._draw_all_layers(
                self.bath_plan,
                self.bath_openings,
                bath_ox,
                bath_oy,
                scale,
                wall_width,
                open_width,
                True,
                'bath',
            )
            if getattr(self, 'bath_liv_openings', None):
                self.bath_liv_openings.p = self.bath_plan
                self._draw_all_layers(
                    self.bath_plan,
                    self.bath_liv_openings,
                    bath_ox,
                    bath_oy,
                    scale,
                    wall_width,
                    open_width,
                    True,
                    'bath',
                )
        if getattr(self, 'liv_plan', None):
            self._draw_all_layers(
                self.liv_plan,
                self.liv_openings,
                liv_ox,
                liv_oy,
                scale,
                wall_width,
                open_width,
                True,
                'living',
            )
            if getattr(self, 'liv_bath_openings', None):
                self.liv_bath_openings.p = self.liv_plan
                self._draw_all_layers(
                    self.liv_plan,
                    self.liv_bath_openings,
                    liv_ox,
                    liv_oy,
                    scale,
                    wall_width,
                    open_width,
                    True,
                    'living',
                )

        col_grid = getattr(self.plan, 'column_grid', None)
        if col_grid:
            self.grid_overlay.redraw(col_grid, ox, oy, scale)
        # clear previous popovers when redrawing
        if hasattr(self, 'popover'):
            self.popover.hide()

        def draw_path(poly, color):
            if len(poly) >= 2:
                for k in range(1, len(poly)):
                    x0, y0 = poly[k - 1]
                    x1, y1 = poly[k]
                    cv.create_line(
                        x0,
                        y0,
                        x1,
                        y1,
                        fill=color,
                        width=2,
                        capstyle=tk.ROUND,
                        tags=('plan',),
                    )
        draw_path(self.sim_poly, HUMAN1_COLOR)
        draw_path(self.sim2_poly, HUMAN2_COLOR)

        if self.sim_path:
            i, j = self.sim_path[min(self.sim_index, len(self.sim_path) - 1)]
            self._draw_human_block(i, j, HUMAN1_COLOR, which=1)
        if self.sim2_path:
            i, j = self.sim2_path[min(self.sim2_index, len(self.sim2_path) - 1)]
            self._draw_human_block(i, j, HUMAN2_COLOR, which=2)

    def _draw_all_layers(
        self,
        plan,
        openings,
        ox,
        oy,
        scale,
        wall_width,
        open_width,
        draw_door=True,
        room_name='bed',
    ):
        cv = self.canvas
        gw, gh = plan.gw, plan.gh
        GRID_COLOR = '#dddddd'
        for i in range(gw + 1):
            x = ox + i * scale
            cv.create_line(
                x,
                oy,
                x,
                oy + gh * scale,
                fill=GRID_COLOR,
                tags=('plan', 'grid'),
            )
        for j in range(gh + 1):
            y = oy + j * scale
            cv.create_line(
                ox,
                y,
                ox + gw * scale,
                y,
                fill=GRID_COLOR,
                tags=('plan', 'grid'),
            )
        bound = set()
        self._draw_clearances(cv, plan, openings, ox, oy, scale)
        for j in range(gh):
            for i in range(gw):
                code = plan.occ[j][i]
                if not code or code == 'DOOR':
                    continue
                base = code.split(':')[0]
                tag = base.split('_')[0]
                color = PALETTE.get(tag, '#888')
                x0 = ox + i * scale
                y0 = oy + (gh - 1 - j) * scale
                cv.create_rectangle(
                    x0,
                    y0,
                    x0 + scale,
                    y0 + scale,
                    outline='',
                    fill=color,
                    tags=('plan', 'furn', tag),
                )
                if tag not in bound:
                    cv.tag_bind(
                        tag,
                        '<Enter>',
                        lambda e, c=tag: self._show_tooltip(e, c),
                    )
                    cv.tag_bind(tag, '<Leave>', self._hide_tooltip)
                    bound.add(tag)
        cv.create_rectangle(
            ox,
            oy,
            ox + gw * scale,
            oy + gh * scale,
            outline=WALL_COLOR,
            fill='',
            width=wall_width,
            tags=('plan', 'room'),
        )
        self._draw_room_openings(
            cv,
            openings,
            ox,
            oy,
            scale,
            wall_width,
            open_width,
            draw_door,
            room_name,
        )
        cv.tag_lower('clear')
        cv.tag_raise('furn', 'clear')
        cv.tag_raise('room', 'furn')
        cv.tag_raise('opening', 'room')

    def _draw_clearances(self, cv, plan, openings, ox, oy, scale):
        """Draw clearance rectangles from ``plan`` and ``openings``.

        Existing rectangles from ``plan.clearzones`` are rendered first.  If a
        door or windows are defined in ``openings`` but their clearances are not
        present in ``plan.clearzones`` (common when rendering unsolved rooms),
        corresponding rectangles are computed on-the-fly.  Door clearances are
        mirrored so that both sides of the doorway are visualised.
        """

        gh = plan.gh
        rects = []

        # Record existing rectangles to avoid duplicates when adding computed
        # ones from openings.
        seen = set()
        for x, y, w, h, *_ in plan.clearzones:
            rects.append((x, y, w, h))
            seen.add((x, y, w, h))

        if openings is not None:
            # Door clearance (interior + mirrored exterior)
            wall, start, width = openings.door_span_cells()
            if wall >= 0 and width > 0:
                depth = plan.meters_to_cells(openings.swing_depth)
                pw = max(1, PATH_WIDTH_CELLS)
                if wall == WALL_BOTTOM:
                    inside = (start, depth, width, pw)
                    outside = (start, -depth - pw, width, pw)
                elif wall == WALL_TOP:
                    inside = (start, plan.gh - depth - pw, width, pw)
                    outside = (start, plan.gh + depth, width, pw)
                elif wall == WALL_LEFT:
                    inside = (depth, start, pw, width)
                    outside = (-depth - pw, start, pw, width)
                else:  # WALL_RIGHT
                    inside = (plan.gw - depth - pw, start, pw, width)
                    outside = (plan.gw + depth, start, pw, width)
                for rect in (inside, outside):
                    if rect not in seen:
                        rects.append(rect)
                        seen.add(rect)

            # Window clearances
            depth = max(1, plan.meters_to_cells(WINDOW_CLEARANCE_M))
            for wall, start, length in openings.window_spans_cells():
                if wall == WALL_BOTTOM:
                    rect = (start, depth, length, 1)
                elif wall == WALL_TOP:
                    rect = (start, plan.gh - 1 - depth, length, 1)
                elif wall == WALL_LEFT:
                    rect = (depth, start, 1, length)
                else:  # WALL_RIGHT
                    rect = (plan.gw - 1 - depth, start, 1, length)
                if rect not in seen:
                    rects.append(rect)
                    seen.add(rect)

        for x, y, w, h in rects:
            x0 = ox + x * scale
            y0 = oy + (gh - (y + h)) * scale
            cv.create_rectangle(
                x0,
                y0,
                x0 + w * scale,
                y0 + h * scale,
                outline=PALETTE['CLEAR'],
                dash=(8, 6),
                width=2,
                tags=('plan', 'clear'),
            )

    def refresh_overlay(self):
        """Rebuild the persistent column grid overlay."""
        self.canvas.delete('overlay')
        col_grid = getattr(self.plan, 'column_grid', None)
        if col_grid:
            self.grid_overlay.redraw(col_grid, self.ox, self.oy, self.scale)

    def _show_tooltip(self, event, code):
        base = code.split('_')[0]
        label = ITEM_LABELS.get(base, code)
        color = PALETTE.get(base, '#fff')
        self._hide_tooltip()

        x = event.x + 12
        y = event.y + 12
        text_id = self.canvas.create_text(x + 16, y, text=label, fill='black',
                                          anchor='nw', tags=('tooltip',))
        bbox = self.canvas.bbox(text_id)
        rect_id = self.canvas.create_rectangle(
            x, y - 4, bbox[2] + 4, max(bbox[3], y + 14) + 4,
            fill='white', outline='black', tags=('tooltip',))
        self.canvas.tag_raise(text_id, rect_id)
        color_id = self.canvas.create_rectangle(
            x + 2, y + 2, x + 14, y + 14,
            fill=color, outline='black', tags=('tooltip',))
        self.canvas.tag_raise(color_id, rect_id)

    def _hide_tooltip(self, event=None):
        self.canvas.delete('tooltip')
        if hasattr(self, 'popover'):
            self.popover.hide()

    def _on_opening_click(self, event):
        """Open an editing dialog for the clicked door or window."""
        item = self.canvas.find_withtag('current')
        if hasattr(self, 'popover'):
            self.popover.hide()
        if not item:
            return
        info = getattr(self, 'opening_item_info', {}).get(item[0])
        if not info:
            return

        def _apply():
            self._apply_openings_from_ui()
            room = info.get('room')
            if room == 'bath':
                self._solve_and_draw_bath()
            elif room == 'living':
                self._solve_and_draw_liv()
            else:
                self._solve_and_draw()

        OpeningDialog(self.root, info, _apply)
        return 'break'

    def _on_canvas_click(self, event):
        return self._on_opening_click(event)

    def _draw_room_openings(self, cv, openings, ox, oy, scale,
                            wall_width, open_width, draw_door=True,
                            room_name='bed'):

        if openings is None:
            return
        gw, gh = openings.p.gw, openings.p.gh
        cell = openings.p.cell

        def seg(wall, start, length, fill_color, kind, idx=None):
            if wall < 0 or length <= 0:
                return
            w = gw * scale
            h = gh * scale
            s = start * scale
            L = length * scale
            half = wall_width / 2
            if wall == 0:

                x0 = ox + s
                x1 = ox + s + L
                y0 = oy + h - half
                y1 = oy + h + half
            elif wall == 2:
                x0 = ox + s
                x1 = ox + s + L
                y0 = oy - half
                y1 = oy + half
            elif wall == 3:
                x0 = ox - half
                x1 = ox + half
                y0 = oy + h - (s + L)
                y1 = oy + h - s
            else:
                x0 = ox + w - half
                x1 = ox + w + half
                y0 = oy + h - (s + L)
                y1 = oy + h - s
            item = cv.create_rectangle(
                x0,
                y0,
                x1,
                y1,
                outline='black',
                width=open_width,
                fill=fill_color,
                tags=('plan', 'opening'),
            )
            self.opening_item_info[item] = {
                'type': kind,
                'room': room_name,
                'wall': wall,
                'start': start,
                'length': length,
                'cell': cell,
                'color': fill_color,
                'openings': openings,
                'index': idx,
            }
            cv.tag_bind(item, '<Button-1>', self._on_opening_click)

        if draw_door:
            dwall, dstart, dwidth = openings.door_span_cells()
            seg(dwall, dstart, dwidth, DOOR_FILL, 'door')

        for i, (wall, start, length) in enumerate(openings.window_spans_cells()):
            seg(wall, start, length, WINDOW_FILL, 'window', i)

    def _draw_opening_segment(self, cv, wall, start, length, color, width):
        if wall<0 or length<=0: return
        x0=self.ox; y0=self.oy; w=self.plan.gw*self.scale; h=self.plan.gh*self.scale
        s=start*self.scale; L=length*self.scale
        if wall==0:
            cv.create_line(x0+s, y0+h, x0+s+L, y0+h, fill=color, width=width, tags=('plan',))
        elif wall==2:
            cv.create_line(x0+s, y0, x0+s+L, y0, fill=color, width=width, tags=('plan',))
        elif wall==3:
            cv.create_line(x0, y0+h-(s+L), x0, y0+h-s, fill=color, width=width, tags=('plan',))
        else:
            cv.create_line(x0+w, y0+h-(s+L), x0+w, y0+h-s, fill=color, width=width, tags=('plan',))

    def _draw_human_block(self, i, j, color, which=1):
        # small square centered in cell
        size = max(4, self.scale*0.5)
        cx, cy = self._cell_center(i,j)
        x0, y0 = cx - size/2, cy - size/2
        x1, y1 = cx + size/2, cy + size/2
        if which==1:
            if self.human_id is not None:
                try: self.canvas.coords(self.human_id, x0,y0,x1,y1)
                except: self.human_id=None
            if self.human_id is None:
                self.human_id = self.canvas.create_rectangle(x0,y0,x1,y1, fill=color, outline='', tags=('plan',))
        else:
            if self.human2_id is not None:
                try: self.canvas.coords(self.human2_id, x0,y0,x1,y1)
                except: self.human2_id=None
            if self.human2_id is None:
                self.human2_id = self.canvas.create_rectangle(x0,y0,x1,y1, fill=color, outline='', tags=('plan',))

    # ------- geometry helpers

    def _cell_center(self, i, j):
        x0 = self.ox + (i + 0.5) * self.scale
        y0 = self.oy + (self.plan.gh - (j + 0.5)) * self.scale
        return (x0, y0)

    def _cell_rect(self, x, y):
        x0 = self.ox + x * self.scale
        y0 = self.oy + (self.plan.gh - 1 - y) * self.scale
        return (x0, y0, x0 + self.scale, y0 + self.scale)

    def _xy_to_cell(self, px, py):
        if not (self.ox <= px <= self.ox + self.plan.gw*self.scale and
                self.oy <= py <= self.oy + self.plan.gh*self.scale):
            return (None, None)
        i = int((px - self.ox) // self.scale)
        j_from_top = int((py - self.oy) // self.scale)
        j = self.plan.gh - 1 - j_from_top
        return (clamp(i, 0, self.plan.gw-1), clamp(j, 0, self.plan.gh-1))

    def _infer_wall(self, x, y, w, h):
        if y == 0: return 0
        if y + h == self.plan.gh: return 2
        if x == 0: return 3
        if x + w == self.plan.gw: return 1
        return -1

    def _components_by_code(self, code):
        return components_by_code(self.plan, code)

    def _hit_component(self, px, py):
        i, j = self._xy_to_cell(px, py)
        if i is None: return None
        cell = self.plan.occ[j][i]
        if not cell or cell == 'DOOR': return None
        base = cell.split(':')[0]
        for (x,y,w,h,wall) in self._components_by_code(base):
            if x <= i < x+w and y <= j < y+h:
                return (x,y,w,h,base)
        return None

    # ------- add/remove furniture & boundary placement

    def _add_furniture(self):
        kind=self.furn_kind.get()
        p=self.plan
        placed=None
        if kind in ('TVU','DESK'):
            spec=BEDROOM_BOOK[kind][list(BEDROOM_BOOK[kind].keys())[0]]
            W=p.meters_to_cells(spec['w']); D=p.meters_to_cells(spec['d'])
            placed=self._place_free_boundary(kind, W, D)
            if placed and 'front_rec' in spec:
                x,y,w,h=placed
                wall=self._infer_wall(x,y,w,h)
                fc=p.meters_to_cells(spec['front_rec'])
                if wall==0: p.mark_clear(x,y+h,w,fc,'FRONT',kind)
                elif wall==2: p.mark_clear(x,y-fc,w,fc,'FRONT',kind)
                elif wall==3: p.mark_clear(x+w,y,fc,h,'FRONT',kind)
                else: p.mark_clear(x-fc,y,fc,h,'FRONT',kind)
        elif kind in ('DRS_4FT','CHEST_SM'):
            spec=BEDROOM_BOOK['DRESSER'][kind]
            W=p.meters_to_cells(spec['w']); D=p.meters_to_cells(spec['d'])
            placed=self._place_free_boundary('DRS', W, D, prefer_longedge=True)
            if placed:
                x, y, w, h = placed
                wall = self._infer_wall(x, y, w, h)
                fc = p.meters_to_cells(spec['front_rec'])
                fc = 2 if fc == 4 else 5 if fc == 3 else fc
                rect_w = 5
                rect_d = 2
                clear_x = x + (w - rect_w) / 2
                clear_y = y + (h - rect_w) / 2
                if wall == 0:
                    p.mark_clear(clear_x, y + h, rect_w, rect_d, 'FRONT', 'DRS')
                elif wall == 2:
                    p.mark_clear(clear_x, y - rect_d, rect_w, rect_d, 'FRONT', 'DRS')
                elif wall == 3:
                    p.mark_clear(x + w, clear_y, rect_d, rect_w, 'FRONT', 'DRS')
                else:
                    p.mark_clear(x - rect_d, clear_y, rect_d, rect_w, 'FRONT', 'DRS')
        elif kind in ('WRD_S_210','WRD_H_180'):
            spec=BEDROOM_BOOK['WARDROBE'][kind]
            W=p.meters_to_cells(spec['w']); D=p.meters_to_cells(spec['d'])
            placed=self._place_free_boundary('WRD', W, D, prefer_longedge=True)
            if placed:
                x,y,w,h=placed; wall=self._infer_wall(x,y,w,h)
                fc=p.meters_to_cells(spec['front_rec'])
                if fc == 4:
                    fc = 2
                if wall in (0,2) and w>1:
                    clear_w = w-1
                    clear_x = x+0.5
                else:
                    clear_w = w
                    clear_x = x
                if wall==0:
                    p.mark_clear(clear_x, y+h, clear_w, fc, 'FRONT', 'WRD')
                elif wall==2:
                    p.mark_clear(clear_x, y-fc, clear_w, fc, 'FRONT', 'WRD')
                elif wall==3:
                    p.mark_clear(x+w, y, fc, h, 'FRONT', 'WRD')
                else:
                    p.mark_clear(x-fc, y, fc, h, 'FRONT', 'WRD')
        if placed:
            p.clearzones=merge_clearances(p.clearzones)
            self._log_event({"event":"add_furniture","kind":kind,"rect":placed})
            self._draw()
        else:
            messagebox.showinfo('Add', 'No valid boundary slot found for this item.')

    def _remove_furniture(self):
        kind=self.furn_kind.get()
        code_map={'TVU':'TVU','DESK':'DESK','DRS_4FT':'DRS','CHEST_SM':'DRS','WRD_S_210':'WRD','WRD_H_180':'WRD'}
        code=code_map.get(kind,kind)
        comps=self._components_by_code(code)
        if not comps:
            messagebox.showinfo('Remove','No such furniture present.'); return
        x,y,w,h,_=comps[-1]
        self.plan.clear(x,y,w,h)
        self.plan.clearzones=[r for r in self.plan.clearzones if r[5]!=code]
        self.plan.clearzones=merge_clearances(self.plan.clearzones)
        self._log_event({"event":"remove_furniture","kind":kind,"rect":[x,y,w,h]})
        self._draw()

    def _place_free_boundary(self, code:str, W:int, D:int, prefer_longedge:bool=True):
        """
        Try placing a boundary-bound furniture piece.
        Enforces:
          - Long edge parallel to wall (for WRD/DRS/DESK/TVU)
          - No blocking door/window spans
          - Area-based size restriction (cells fraction cap)
        Returns (x,y,w,h) or None.
        """
        p = self.plan
        gw, gh = p.gw, p.gh

        # area cap
        room_cells = gw * gh
        fp = W * D
        frac_caps = {'WRD': 0.18, 'DRS': 0.14, 'DESK': 0.10, 'TVU': 0.08}
        cap = frac_caps.get(code, 0.20) * room_cells
        if fp > cap: return None

        # openings
        dwall, dstart, dwidth = self.openings.door_span_cells()
        bath_door_span = None
        if getattr(self, "bath_openings", None):
            bwall, bstart, bwidth = self.bath_openings.door_span_cells()
            if bwall == 3:  # Shared-wall bathroom door
                bath_door_span = (bstart, bwidth)
        win_spans = self.openings.window_spans_cells()

        def spans_block(wall, x, y, w, h):
            if bath_door_span and wall == 3:  # Shared-wall bathroom door
                bstart, bwidth = bath_door_span
                if self._span_blocks_opening(3, max(0, bstart - 1),
                                             max(1, bwidth + 2), x, y, w, h):
                    return True
            if wall == dwall and self._span_blocks_opening(
                    wall, max(0, dstart - 1), max(1, dwidth + 2), x, y, w, h):
                return True
            for ww, start, L in win_spans:
                if wall == ww and self._span_blocks_opening(wall, start, L, x, y, w, h):
                    return True
            return False

        walls = [0, 2, 3, 1]
        for wall in walls:
            if prefer_longedge:
                if wall in (0,2): ww, hh = max(W,D), min(W,D)
                else:             ww, hh = min(W,D), max(W,D)
            else:
                ww, hh = W, D

            if wall == 0:
                y = 0; slots = [(x, y, ww, hh) for x in range(0, gw-ww+1)]
            elif wall == 2:
                y = gh - hh; slots = [(x, y, ww, hh) for x in range(0, gw-ww+1)]
            elif wall == 3:
                x = 0; slots = [(x, y, ww, hh) for y in range(0, gh-hh+1)]
            else:
                x = gw - ww; slots = [(x, y, ww, hh) for y in range(0, gh-hh+1)]

            self.rng.shuffle(slots)
            for x, y, w, h in slots:
                if spans_block(wall, x, y, w, h): continue
                if not p.fits(x, y, w, h): continue
                p.place(x, y, w, h, code)
                return (x, y, w, h)
        return None

    # ------- drag with ghost preview

    def _on_down(self, e):
        if getattr(self, 'selected_locked', False):
            return

        # Ensure arrow keys go to the canvas
        try: self.canvas.focus_set()
        except Exception: pass


        comp = self._hit_component(e.x, e.y)
        if comp and comp[4] in self.ALL_FURN_CODES:
            x, y, w, h = comp[:4]
            code = comp[4]
            if code in self.BATH_CODES:
                room = 'bath'
            elif code in self.LIV_CODES:
                room = 'living'
            elif code in self.KITCH_CODES:
                room = 'kitchen'
            else:
                room = 'bed'
            self.selected = {'rect': [x, y, w, h], 'code': code}

            self.drag_item = {
                'orig': [x, y, w, h],
                'live': [x, y, w, h],
                'code': code,
                'room': room,
                'ghost': None
            }
            x0, y0, _, _ = self._cell_rect(x, y)
            _, _, x1, y1 = self._cell_rect(x+w-1, y+h-1)
            self.drag_item['ghost'] = self.canvas.create_rectangle(
                x0, y0, x1, y1, outline='#ffffff', dash=(4,3), width=2
            )
        else:
            if not getattr(self, 'selected_locked', False):
                self.selected = None
                self._draw()


    def _on_drag(self, e):
        if not self.drag_item:
            return
        _, _, w, h = self.drag_item['live']
        i, j = self._xy_to_cell(e.x, e.y)
        if i is None:
            return
        code = self.drag_item['code']
        room = self.drag_item.get('room')
        bed_gw = getattr(self.bed_plan, 'gw', 0)
        if room == 'bed':
            nx = clamp(i, 0, bed_gw - w)
            ny = clamp(j, 0, self.bed_plan.gh - h)
        elif room == 'bath' and getattr(self, 'bath_plan', None):
            xoff = bed_gw
            nx = clamp(i, xoff, xoff + self.bath_plan.gw - w)
            ny = clamp(j, 0, self.bath_plan.gh - h)
        elif room == 'living' and getattr(self, 'liv_plan', None):
            yoff = max(self.bed_plan.gh, self.bath_plan.gh if self.bath_plan else 0)
            nx = clamp(i, 0, self.liv_plan.gw - w)
            ny = clamp(j, yoff, yoff + self.liv_plan.gh - h)
        else:
            nx = clamp(i, 0, self.plan.gw - w)
            ny = clamp(j, 0, self.plan.gh - h)

        x0, y0, _, _ = self._cell_rect(nx, ny)
        _, _, x1, y1 = self._cell_rect(nx+w-1, ny+h-1)
        if self.drag_item.get('ghost') is not None:
            self.canvas.coords(self.drag_item['ghost'], x0, y0, x1, y1)

        self.drag_item['live'] = [nx, ny, w, h]

    def _on_up(self, e):
        if not self.drag_item:
            return

        ox, oy, ow, oh = self.drag_item['orig']
        nx, ny, w, h = self.drag_item['live']
        code = self.drag_item['code']
        room = self.drag_item.get('room')

        if self.drag_item.get('ghost') is not None:
            try:
                self.canvas.delete(self.drag_item['ghost'])
            except Exception:
                pass

        if room == 'bed':
            target_plan = self.bed_plan
            xoff = yoff = 0
        elif room == 'bath':
            target_plan = self.bath_plan
            xoff = self.bed_plan.gw
            yoff = 0
            ox -= xoff; nx -= xoff
        else:
            target_plan = self.liv_plan
            xoff = 0
            yoff = max(self.bed_plan.gh, self.bath_plan.gh if self.bath_plan else 0)
            oy -= yoff; ny -= yoff

        # clear original block before testing commit
        target_plan.clear(ox, oy, ow, oh)
        from_rect = [ox + xoff, oy + yoff, ow, oh]

        # bounds/overlap only for drag commit (stable & predictable)
        ok = target_plan.fits(nx, ny, w, h)
        in_room = (0 <= nx and nx + w <= target_plan.gw and
                   0 <= ny and ny + h <= target_plan.gh)
        ok = ok and in_room

        if ok:
            target_plan.place(nx, ny, w, h, code)
            to_rect = [nx + xoff, ny + yoff, w, h]
            self.selected = {'rect': to_rect, 'code': code}
            self._log_event({"event": "drag", "code": code,
                             "from": from_rect, "to": to_rect})
        else:
            target_plan.place(ox, oy, ow, oh, code)
            to_rect = [ox + xoff, oy + yoff, ow, oh]

        target_plan.clearzones = merge_clearances(target_plan.clearzones)
        self._commit_drag(room, from_rect, to_rect, code)
        self.drag_item = None
        self._combine_plans()
        self._draw()


    # ------- keyboard + opening-span helpers

    def _commit_drag(self, room, from_rect, to_rect, code):
        if from_rect == to_rect:
            return
        if not hasattr(self, 'undo_stack'):
            self.undo_stack = []
        if not hasattr(self, 'redo_stack'):
            self.redo_stack = []
        self.undo_stack.append({'room': room, 'code': code,
                                 'from': from_rect, 'to': to_rect})
        self.redo_stack.clear()

    def _on_double_click(self, e):
        """Toggle sticky selection lock. Double-click an item to lock; double-click again to unlock."""
        comp = self._hit_component(e.x, e.y)
        
                # Keep keyboard focus on canvas while locked
        try: self.canvas.focus_set()
        except Exception: pass

        
        if getattr(self, 'selected_locked', False):
            # unlock regardless of where we double-click
            self.selected_locked = False
            return "break"
        if comp:
            self.selected = {'rect': list(comp[:4]), 'code': comp[4]}
            self.selected_locked = True
            self._draw()
            return "break"

    def _on_arrow_rotate(self, e):
        # Rotate selected item 90° clockwise (swap w↔h)
        self._rotate_selected(+1)
        return "break"


    def _on_ctrl_left(self, e):
        self._nudge_selected(-1, 0)
        return "break"
    def _on_ctrl_right(self, e):
        self._nudge_selected(+1, 0)
        return "break"

    def _on_ctrl_up(self, e):
        self._nudge_selected(0, +1)
        return "break"

    def _on_ctrl_down(self, e):
        self._nudge_selected(0, -1)
        return "break"

    def _span_blocks_opening(self, wall:int, start:int, width:int, x:int,y:int,w:int,h:int)->bool:
        p = self.plan
        # top/bottom walls (0 = top, 2 = bottom)
        if wall in (0, 2):
            touches = (y == 0) if wall == 0 else (y + h == p.gh)
            if not touches: return False
            a0, a1 = x, x + w
            b0, b1 = start, start + width
            return not (a1 <= b0 or b1 <= a0)
        # left/right walls (3 = left, 1 = right)
        if wall in (3, 1):
            touches = (x == 0) if wall == 3 else (x + w == p.gw)
            if not touches: return False
            a0, a1 = y, y + h
            b0, b1 = start, start + width
            return not (a1 <= b0 or b1 <= a0)
        return False

    def _nudge_selected(self, dx:int, dy:int):
        sel = getattr(self, 'selected', None)
        if not sel: return
        x, y, w, h = sel['rect']
        code = sel['code']
        nx = clamp(x + dx, 0, self.plan.gw - w)
        ny = clamp(y + dy, 0, self.plan.gh - h)

        # temporarily clear, then validate
        self.plan.clear(x, y, w, h)
        ok = self.plan.fits(nx, ny, w, h)
        if ok:
            wall = self._infer_wall(nx, ny, w, h)
            dwall, dstart, dwidth = self.openings.door_span_cells()
            if wall == dwall and self._span_blocks_opening(wall, max(0, dstart-1), max(1, dwidth+2), nx, ny, w, h):
                ok = False
            if ok:
                for wwspan, start, L in self.openings.window_spans_cells():
                    if wall == wwspan and self._span_blocks_opening(wall, start, L, nx, ny, w, h):
                        ok = False; break

        if ok:
            self.plan.place(nx, ny, w, h, code)
            self.selected['rect'] = [nx, ny, w, h]
            self._log_event({"event":"nudge","code":code,"from":[x,y,w,h],"to":[nx,ny,w,h]})
        else:
            self.plan.place(x, y, w, h, code)

        self.plan.clearzones = merge_clearances(self.plan.clearzones)
        self._sync_room_plans()
        self._draw()

    def _rotate_selected(self, direction=+1):
        sel = getattr(self, 'selected', None)
        if not sel:
            return
        x, y, w, h = sel['rect']
        code = sel['code']

        # swap length ↔ breadth (90° clockwise)
        nw, nh = h, w
        nx = max(0, min(x, self.plan.gw - nw))
        ny = max(0, min(y, self.plan.gh - nh))

        self.plan.clear(x, y, w, h)
        if self.plan.fits(nx, ny, nw, nh):
            self.plan.place(nx, ny, nw, nh, code)
            self.selected['rect'] = [nx, ny, nw, nh]
            self._log_event({"event": "rotate", "code": code,
                             "from": [x, y, w, h], "to": [nx, ny, nw, nh]})
        else:
            self.plan.place(x, y, w, h, code)

        self.plan.clearzones = merge_clearances(self.plan.clearzones)
        self._sync_room_plans()
        self._draw()


    def _combine_plans(self):
        """Merge per-room plans into ``self.plan``."""
        plans = [self.bed_plan]
        has_bath = bool(self.bath_plan)
        has_liv = bool(getattr(self, 'liv_plan', None))
        has_kitch = bool(getattr(self, 'kitch_plan', None))
        if has_bath:
            plans.append(self.bath_plan)
        if has_liv:
            plans.append(self.liv_plan)
        if has_kitch:
            plans.append(self.kitch_plan)
        if len(plans) == 1:
            self.plan = self.bed_plan
            return

        if all(getattr(p, 'x_offset', 0) == 0 and getattr(p, 'y_offset', 0) == 0 for p in plans):
            self._layout_rooms()

        total_gw = max(p.x_offset + p.gw for p in plans)
        total_gh = max(p.y_offset + p.gh for p in plans)
        col_grid = ColumnGrid(total_gw, total_gh)
        for p in plans:
            p.column_grid = col_grid

        # ``GridPlan`` derives its internal grid dimensions from the supplied
        # physical size (``Wm``/``Hm``).  When the per-room plans use widths or
        # heights that are not exact multiples of ``CELL_M`` this can lead to
        # rounding differences: the summed metres may produce a grid that is
        # smaller than the total number of columns/rows we intend to index.
        # Derive the combined physical dimensions directly from the desired
        # grid dimensions so that ``combined.occ`` is guaranteed to be large
        # enough for all offsets.
        total_wm = total_gw * CELL_M
        total_hm = total_gh * CELL_M

        combined = GridPlan(total_wm, total_hm, column_grid=col_grid)
        for p in plans:
            xoff = getattr(p, 'x_offset', 0)
            yoff = getattr(p, 'y_offset', 0)
            for j in range(p.gh):
                for i in range(p.gw):
                    code = p.occ[j][i]
                    if code:
                        combined.occ[j + yoff][i + xoff] = code
            for x, y, w, h, kind, owner in p.clearzones:
                combined.clearzones.append((x + xoff, y + yoff, w, h, kind, owner))
        combined.clearzones = merge_clearances(combined.clearzones)
        self.plan = combined


    def _sync_room_plans(self):
        """Synchronize per-room plans with current combined plan."""
        has_bath = bool(self.bath_plan)
        has_liv = bool(getattr(self, 'liv_plan', None))
        has_kitch = bool(getattr(self, 'kitch_plan', None))
        if has_bath or has_liv or has_kitch:
            bed = GridPlan(self.bed_Wm, self.bed_Hm)
            bath = GridPlan(self.bath_Wm, self.bath_Hm) if has_bath else None
            liv = GridPlan(self.liv_Wm, self.liv_Hm) if has_liv else None
            kitch = GridPlan(self.kitch_Wm, self.kitch_Hm) if has_kitch else None
            for j in range(self.plan.gh):
                for i in range(self.plan.gw):
                    code = self.plan.occ[j][i]
                    if not code:
                        continue
                    if (
                        self.bed_plan.x_offset <= i < self.bed_plan.x_offset + self.bed_plan.gw
                        and self.bed_plan.y_offset <= j < self.bed_plan.y_offset + self.bed_plan.gh
                    ):
                        bed.occ[j - self.bed_plan.y_offset][i - self.bed_plan.x_offset] = code
                    elif has_bath and (
                        self.bath_plan.x_offset <= i < self.bath_plan.x_offset + self.bath_plan.gw
                        and self.bath_plan.y_offset <= j < self.bath_plan.y_offset + self.bath_plan.gh
                    ):
                        bath.occ[j - self.bath_plan.y_offset][i - self.bath_plan.x_offset] = code
                    elif has_liv and (
                        self.liv_plan.x_offset <= i < self.liv_plan.x_offset + self.liv_plan.gw
                        and self.liv_plan.y_offset <= j < self.liv_plan.y_offset + self.liv_plan.gh
                    ):
                        liv.occ[j - self.liv_plan.y_offset][i - self.liv_plan.x_offset] = code
                    elif has_kitch and (
                        self.kitch_plan.x_offset <= i < self.kitch_plan.x_offset + self.kitch_plan.gw
                        and self.kitch_plan.y_offset <= j < self.kitch_plan.y_offset + self.kitch_plan.gh
                    ):
                        kitch.occ[j - self.kitch_plan.y_offset][i - self.kitch_plan.x_offset] = code
            for x, y, w, h, kind, owner in self.plan.clearzones:
                bx0, by0 = self.bed_plan.x_offset, self.bed_plan.y_offset
                bx1, by1 = bx0 + self.bed_plan.gw, by0 + self.bed_plan.gh
                if bx0 <= x and x + w <= bx1 and by0 <= y and y + h <= by1:
                    bed.clearzones.append((x - bx0, y - by0, w, h, kind, owner))
                    continue
                if has_bath:
                    bpx0, bpy0 = self.bath_plan.x_offset, self.bath_plan.y_offset
                    bpx1, bpy1 = bpx0 + self.bath_plan.gw, bpy0 + self.bath_plan.gh
                    if bpx0 <= x and x + w <= bpx1 and bpy0 <= y and y + h <= bpy1:
                        bath.clearzones.append((x - bpx0, y - bpy0, w, h, kind, owner))
                        continue
                if has_liv:
                    lx0, ly0 = self.liv_plan.x_offset, self.liv_plan.y_offset
                    lx1, ly1 = lx0 + self.liv_plan.gw, ly0 + self.liv_plan.gh
                    if lx0 <= x and x + w <= lx1 and ly0 <= y and y + h <= ly1:
                        liv.clearzones.append((x - lx0, y - ly0, w, h, kind, owner))
                        continue
                if has_kitch:
                    kx0, ky0 = self.kitch_plan.x_offset, self.kitch_plan.y_offset
                    kx1, ky1 = kx0 + self.kitch_plan.gw, ky0 + self.kitch_plan.gh
                    if kx0 <= x and x + w <= kx1 and ky0 <= y and y + h <= ky1:
                        kitch.clearzones.append((x - kx0, y - ky0, w, h, kind, owner))
            bed.clearzones = merge_clearances(bed.clearzones)
            if has_bath:
                bath.clearzones = merge_clearances(bath.clearzones)
            if has_liv:
                liv.clearzones = merge_clearances(liv.clearzones)
            if has_kitch:
                kitch.clearzones = merge_clearances(kitch.clearzones)
            self.bed_plan = bed
            self.bath_plan = bath
            self.liv_plan = liv
            self.kitch_plan = kitch
        else:
            self.bed_plan = self.plan
            self.bath_plan = None
            self.liv_plan = None
            self.kitch_plan = None


    def _solve_and_draw_preserve(self):
        """Rebuild the plan ONLY from previously placed items, no new furniture."""
        if self.sim_timer: self.root.after_cancel(self.sim_timer); self.sim_timer=None
        if self.sim2_timer: self.root.after_cancel(self.sim2_timer); self.sim2_timer=None
        self.sim_path=[]; self.sim_poly=[]; self.sim2_path=[]; self.sim2_poly=[]

        # Re-apply door/window positions from UI (doesn't add furniture)
        if not self._apply_openings_from_ui():
            self._draw()
            return
        if self.bath_dims:
            bath_wall, _, _ = self.bath_openings.door_span_cells()
            if bath_wall != WALL_LEFT:
                self.status.set('Bathroom door must be on shared wall.')
                self._draw()
                return

        # New empty grid for bedroom, then re-place exactly what the user already had
        best = GridPlan(self.bed_Wm, self.bed_Hm)
        for j in range(self.plan.gh):
            for i in range(self.plan.gw):
                code = self.plan.occ[j][i]
                if code:
                    best.occ[j][i] = code
        best.clearzones.extend(self.plan.clearzones)

        # Place back snapshot exactly (with front/bed clearances)
        sticky = getattr(self, '_sticky_items', [])
        FRONT_REC_DEFAULT = {'WRD': 0.80, 'DRS': 0.90, 'DESK': 0.90, 'TVU': 0.75, 'BST': 0.75}
        for (code,x,y,w,h,wall) in sticky:
            best.clear(x, y, w, h)
            best.place(x,y,w,h, code)
            # Re-apply clearances
            if code == 'BED':
                # No special method required here in preserve path
                pass

            else:
                fc_m = FRONT_REC_DEFAULT.get(code, 0.0)
                if fc_m > 0.0:
                    fc = best.meters_to_cells(fc_m)
                    if code == 'WRD' and fc == 4:
                        fc = 2
                    clear_w, clear_x = w, x
                    clear_h, clear_y = h, y
                    if code == 'WRD' and wall in (0, 2) and w > 1:
                        clear_w = w - 1
                        clear_x = x + 0.5
                    elif code == 'DRS':
                        fc = 2 if fc == 4 else 5 if fc == 3 else fc
                        rect_w = 5
                        rect_d = 2
                        clear_x = x + (w - rect_w) / 2
                        clear_y = y + (h - rect_w) / 2
                        if wall == 0:
                            best.mark_clear(clear_x, y + h, rect_w, rect_d, 'FRONT', code)
                        elif wall == 2:
                            best.mark_clear(clear_x, y - rect_d, rect_w, rect_d, 'FRONT', code)
                        elif wall == 3:
                            best.mark_clear(x + w, clear_y, rect_d, rect_w, 'FRONT', code)
                        elif wall == 1:
                            best.mark_clear(x - rect_d, clear_y, rect_d, rect_w, 'FRONT', code)
                        continue
                    if wall == 0:
                        best.mark_clear(clear_x, y + h, clear_w, fc, 'FRONT', code)
                    elif wall == 2:
                        best.mark_clear(clear_x, y - fc, clear_w, fc, 'FRONT', code)
                    elif wall == 3:
                        best.mark_clear(x + w, clear_y, fc, clear_h, 'FRONT', code)
                    elif wall == 1:
                        best.mark_clear(x - fc, clear_y, fc, clear_h, 'FRONT', code)

        best.clearzones = merge_clearances(best.clearzones)

        # Adopt as current plan, compute META minimally, draw
        self.plan = best
        self.bed_plan = best
        if self.bath_dims:
            self.bath_plan = arrange_bathroom(
                self.bath_dims[0],
                self.bath_dims[1],
                BATH_RULES,
                self.bath_openings,
                secondary_openings=self.bath_liv_openings,
            )
            add_door_clearance(self.bath_plan, self.bath_openings, 'DOOR')
            if self.bath_liv_openings:
                add_door_clearance(self.bath_plan, self.bath_liv_openings, 'LIVING_DOOR')
            if bed_wall == WALL_RIGHT:
                add_door_clearance(self.bed_plan, self.bed_openings, 'DOOR')
        if getattr(self, 'liv_dims', None):
            self.liv_plan = arrange_livingroom(
                self.liv_dims[0],
                self.liv_dims[1],
                LIV_RULES,
                openings=self.liv_openings,
            )
            add_door_clearance(self.liv_plan, self.liv_openings, 'DOOR')
            if getattr(self, 'liv_bath_openings', None):
                self.liv_bath_openings.p = self.liv_plan
                add_door_clearance(
                    self.liv_plan, self.liv_bath_openings, 'LIVING_DOOR'
                )
        if self.bath_dims or getattr(self, 'liv_dims', None):
            self._combine_plans()
        meta = {
            'coverage': getattr(best, 'coverage', lambda: 0.0)(),
            'paths_ok': True,
            'reach_windows': True,
            'score': 0.0,
        }
        self.meta = meta
        self._log_run(meta)
        self._draw()
        self.status.set("Layout preserved. No new furniture added.")


    # ------- adjacency & access rules

    def _adjacency_required_cells(self) -> Set[tuple]:
        """
        Return empty cells that represent must-touch adjacency to furniture faces:
          - Bed accessible from three free sides (all except wall face)
          - Side tables accessible from face opposite the wall
          - Wardrobe/Dresser/Desk/TVU accessible from face opposite the wall
        """
        p = self.plan
        gw, gh = p.gw, p.gh
        must = set()

        def add_face_cells(x, y, w, h, wall, faces):
            for face in faces:
                if face == 'N' and y-1 >= 0:
                    for i in range(w):
                        if p.occ[y-1][x+i] is None: must.add((x+i, y-1))
                if face == 'S' and y+h < gh:
                    for i in range(w):
                        if p.occ[y+h][x+i] is None: must.add((x+i, y+h))
                if face == 'W' and x-1 >= 0:
                    for j in range(h):
                        if p.occ[y+j][x-1] is None: must.add((x-1, y+j))
                if face == 'E' and x+w < gw:
                    for j in range(h):
                        if p.occ[y+j][x+w] is None: must.add((x+w, y+j))

        # BED
        for (bx, by, bw, bd, bwall) in self._components_by_code('BED'):
            faces = {'N','S','W','E'}
            if bwall == 0: faces.discard('N')
            elif bwall == 2: faces.discard('S')
            elif bwall == 3: faces.discard('W')
            elif bwall == 1: faces.discard('E')
            add_face_cells(bx, by, bw, bd, bwall, faces)

        # BST / WRD / DRS / DESK / TVU: opposite to wall
        face_map = {'BST': None, 'WRD': None, 'DRS': None, 'DESK': None, 'TVU': None}
        for code in list(face_map.keys()):
            for (x, y, w, h, wall) in self._components_by_code(code):
                if wall == 0: faces = {'S'}
                elif wall == 2: faces = {'N'}
                elif wall == 3: faces = {'E'}
                else: faces = {'W'}
                add_face_cells(x, y, w, h, wall, faces)

        return must

    # ------- simulations (full coverage with animation + RL)

    def _simulate_one(self):
        self.simulate_circulation()

    def simulate_circulation(self):
        """Build full-coverage path + animate one human block."""
        path, meta = self._build_full_coverage_path()
        self.sim_path = path
        self.sim_poly = meta['poly']
        self.sim_index = 0
        self._apply_rl_from_sim(meta)
        self._animate_one()

    def _simulate_two(self):
        """Two humans with different traversal orders."""
        path1, meta1 = self._build_full_coverage_path(order='rdlu')  # right,down,left,up
        path2, meta2 = self._build_full_coverage_path(order='lurd')  # left,up,right,down
        self.sim_path = path1; self.sim_poly = meta1['poly']; self.sim_index = 0
        self.sim2_path = path2; self.sim2_poly = meta2['poly']; self.sim2_index = 0
        self._apply_rl_from_sim(meta1)  # one RL update is enough per invocation
        self._animate_two()

    def _build_full_coverage_path(self, order='rdlu'):
        p = self.plan
        gw, gh = p.gw, p.gh

        # empty predicate
        def is_empty(x, y):
            c = p.occ[y][x]
            return c is None

        # door start fringe
        dx, dy, dw, dh = self.openings.door_rect_cells()
        if self.openings.door_wall == 0: start = (dx + dw//2, dh)
        elif self.openings.door_wall == 2: start = (dx + dw//2, gh - dh - 1)
        elif self.openings.door_wall == 3: start = (dw, dy + dh//2)
        else: start = (gw - dw - 1, dy + dh//2)

        free = {(x, y) for y in range(gh) for x in range(gw) if is_empty(x, y)}
        if start not in free:
            start = self._nearest_reachable_cell(start, free)

        targets = set(self._adjacency_required_cells())

        # window end candidates
        win_end = []
        for wall, s, L in self.openings.window_spans_cells():
            mid = s + max(0, L//2)
            if wall == 0 and (mid, 1) in free: win_end.append((mid, 1))
            elif wall == 2 and (mid, gh-2) in free: win_end.append((mid, gh-2))
            elif wall == 3 and (1, mid) in free: win_end.append((1, mid))
            elif wall == 1 and (gw-2, mid) in free: win_end.append((gw-2, mid))

        # neighbor ordering
        dir_map = {
            'r': (1,0), 'l': (-1,0), 'u': (0,-1), 'd': (0,1)
        }
        order_dirs = [dir_map[c] for c in order]

        def neighbors(x, y):
            for dx, dy in order_dirs:
                nx, ny = x+dx, y+dy
                if 0 <= nx < gw and 0 <= ny < gh:
                    yield nx, ny

        # DFS cover
        stack=[start]; seen=set(); path=[]; poly=[]
        collisions=0
        while stack:
            u = stack.pop()
            if u in seen: continue
            seen.add(u); path.append(u)
            if u not in free: collisions += 1
            if len(path)>=2:
                a=path[-2]; b=path[-1]
                ax,ay=self._cell_center(*a); bx,by=self._cell_center(*b)
                poly.append((ax,ay)); poly.append((bx,by))
            # push neighbors
            nbrs=[v for v in neighbors(*u) if v not in seen]
            stack.extend(nbrs)

        # ensure all targets
        from collections import deque
        def bfs_path(src, dst):
            if src == dst: return [src]
            q=deque([src]); parent={src:None}
            while q:
                u=q.popleft()
                for v in ((u[0]+1,u[1]),(u[0]-1,u[1]),(u[0],u[1]+1),(u[0],u[1]-1)):
                    if not (0<=v[0]<gw and 0<=v[1]<gh): continue
                    if v in parent: continue
                    if v not in free: continue
                    parent[v]=u
                    if v==dst: q.clear(); break
                    q.append(v)
            if dst not in parent: return None
            out=[]; cur=dst
            while cur is not None:
                out.append(cur); cur=parent[cur]
            return list(reversed(out))

        current = path[-1] if path else start
        misses=0
        for t in [t for t in targets if t not in set(path) and t in free]:
            sp=bfs_path(current,t)
            if not sp:
                misses += 1
                continue
            for s in sp[1:]:
                ax,ay=self._cell_center(*current); bx,by=self._cell_center(*s)
                poly.append((ax,ay)); poly.append((bx,by))
                path.append(s); current=s

        # finish at nearest window
        end_is_window=False
        if win_end:
            best=None; best_len=None
            for wcell in win_end:
                sp=bfs_path(current,wcell)
                if sp and (best_len is None or len(sp)<best_len):
                    best=sp; best_len=len(sp)
            if best:
                for s in best[1:]:
                    ax,ay=self._cell_center(*current); bx,by=self._cell_center(*s)
                    poly.append((ax,ay)); poly.append((bx,by))
                    path.append(s); current=s
                end_is_window=True

        coverage = len(set(path)&free)/max(1,len(free))
        rec = {
            "ts": time.time(),
            "event": "simulate_full_coverage",
            "steps": len(path),
            "coverage_free": coverage,
            "unmet_adjacency": misses,
            "end_is_window": end_is_window,
            "collisions": collisions,
            "path_len": len(path),
            "path_trunc": path[:1000],
        }
        try:
            with open(SIM_FILE,'a') as f: f.write(json.dumps(rec)+"\n")
        except Exception as e:
            print("SIM save error:", e)

        return path, {"poly": poly, "coverage": coverage, "misses": misses,
                      "end_is_window": end_is_window, "collisions": collisions}

    def _nearest_reachable_cell(self, start, pool:Set[tuple]):
        from collections import deque
        gw, gh = self.plan.gw, self.plan.gh
        def neighbors(x, y):
            for dx, dy in ((1,0),(-1,0),(0,1),(0,-1)):
                nx, ny = x+dx, y+dy
                if 0 <= nx < gw and 0 <= ny < gh:
                    yield nx, ny
        q = deque([start]); seen = {start}
        while q:
            u = q.popleft()
            if u in pool: return u
            for v in neighbors(*u):
                if v not in seen and (self.plan.occ[v[1]][v[0]] is None):
                    seen.add(v); q.append(v)
        return start

    # ------- RL integration

    def _apply_rl_from_sim(self, meta):
        coverage = meta["coverage"]
        misses = meta["misses"]
        end_is_window = meta["end_is_window"]
        collisions = meta["collisions"]

        Q = rl_load()
        s = rl_state_from_metrics(coverage, misses, end_is_window)
        a = rl_choose_action(Q, s, epsilon=0.15)
        # reward: high coverage, zero misses, end at window, no collisions
        reward = (coverage*3.0) + (1.0 if end_is_window else -0.5) + (-0.8*misses) + (-0.2*collisions)
        # mock next state as same (single-step update)
        ns = s
        Q = rl_update(Q, s, a, reward, ns)
        rl_save(Q)
        rl_apply_action_to_weights(self.weights, a, scale=0.25)
        self.status.set(self.status.get() + f" · RL nudged weights with action {a}")

    # ------- animation

    def _animate_one(self):
        if not self.sim_path: return
        self.sim_index = min(self.sim_index+1, len(self.sim_path)-1)
        i,j = self.sim_path[self.sim_index]
        self._draw()  # redraw and human block moves
        if self.sim_index < len(self.sim_path)-1:
            self.sim_timer = self.root.after(5, self._animate_one)  # hyperspeed

    def _animate_two(self):
        progressed = False
        if self.sim_path:
            self.sim_index = min(self.sim_index+1, len(self.sim_path)-1)
            progressed = progressed or (self.sim_index < len(self.sim_path)-1)
        if self.sim2_path:
            self.sim2_index = min(self.sim2_index+2, len(self.sim2_path)-1)  # slightly faster
            progressed = progressed or (self.sim2_index < len(self.sim2_path)-1)
        self._draw()
        if progressed:
            self.sim2_timer = self.root.after(5, self._animate_two)

    # ------- export

    def _export_png(self):
        # Tk doesn't export PNG natively; export EPS then let user convert
        path = filedialog.asksaveasfilename(defaultextension='.eps', filetypes=[('EPS','*.eps')], title='Export as EPS')
        if not path: return
        try:
            self.canvas.postscript(file=path, colormode='color')
            messagebox.showinfo('Export', f'Exported drawing to {path}\nTip: convert EPS→PNG via Preview or ImageMagick.')
        except Exception as e:
            messagebox.showerror('Export failed', str(e))

    # ------- logging

    def _log_event(self, obj):
        obj['ts'] = time.time()
        append_jsonl_locked(SIM_FILE, obj)

    def _grid_snapshot(self, plan: 'GridPlan', max_hw: int = 16):
        mapping = {'BED':1,'BST':2,'WRD':3,'DRS':4,'DESK':5,'TVU':6}
        H = min(max_hw, plan.gh); W = min(max_hw, plan.gw)
        sx = max(1, plan.gw // W); sy = max(1, plan.gh // H)
        G = np.zeros((H, W), dtype=np.int8)
        jj = 0
        for y in range(0, plan.gh, sy):
            ii = 0
            for x in range(0, plan.gw, sx):
                c = plan.occ[y][x]
                if c:
                    base = c.split(':')[0]
                    G[jj, ii] = mapping.get(base, 7)
                ii += 1
                if ii >= W:
                    break
            jj += 1
            if jj >= H:
                break
        return G

    def _grid_for_log(self):
        plans = [self.bed_plan]
        if self.bath_plan:
            plans.append(self.bath_plan)
        if getattr(self, 'liv_plan', None):
            plans.append(self.liv_plan)
        grids = [self._grid_snapshot(p, 16) for p in plans if p is not None]
        if not grids:
            return []
        max_h = max(g.shape[0] for g in grids)
        combined = []
        for j in range(max_h):
            row = []
            for g in grids:
                h, w = g.shape
                if j < h:
                    row.extend(int(v) for v in g[j])
                else:
                    row.extend([0] * w)
            combined.append(row)
        return combined


    def _log_run(self, meta):
        rec = {
            "ts": time.time(),
            "event": "solve_result",
            "features": meta.get('features', {}),
            "score": meta.get('score', 0.0),
            "coverage": meta.get('coverage', 0.0),
            "reach_windows": meta.get('reach_windows', False)
        }
        append_jsonl_locked(SIM_FILE, rec)

# -----------------------
# App shell (final fix: wait for dialogs)
# -----------------------

# ---- REPLACE FROM HERE -------------------------------------------------------

class App:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title('VASTU – Bedroom (Generator)')
        apply_modern_theme(self.root)
        self.root.protocol('WM_DELETE_WINDOW', self._on_close)
        # landing container (always visible fallback)
        self.landing = ttk.Frame(self.root, padding=16)
        self.landing.pack(fill=tk.BOTH, expand=True)
        self._build_landing()

    def _build_landing(self):
        for w in list(self.landing.children.values()):
            w.destroy()
        ttk.Label(self.landing, text='VASTU – Bedroom', font=('SF Pro Text', 18, 'bold')).pack(pady=(8,2))
        ttk.Label(self.landing, text='Click Start to choose mode and room inputs.').pack(pady=(0,12))
        btns = ttk.Frame(self.landing); btns.pack()
        ttk.Button(btns, text='Start', style='Primary.TButton', command=self._startup_flow).pack(side=tk.LEFT, padx=6)
        ttk.Button(btns, text='Open Default Generate', command=lambda: self._open_generate_default()).pack(side=tk.LEFT, padx=6)

    def _on_close(self):
        try: self.root.destroy()
        except: pass

    def run(self):
        # Show window up front, even on macOS
        self.root.geometry('1200x800')
        self.root.update_idletasks()
        self.root.deiconify()
        self.root.lift()
        try:
            # toggle topmost briefly to beat macOS “behind other apps” quirk
            self.root.attributes('-topmost', True)
            self.root.after(300, lambda: self.root.attributes('-topmost', False))
        except Exception:
            pass
        self.root.mainloop()

    # ---------- startup flow (modal dialogs, but safe) ----------

    def _startup_flow(self):
        # Bring to front again before showing dialogs
        self.root.lift()
        self.root.update()

        # 1) Mode dialog
        md = ModeDialog(self.root)
        self.root.wait_window(md)
        if not getattr(md, 'result', None):
            # user cancelled; keep landing visible
            return
        mode = md.result
        

        # 2) Room input dialog capturing both rooms
        label = 'Sketch' if mode == 'sketch' else 'Generate'
        try:
            cd = AreaDialogCombined(self.root, label)
            self.root.wait_window(cd)
            if not getattr(cd, 'result', None):
                return
            bed_res = cd.result.get('bedroom', {})
            bath_res = cd.result.get('bathroom', {})
            liv_res = cd.result.get('livingroom', {})
            kitch_res = cd.result.get('kitchen', {})
        except Exception:
            bed_res = {"mode": "dims", "W": 4.2, "H": 3.0, "len_units": "m", "bed": "Auto"}
            bath_res = {"mode": "dims", "W": 2.4, "H": 1.8, "len_units": "m", "bed": "Auto"}
            liv_res = {"mode": "dims", "W": 3.0, "H": 3.0, "len_units": "m", "bed": "Auto"}
            kitch_res = {"mode": "dims", "W": 3.0, "H": 3.0, "len_units": "m", "bed": "Auto"}

        bed_dims = self._compute_dims_from_result(bed_res)
        bath_dims = self._compute_dims_from_result(bath_res)
        liv_dims = self._compute_dims_from_result(liv_res) if liv_res else None
        kitch_dims = self._compute_dims_from_result(kitch_res) if kitch_res else None

        # open the chosen workspace
        self._open_workspace(mode, bed_dims, bath_dims, liv_dims, kitch_dims)

    def _compute_dims_from_result(self, res: Dict) -> Tuple[float,float,Optional[str]]:
        if res.get("mode") == "area":
            A = float(res.get("area", 12.0))
            unit = res.get("area_units", "m²")
            A_m2 = A * AREA_UNIT_TO_M2.get(unit, 1.0)
            # choose square-ish box
            Wm = max(2.6, float(np.sqrt(A_m2)))
            Hm = max(2.6, float(A_m2 / Wm))
        else:
            W = float(res.get("W", 4.2)); H = float(res.get("H", 3.0))
            lu = res.get("len_units", "m")
            if lu == "ft":
                Wm = W * FT_TO_M; Hm = H * FT_TO_M
            else:
                Wm, Hm = W, H
        bed_key = res.get("bed", "Auto")
        return Wm, Hm, (None if bed_key == 'Auto' else bed_key)

    def _open_workspace(
        self,
        mode: str,
        bed_dims: Tuple[float, float, Optional[str]],
        bath_dims: Tuple[float, float, Optional[str]],
        liv_dims: Optional[Tuple[float, float, Optional[str]]] = None,
        kitch_dims: Optional[Tuple[float, float, Optional[str]]] = None,
    ):
      
        # clear landing and any previous workspaces so only one view remains
        for child in self.root.winfo_children():
            if child is not self.landing:
                try:
                    child.destroy()
                except Exception:
                    pass

        self.landing.pack_forget()
        for w in list(self.landing.children.values()):
            try:
                w.destroy()
            except Exception:
                pass
        if mode == 'sketch':
            Wm, Hm, _ = bed_dims
            SketchGrid(self.root, int(round((Wm*Hm)/(CELL_M*CELL_M))), 'm', on_back=self._back_to_landing)
        else:
            Wb, Hb, bed_key = bed_dims
            Wc, Hc, _ = bath_dims
            if liv_dims:
                Wl, Hl, _ = liv_dims
                liv_tuple = (Wl, Hl)
            else:
                liv_tuple = None
            if kitch_dims:
                Wk, Hk, _ = kitch_dims
                kitch_tuple = (Wk, Hk)
            else:
                kitch_tuple = None
            GenerateView(
                self.root,
                Wb,
                Hb,
                bed_key,
                room_label='Plan',
                bath_dims=(Wc, Hc),
                liv_dims=liv_tuple,
                kitch_dims=kitch_tuple,
                pack_side=tk.LEFT,
                on_back=self._back_to_landing,
            )


    def _back_to_landing(self):
        # remove any leftover top-level frames that views added
        for child in self.root.winfo_children():
            # keep only our landing frame
            if child is not self.landing:
                try: child.destroy()
                except: pass
        self.landing.pack(fill=tk.BOTH, expand=True)
        self._build_landing()

    def _open_generate_default(self):
        # Quick path: default bedroom and bathroom sizes
        bed_dims = (4.2, 3.0, None)
        bath_dims = (2.4, 1.8, None)
        self._open_workspace('generate', bed_dims, bath_dims, None, None)

# ---- AND REPLACE YOUR MAIN GUARD WITH THIS -----------------------------------

if __name__ == "__main__":
    App().run()

# ---- END REPLACEMENT ---------------------------------------------------------
