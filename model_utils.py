import os
import json
import random
import sys
import numpy as np
import networkx as nx
from tqdm import tqdm
from rdkit import Chem

# PyTorch
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import autocast, GradScaler

# Metrics
from sklearn.metrics import r2_score
from scipy.stats import pearsonr, spearmanr

# ---------------------------
# 1. 全局配置与环境
# ---------------------------
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
torch.set_float32_matmul_precision('high')
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True

# 路径配置
DATA_PATH = "Data/KIBA/"
PLM_FEAT_PATH = os.path.join(DATA_PATH, "plm_features_SOTA", "plm_embeddings_sota_final.pt")
CACHE_DIR = os.path.join(DATA_PATH, "cache_kiba_max_speed")
os.makedirs(CACHE_DIR, exist_ok=True)

# 超参数
PLM_DRUG_DIM = 768
PLM_PROT_DIM = 2560
SMILES_MAX_LEN = 512
PROT_MAX_LEN = 1024
DRUG_FEAT_DIM_RAW = 78
PROT_FEAT_DIM_RAW = 21

GNN_HID = 512
GNN_OUT = 512
DROPOUT = 0.25

BATCH_SIZE = 128
EPOCHS = 200
PATIENCE = 60
MAX_LR = 2e-4
WEIGHT_DECAY = 1e-4

ALPHA_MSE = 1.0
ALPHA_RANK = 0.5
ALPHA_CL = 0.05

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BINARY_THRESHOLD = 12.1
_SOFTMAX_NEG_INF = -1e4

# 混合精度配置
try:
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        AMP_TYPE = torch.bfloat16
        USE_SCALER = False
        print("[INFO] Acceleration: bfloat16 enabled (No Scaler).")
    else:
        AMP_TYPE = torch.float16
        USE_SCALER = True
        print("[INFO] Acceleration: float16 enabled (With Scaler).")
except:
    AMP_TYPE = torch.float16
    USE_SCALER = True

# ---------------------------
# 2. 数据与图处理函数
# ---------------------------
def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

def load_data_and_plm():
    files_needed = {
        "Drugs": os.path.join(DATA_PATH, "drug_smiles.json"),
        "Proteins": os.path.join(DATA_PATH, "protein_seq.json"),
        "Affinity": os.path.join(DATA_PATH, "Y.txt")
    }
    for name, path in files_needed.items():
        if not os.path.exists(path):
            print(f"[FATAL] missing {name} at {path}")
            sys.exit(1)

    with open(files_needed["Drugs"]) as f:
        drug_data = json.load(f)
        drug_ids = sorted(drug_data.keys())
        drugs = [drug_data[k] for k in drug_ids]
    with open(files_needed["Proteins"]) as f:
        prot_data = json.load(f)
        prot_ids = sorted(prot_data.keys())
        prots = [prot_data[k] for k in prot_ids]

    Y = np.loadtxt(files_needed["Affinity"])
    n_drugs = len(drug_ids)
    n_prots = len(prot_ids)
    if Y.shape == (n_prots, n_drugs): Y = Y.T

    drug_indices, prot_indices, aff_values = [], [], []
    for i in range(n_drugs):
        for j in range(n_prots):
            if not np.isnan(Y[i, j]):
                drug_indices.append(i)
                prot_indices.append(j)
                aff_values.append(Y[i, j])

    if not os.path.exists(PLM_FEAT_PATH):
        sys.exit(f"[FATAL] PLM features not found: {PLM_FEAT_PATH}")

    plm_data = torch.load(PLM_FEAT_PATH, map_location='cpu')
    d_id2idx = plm_data['drug_id2idx']
    p_id2idx = plm_data['prot_id2idx']
    d_gather_idx = torch.tensor([d_id2idx[k] for k in drug_ids], dtype=torch.long)
    p_gather_idx = torch.tensor([p_id2idx[k] for k in prot_ids], dtype=torch.long)

    raw_d_feat = plm_data['drug_feats'].float()
    raw_p_feat = plm_data['prot_feats'].float()
    raw_d_mask = plm_data['drug_masks'].float()
    raw_p_mask = plm_data['prot_masks'].float()

    d_feat = raw_d_feat[d_gather_idx]
    d_mask = raw_d_mask[d_gather_idx]
    p_feat = raw_p_feat[p_gather_idx]
    p_mask = raw_p_mask[p_gather_idx]

    if d_feat.shape[1] > SMILES_MAX_LEN:
        d_feat = d_feat[:, :SMILES_MAX_LEN]
        d_mask = d_mask[:, :SMILES_MAX_LEN]
    if p_feat.shape[1] > PROT_MAX_LEN:
        p_feat = p_feat[:, :PROT_MAX_LEN]
        p_mask = p_mask[:, :PROT_MAX_LEN]

    return drugs, prots, torch.nan_to_num(d_feat), d_mask, torch.nan_to_num(p_feat), p_mask, \
        torch.tensor(drug_indices, dtype=torch.long), torch.tensor(prot_indices, dtype=torch.long), \
        torch.tensor(aff_values, dtype=torch.float32)

def one_hot_embedding(value, options):
    embedding = [0] * (len(options) + 1)
    index = options.index(value) if value in options else -1
    if index >= 0: embedding[index] = 1
    return embedding

def get_atom_features(atom):
    possible_atoms = ['C', 'N', 'O', 'S', 'F', 'P', 'Cl', 'Br', 'I', 'H']
    atom_type = one_hot_embedding(atom.GetSymbol(), possible_atoms)
    possible_degrees = list(range(11))
    degree = one_hot_embedding(atom.GetTotalDegree(), possible_degrees)
    possible_implicit_valence = list(range(7))
    implicit_valence = one_hot_embedding(atom.GetImplicitValence(), possible_implicit_valence)
    aromatic = [1] if atom.GetIsAromatic() else [0]
    feat = np.array(atom_type + degree + implicit_valence + aromatic, dtype=np.float32)
    pad_len = DRUG_FEAT_DIM_RAW - feat.shape[0]
    if pad_len > 0:
        feat = np.concatenate([feat, np.zeros(pad_len, dtype=np.float32)])
    else:
        feat = feat[:DRUG_FEAT_DIM_RAW]
    return feat

def get_residue_features(residue_char):
    aa_list = list("ACDEFGHIKLMNPQRSTVWY")
    aa_map = {aa: i for i, aa in enumerate(aa_list)}
    idx = aa_map.get(residue_char.upper(), -1)
    feat = np.zeros(PROT_FEAT_DIM_RAW, dtype=np.float32)
    if 0 <= idx < PROT_FEAT_DIM_RAW: feat[idx] = 1.0
    return feat

def build_drug_graph(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None: return nx.Graph()
    g = nx.Graph()
    for atom in mol.GetAtoms():
        feat = get_atom_features(atom)
        g.add_node(atom.GetIdx(), feat=feat)
    for bond in mol.GetBonds():
        g.add_edge(bond.GetBeginAtomIdx(), bond.GetEndAtomIdx())
    return g

def build_prot_graph(seq):
    g = nx.Graph()
    seq = seq[:PROT_MAX_LEN]
    for i, aa in enumerate(seq):
        feat = get_residue_features(aa)
        g.add_node(i, feat=feat)
    for i in range(len(seq) - 1):
        g.add_edge(i, i + 1)
    return g

def process_graph(G):
    N = G.number_of_nodes()
    if N == 0: return None, None
    x = np.array([G.nodes[n]['feat'] for n in sorted(G.nodes())], dtype=np.float32)
    A = np.asarray(nx.adjacency_matrix(G).todense(), dtype=np.int8)
    return x, A

def prepad_graphs(graphs, feat_dim):
    max_nodes = 0
    valid = []
    for x, A in graphs:
        if x is None:
            x = np.zeros((1, feat_dim), dtype=np.float32)
            A = np.eye(1, dtype=np.int8)
        max_nodes = max(max_nodes, x.shape[0])
        valid.append((x, A))

    num = len(valid)
    X_t = torch.zeros((num, max_nodes, feat_dim), dtype=torch.float32)
    A_t = torch.zeros((num, max_nodes, max_nodes), dtype=torch.int8)
    M_t = torch.zeros((num, max_nodes), dtype=torch.float32)
    A_norm_t = torch.zeros((num, max_nodes, max_nodes), dtype=torch.float16)

    for i, (x, A) in enumerate(tqdm(valid, desc="Padding", leave=False)):
        N = x.shape[0]
        X_t[i, :N, :] = torch.from_numpy(x)
        A_t[i, :N, :N] = torch.from_numpy(A)
        M_t[i, :N] = 1.0
        if N > 0:
            A_float = A.astype(np.float32)
            A_mat = torch.from_numpy(A_float) + torch.eye(N, dtype=torch.float32)
            deg = A_mat.sum(1)
            deg_inv_sqrt = torch.pow(deg, -0.5)
            deg_inv_sqrt[torch.isinf(deg_inv_sqrt)] = 0.0
            D = torch.diag(deg_inv_sqrt)
            A_norm = D.mm(A_mat).mm(D)
            A_norm_t[i, :N, :N] = A_norm.to(torch.float16)
    return X_t, A_t, M_t, A_norm_t

# ---------------------------
# 3. 模型类定义
# ---------------------------
class RobustProjector(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        bottleneck = 256
        self.proj = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, bottleneck),
            nn.GELU(),
            nn.LayerNorm(bottleneck),
            nn.Dropout(DROPOUT),
            nn.Linear(bottleneck, out_dim),
            nn.LayerNorm(out_dim)
        )
    def forward(self, x):
        return self.proj(x)

class GINLayer(nn.Module):
    def __init__(self, ind, outd):
        super().__init__()
        self.ind = ind
        self.outd = outd
        self.mlp = nn.Sequential(
            nn.Linear(ind if ind == outd else ind, outd),
            nn.LayerNorm(outd),
            nn.GELU(),
            nn.Dropout(DROPOUT),
            nn.Linear(outd, outd),
            nn.LayerNorm(outd),
            nn.GELU()
        )
        self.eps = nn.Parameter(torch.tensor(0.0))
    def forward(self, x, adj_norm, node_mask):
        if x.dim() == 2: x = x.unsqueeze(0)
        mask = node_mask.unsqueeze(-1)
        x = x * mask
        neigh = torch.matmul(adj_norm, x)
        out = neigh + self.eps * x
        B, N, F = out.shape
        out = self.mlp(out.view(-1, F)).view(B, N, -1)
        if self.ind == self.outd: out = out + x
        return out * mask

class GATLayer(nn.Module):
    def __init__(self, ind, outd, dropout=DROPOUT, alpha=0.2):
        super().__init__()
        self.fc = nn.Linear(ind, outd, bias=False)
        self.attn_l = nn.Linear(outd, 1, bias=False)
        self.attn_r = nn.Linear(outd, 1, bias=False)
        self.leaky = nn.LeakyReLU(alpha)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(outd)
        self.use_residual = (ind == outd)
    def forward(self, x, adj_bool, node_mask):
        if x.dim() == 2: x = x.unsqueeze(0)
        mask = node_mask.unsqueeze(-1)
        x = x * mask
        h = self.fc(x)
        e_l = self.attn_l(h)
        e_r = self.attn_r(h).permute(0, 2, 1)
        e = e_l + e_r
        e = self.leaky(e)
        node_pair_mask = (node_mask.unsqueeze(1) * node_mask.unsqueeze(2))
        allowed = (adj_bool > 0) & (node_pair_mask.bool())
        e_masked = e.masked_fill(~allowed, _SOFTMAX_NEG_INF)
        attention = F.softmax(e_masked, dim=-1)
        attention = torch.nan_to_num(attention, nan=0.0)
        out = torch.matmul(self.dropout(attention), h)
        if self.use_residual: out = out + x
        out = self.norm(out * mask)
        return out

class GIN_GAT_Encoder(nn.Module):
    def __init__(self, in_dim, hid_dim, out_dim):
        super().__init__()
        self.gin1 = GINLayer(in_dim, hid_dim)
        self.gin2 = GINLayer(hid_dim, hid_dim)
        self.gat = GATLayer(hid_dim, out_dim)
    def forward(self, x, adj_norm, adj_bool, node_mask):
        x = self.gin1(x, adj_norm, node_mask)
        x = self.gin2(x, adj_norm, node_mask)
        x = self.gat(x, adj_bool, node_mask)
        return F.elu(x)

class AttentionPooling(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.attn = nn.Sequential(
            nn.Linear(in_dim, max(8, in_dim // 2)),
            nn.Tanh(),
            nn.Linear(max(8, in_dim // 2), 1)
        )
    def forward(self, x, mask=None):
        scores = self.attn(x).squeeze(-1)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, _SOFTMAX_NEG_INF)
        weights = F.softmax(scores, dim=1).unsqueeze(-1)
        if mask is not None: weights = weights * mask.unsqueeze(-1)
        return torch.sum(x * weights, dim=1)

class InteractivePooling(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.W_self = nn.Linear(dim, dim)
        self.W_ctx = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, 1, bias=False)
    def forward(self, x_self, x_context_global, mask=None):
        if x_self.dim() == 2: return x_self
        h_self = self.W_self(x_self)
        h_ctx = self.W_ctx(x_context_global).unsqueeze(1)
        energy = torch.tanh(h_self + h_ctx)
        scores = self.v(energy).squeeze(-1)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, _SOFTMAX_NEG_INF)
        weights = F.softmax(scores, dim=1).unsqueeze(-1)
        if mask is not None: weights = weights * mask.unsqueeze(-1)
        return torch.sum(x_self * weights, dim=1)

class SimilarityGraphBlock_Fast(nn.Module):
    def __init__(self, dim, dropout=DROPOUT):
        super().__init__()
        self.dim = dim
        self.dropout = dropout
        self.norm = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(nn.Linear(dim, dim), nn.GELU(), nn.Linear(dim, dim))
    def forward(self, x):
        if x.dim() == 2 and x.size(0) <= 1:
            return x
        x_norm = x / (torch.norm(x, dim=-1, keepdim=True) + 1e-6)
        q = (x_norm * self.dim).unsqueeze(0).unsqueeze(0)
        k = x_norm.unsqueeze(0).unsqueeze(0)
        v = x.unsqueeze(0).unsqueeze(0)
        B_size = x.size(0)
        diag_mask = torch.eye(B_size, device=x.device, dtype=torch.bool)
        x_agg = F.scaled_dot_product_attention(
            query=q, key=k, value=v,
            attn_mask=diag_mask.unsqueeze(0).unsqueeze(0),
            dropout_p=self.dropout if self.training else 0.0
        )
        x_agg = x_agg.squeeze(0).squeeze(0)
        out = x + self.mlp(x_agg)
        return self.norm(out)

class AdaptiveGatedFusion(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.inter_proj = nn.Linear(dim, dim)
        self.gate_net = nn.Sequential(nn.Linear(dim * 2, dim), nn.GELU(), nn.Linear(dim, dim), nn.Sigmoid())
        self.alpha_net = nn.Sequential(nn.Linear(dim * 2, dim), nn.GELU(), nn.Linear(dim, dim), nn.Tanh())
        self.norm = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(DROPOUT)
    def forward(self, a, b):
        interact = self.inter_proj(a * b)
        x = torch.cat([a, b], dim=-1)
        z = self.gate_net(x)
        alpha = self.alpha_net(x)
        out = z * a + (1.0 - z) * b + alpha * interact
        out = self.dropout(out)
        return self.norm(out)

class Model_Hybrid(nn.Module):
    def __init__(self):
        super().__init__()
        self.d_proj = RobustProjector(PLM_DRUG_DIM, GNN_OUT)
        self.p_proj = RobustProjector(PLM_PROT_DIM, GNN_OUT)
        self.d_gnn = GIN_GAT_Encoder(DRUG_FEAT_DIM_RAW, GNN_HID, GNN_OUT)
        self.p_gnn = GIN_GAT_Encoder(PROT_FEAT_DIM_RAW, GNN_HID, GNN_OUT)
        self.pool = AttentionPooling(GNN_OUT)
        self.inter_d = InteractivePooling(GNN_OUT)
        self.inter_p = InteractivePooling(GNN_OUT)
        self.sim_d = SimilarityGraphBlock_Fast(GNN_OUT)
        self.sim_p = SimilarityGraphBlock_Fast(GNN_OUT)
        self.fuse_d = AdaptiveGatedFusion(GNN_OUT)
        self.fuse_p = AdaptiveGatedFusion(GNN_OUT)
        self.classifier = nn.Sequential(
            nn.Linear(GNN_OUT * 2, 1024),
            nn.LayerNorm(1024),
            nn.GELU(),
            nn.Dropout(DROPOUT),
            nn.Linear(1024, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Dropout(DROPOUT),
            nn.Linear(512, 1)
        )
    def forward(self, d_plm, d_mask, p_plm, p_mask, dx, dadj_norm, dadj_bool, dnode_mask, px, padj_norm, padj_bool, pnode_mask):
        ed = self.d_proj(d_plm)
        ep = self.p_proj(p_plm)
        gd = self.d_gnn(dx, dadj_norm, dadj_bool, dnode_mask)
        gp = self.p_gnn(px, padj_norm, padj_bool, pnode_mask)
        ctx_d = self.pool(ed, d_mask)
        ctx_p = self.pool(ep, p_mask)
        ctx_gd = self.pool(gd, dnode_mask)
        ctx_gp = self.pool(gp, pnode_mask)
        vec_d_plm = self.inter_d(ed, ctx_p, d_mask)
        vec_p_plm = self.inter_p(ep, ctx_d, p_mask)
        vec_d_plm = self.sim_d(vec_d_plm)
        vec_p_plm = self.sim_p(vec_p_plm)
        vec_d_gnn = self.inter_d(gd, ctx_gp, dnode_mask)
        vec_p_gnn = self.inter_p(gp, ctx_gd, pnode_mask)
        final_d = self.fuse_d(vec_d_plm, vec_d_gnn)
        final_p = self.fuse_p(vec_p_plm, vec_p_gnn)
        combined = torch.cat([final_d, final_p], dim=-1)
        pred = self.classifier(combined).squeeze(-1)
        return pred, vec_d_plm, vec_p_plm

# ---------------------------
# 4. Loss 与 评价指标
# ---------------------------
def pairwise_ranking_loss(y_pred, y_true):
    B = y_true.shape[0]
    if B > 512:
        idx = torch.randperm(B, device=y_true.device)[:512]
        y_true = y_true[idx]
        y_pred = y_pred[idx]
    diff = y_true.unsqueeze(1) - y_true.unsqueeze(0)
    pred_diff = y_pred.unsqueeze(1) - y_pred.unsqueeze(0)
    loss = torch.relu(0.1 - torch.sign(diff) * pred_diff)
    mask = (diff.abs() > 1e-3).float()
    return (loss * mask).sum() / (mask.sum() + 1e-6)

def supcon_loss(features, labels, temperature=0.07, num_bins=10):
    device = features.device
    features = F.normalize(features, dim=1)
    try:
        quantiles = torch.linspace(0.0, 1.0, steps=num_bins + 1, device=device)
        bins = torch.quantile(labels, quantiles)
        labels_q = torch.bucketize(labels, bins, right=False)
    except:
        return torch.tensor(0.0, device=device)
    mask = torch.eq(labels_q.unsqueeze(1), labels_q.unsqueeze(0)).float()
    logits = torch.matmul(features, features.T) / temperature
    logits_max, _ = torch.max(logits, dim=1, keepdim=True)
    logits = logits - logits_max.detach()
    logits_mask = 1.0 - torch.eye(features.shape[0], device=device)
    mask = mask * logits_mask
    exp_logits = torch.exp(logits) * logits_mask
    log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + 1e-6)
    mean_log_prob_pos = (mask * log_prob).sum(1) / (mask.sum(1) + 1e-6)
    valid = (mask.sum(1) > 0).float()
    if valid.sum() == 0: return torch.tensor(0.0, device=device)
    loss = -(valid * mean_log_prob_pos).sum() / (valid.sum() + 1e-6)
    return loss

def calculate_rm2(y_true, y_pred):
    try:
        y_true = np.array(y_true).flatten()
        y_pred = np.array(y_pred).flatten()
        r2 = r2_score(y_true, y_pred)
        numerator = np.sum(y_true * y_pred)
        denominator = np.sum(y_pred ** 2)
        if denominator == 0: return 0.0
        k = numerator / denominator
        y_pred_0 = k * y_pred
        sse_0 = np.sum((y_true - y_pred_0) ** 2)
        sst = np.sum((y_true - np.mean(y_true)) ** 2)
        if sst == 0: return 0.0
        r02 = 1 - (sse_0 / sst)
        rm2 = r2 * (1 - np.sqrt(np.abs(r2 - r02)))
        return rm2
    except:
        return 0.0

def fast_concordance_index_vectorized(y_true, y_pred, max_pairs=20000):
    try:
        t = np.asarray(y_true)
        p = np.asarray(y_pred)
        n = len(t)
        if n > max_pairs:
            idx = np.random.choice(n, max_pairs, replace=False)
            t = t[idx]
            p = p[idx]
        perm = np.argsort(t)
        t = t[perm]
        p = p[perm]
        dt = t[:, None] - t[None, :]
        dp = p[:, None] - p[None, :]
        mask = dt > 0
        concordant = (dp > 0) & mask
        ties = (dp == 0) & mask
        n_pairs = mask.sum()
        if n_pairs == 0: return 0.0
        return (concordant.sum() + 0.5 * ties.sum()) / n_pairs
    except:
        return 0.0

def compute_metrics(y_true, y_pred):
    y_t = y_true.detach().float().cpu().numpy().flatten()
    y_p = y_pred.detach().float().cpu().numpy().flatten()
    mse = ((y_t - y_p) ** 2).mean()
    r2 = r2_score(y_t, y_p)
    rm2 = calculate_rm2(y_t, y_p)
    ci = fast_concordance_index_vectorized(y_t, y_p)
    try:
        pearson = pearsonr(y_t, y_p)[0]
        spearman = spearmanr(y_t, y_p)[0]
    except:
        pearson = 0.0
        spearman = 0.0
    return {
        "MSE": mse, "CI": ci,  "rm2": rm2,
        "Pearson": pearson, "Spearman": spearman
    }