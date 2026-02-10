import os
import gc
import pandas as pd
import torch
from tqdm import tqdm
from torch.optim.lr_scheduler import OneCycleLR
from torch.amp import autocast, GradScaler
import torch.nn.functional as F

# 导入 model_utils 中定义的所有内容
from model_utils import *

def run_cycle(seed, run_idx, data):
    drugs, prots, d_plm, d_mask, p_plm, p_mask, d_idx, p_idx, aff = data

    # Split
    seed_everything(seed)
    total_len = len(aff)
    indices = np.arange(total_len)
    np.random.shuffle(indices)
    split_1 = int(0.1 * total_len)
    split_2 = int(0.2 * total_len)
    te_idx = indices[:split_1]
    val_idx = indices[split_1:split_2]
    tr_idx = indices[split_2:]

    # To GPU
    tr_idx_gpu = torch.tensor(tr_idx, device=DEVICE)
    val_idx_gpu = torch.tensor(val_idx, device=DEVICE)
    te_idx_gpu = torch.tensor(te_idx, device=DEVICE)

    # Model
    model = Model_Hybrid().to(DEVICE)

    try:
        print("[INFO] Compiling model (Default Mode)...")
        model = torch.compile(model)
    except Exception as e:
        print(f"[WARN] Compilation failed: {e}. Running in eager mode.")

    try:
        optimizer = torch.optim.AdamW(model.parameters(), lr=MAX_LR, weight_decay=WEIGHT_DECAY, fused=True)
        print("[INFO] Using FusedAdamW.")
    except:
        optimizer = torch.optim.AdamW(model.parameters(), lr=MAX_LR, weight_decay=WEIGHT_DECAY)

    steps_per_epoch = (len(tr_idx) + BATCH_SIZE - 1) // BATCH_SIZE
    scheduler = OneCycleLR(optimizer, max_lr=MAX_LR, epochs=EPOCHS, steps_per_epoch=steps_per_epoch, pct_start=0.3)
    scaler = GradScaler(enabled=USE_SCALER)

    # Resident Data (Static)
    G_d_plm = d_plm.to(DEVICE, dtype=AMP_TYPE)
    G_d_mask = d_mask.to(DEVICE, dtype=AMP_TYPE)
    G_p_plm = p_plm.to(DEVICE, dtype=AMP_TYPE)
    G_p_mask = p_mask.to(DEVICE, dtype=AMP_TYPE)
    G_idx_d = d_idx.to(DEVICE)
    G_idx_p = p_idx.to(DEVICE)
    G_aff = aff.to(DEVICE)

    # Load Graph Cache
    graph_cache = os.path.join(CACHE_DIR, "graphs_processed_maxspeed.pt")
    if not os.path.exists(graph_cache):
        print("[INFO] Building graphs (first run)...")
        d_gs = [process_graph(build_drug_graph(s)) for s in tqdm(drugs)]
        p_gs = [process_graph(build_prot_graph(p)) for p in tqdm(prots)]
        d_tensors = prepad_graphs(d_gs, DRUG_FEAT_DIM_RAW)
        p_tensors = prepad_graphs(p_gs, PROT_FEAT_DIM_RAW)
        torch.save({'drug': d_tensors, 'prot': p_tensors}, graph_cache)

    g_data = torch.load(graph_cache, map_location='cpu')

    G_dX = g_data['drug'][0].to(DEVICE, dtype=torch.float32)
    G_dA_bool = (g_data['drug'][1].to(DEVICE) > 0).to(torch.bool)
    G_dA_norm = g_data['drug'][3].to(DEVICE, dtype=AMP_TYPE)
    G_dM = g_data['drug'][2].to(DEVICE, dtype=AMP_TYPE)

    G_pX = g_data['prot'][0].to(DEVICE, dtype=torch.float32)
    G_pA_bool = (g_data['prot'][1].to(DEVICE) > 0).to(torch.bool)
    G_pA_norm = g_data['prot'][3].to(DEVICE, dtype=AMP_TYPE)
    G_pM = g_data['prot'][2].to(DEVICE, dtype=AMP_TYPE)

    # Resume Logic
    ckpt_path = os.path.join(CACHE_DIR, f"ckpt_seed_{seed}.pth")
    best_mse = 999.0
    start_epoch = 0
    early_stop = 0
    best_state = None

    if os.path.exists(ckpt_path):
        print(f"[RESUME] Loading {ckpt_path}...")
        ckpt = torch.load(ckpt_path)
        try:
            model.load_state_dict(ckpt['model'])
        except:
            current_dict = model.state_dict()
            saved_dict = ckpt['model']
            new_dict = {}
            for k, v in saved_dict.items():
                k_new = k.replace("_orig_mod.", "")
                new_dict[k_new] = v
            model.load_state_dict(new_dict, strict=False)

        optimizer.load_state_dict(ckpt['optim'])
        scheduler.load_state_dict(ckpt['sched'])

        correct_total_steps = steps_per_epoch * EPOCHS
        if scheduler.total_steps != correct_total_steps:
            print(f"[FIX] Correcting scheduler total_steps from {scheduler.total_steps} to {correct_total_steps}")
            scheduler.total_steps = correct_total_steps

        start_epoch = ckpt['epoch'] + 1
        best_mse = ckpt['best_mse']
        best_state = ckpt['best_state']
        early_stop = ckpt['early_stop']

    print(f"Start Training Seed {seed} from Epoch {start_epoch}")

    for epoch in range(start_epoch, EPOCHS):
        model.train()
        perm = torch.randperm(len(tr_idx), device=DEVICE)
        tr_idx_perm = tr_idx_gpu[perm]

        loss_accum = 0
        steps = 0
        pbar = tqdm(range(0, len(tr_idx), BATCH_SIZE), leave=False, desc=f"Ep {epoch + 1}")

        for i in pbar:
            b_idx = tr_idx_perm[i: i + BATCH_SIZE]
            bd, bp = G_idx_d[b_idx], G_idx_p[b_idx]
            y = G_aff[b_idx]
            real_d = int(G_dM[bd].sum(1).max().item())
            real_p = int(G_pM[bp].sum(1).max().item())

            d_in = {
                'd_plm': G_d_plm[bd], 'd_mask': G_d_mask[bd],
                'dx': G_dX[bd][:, :real_d],
                'dadj_norm': G_dA_norm[bd][:, :real_d, :real_d],
                'dadj_bool': G_dA_bool[bd][:, :real_d, :real_d],
                'dnode_mask': G_dM[bd][:, :real_d]
            }
            p_in = {
                'p_plm': G_p_plm[bp], 'p_mask': G_p_mask[bp],
                'px': G_pX[bp][:, :real_p],
                'padj_norm': G_pA_norm[bp][:, :real_p, :real_p],
                'padj_bool': G_pA_bool[bp][:, :real_p, :real_p],
                'pnode_mask': G_pM[bp][:, :real_p]
            }

            optimizer.zero_grad(set_to_none=True)

            with autocast('cuda', dtype=AMP_TYPE):
                pred, xs, ys = model(**d_in, **p_in)
                loss = ALPHA_MSE * F.mse_loss(pred, y) + ALPHA_RANK * pairwise_ranking_loss(pred, y)
                if ALPHA_CL > 0:
                    loss += ALPHA_CL * (supcon_loss(xs, y) + supcon_loss(ys, y))

            if USE_SCALER:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

            try:
                scheduler.step()
            except ValueError:
                pass

            loss_accum += loss.item()
            steps += 1

        # Validation
        model.eval()
        val_preds, val_trues = [], []
        with torch.no_grad():
            for i in range(0, len(val_idx), BATCH_SIZE * 2):
                b_idx = val_idx_gpu[i: i + BATCH_SIZE * 2]
                bd, bp = G_idx_d[b_idx], G_idx_p[b_idx]
                real_d = int(G_dM[bd].sum(1).max().item())
                real_p = int(G_pM[bp].sum(1).max().item())

                with autocast('cuda', dtype=AMP_TYPE):
                    p, _, _ = model(
                        G_d_plm[bd], G_d_mask[bd], G_p_plm[bp], G_p_mask[bp],
                        G_dX[bd][:, :real_d], G_dA_norm[bd][:, :real_d, :real_d],
                        G_dA_bool[bd][:, :real_d, :real_d], G_dM[bd][:, :real_d],
                        G_pX[bp][:, :real_p], G_pA_norm[bp][:, :real_p, :real_p],
                        G_pA_bool[bp][:, :real_p, :real_p], G_pM[bp][:, :real_p]
                    )
                val_preds.append(p)
                val_trues.append(G_aff[b_idx])

        val_mse = F.mse_loss(torch.cat(val_preds), torch.cat(val_trues)).item()

        if val_mse < best_mse:
            best_mse = val_mse
            early_stop = 0
            best_state = {k: v.cpu() for k, v in model.state_dict().items()}
            torch.save(best_state, os.path.join(CACHE_DIR, f"best_model_seed_{seed}.pth"))
        else:
            early_stop += 1

        torch.save({
            'epoch': epoch,
            'model': model.state_dict(),
            'optim': optimizer.state_dict(),
            'sched': scheduler.state_dict(),
            'scaler': scaler.state_dict() if USE_SCALER else None,
            'best_mse': best_mse,
            'best_state': best_state,
            'early_stop': early_stop
        }, ckpt_path)

        print(f"Ep {epoch + 1} | Loss: {loss_accum / steps:.4f} | Val MSE: {val_mse:.4f} | Best: {best_mse:.4f}")

        if early_stop >= PATIENCE:
            print("Early Stopping.")
            break

    print("Testing Best Model...")
    model.load_state_dict(best_state)
    model.eval()
    test_preds, test_trues = [], []
    with torch.no_grad():
        for i in range(0, len(te_idx), BATCH_SIZE * 2):
            b_idx = te_idx_gpu[i: i + BATCH_SIZE * 2]
            bd, bp = G_idx_d[b_idx], G_idx_p[b_idx]
            real_d = int(G_dM[bd].sum(1).max().item())
            real_p = int(G_pM[bp].sum(1).max().item())
            with autocast('cuda', dtype=AMP_TYPE):
                p, _, _ = model(
                    G_d_plm[bd], G_d_mask[bd], G_p_plm[bp], G_p_mask[bp],
                    G_dX[bd][:, :real_d], G_dA_norm[bd][:, :real_d, :real_d],
                    G_dA_bool[bd][:, :real_d, :real_d], G_dM[bd][:, :real_d],
                    G_pX[bp][:, :real_p], G_pA_norm[bp][:, :real_p, :real_p],
                    G_pA_bool[bp][:, :real_p, :real_p], G_pM[bp][:, :real_p]
                )
            test_preds.append(p)
            test_trues.append(G_aff[b_idx])

    res = compute_metrics(torch.cat(test_trues), torch.cat(test_preds))
    print(f"Seed {seed} Result: MSE: {res['MSE']:.4f} | CI: {res['CI']:.4f} | R2: {res['R2']:.4f} | Pearson: {res['Pearson']:.4f} | Spearman: {res['Spearman']:.4f}")
    return res

def main():
    gc.collect()
    torch.cuda.empty_cache()
    data = load_data_and_plm()

    seeds = [42, 1, 123, 777, 666]
    results = []

    for i, s in enumerate(seeds):
        res = run_cycle(s, i, data)
        results.append(res)

    print(pd.DataFrame(results))

if __name__ == "__main__":
    main()