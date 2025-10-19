import torch
import torch.optim as optim
import torch.nn.functional as F
import os
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from data_loader import load_twitch_domain
from gnn_extractor import GNNExtractor
from classifier import NodeClassifier
from discriminator import DomainDiscriminator
from losses import sampled_gromov_wasserstein_loss

# -- Hyperparameters --
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
LEARNING_RATE = 0.001
EPOCHS = 200
HIDDEN_DIM = 64
EMBEDDING_DIM = 32
NUM_CLASSES = 2
ALPHA = 0.5  # Adversarial loss weight
BETA = 0.1   # OT loss weight
OT_SAMPLE_SIZE = 500  # Sample size for OT computation
OT_UPDATE_FREQ = 5  # Update OT loss every N epochs

# Source and target domains
SOURCE_DOMAIN = 'ENGB'  # English domain
TARGET_DOMAIN = 'FR'    # French domain

print(f"Using device: {DEVICE}")
print(f"Source domain: {SOURCE_DOMAIN}, Target domain: {TARGET_DOMAIN}")

# -- 1. Load Data --
print("\n=== Loading Data ===")
data_path = '/Users/mohammad/Desktop/Domain-Adaptation/data'

print(f"Loading source domain ({SOURCE_DOMAIN})...")
source_data, source_adj = load_twitch_domain(data_path, SOURCE_DOMAIN, use_labels=True)
source_data = source_data.to(DEVICE)

print(f"Loading target domain ({TARGET_DOMAIN})...")
target_data, target_adj = load_twitch_domain(data_path, TARGET_DOMAIN, use_labels=True)
target_data = target_data.to(DEVICE)

print(f"Source: {source_data.num_nodes} nodes, {source_data.edge_index.shape[1]} edges, "
      f"{source_data.x.shape[1]} features")
print(f"Target: {target_data.num_nodes} nodes, {target_data.edge_index.shape[1]} edges, "
      f"{target_data.x.shape[1]} features")
print(f"Number of classes: {NUM_CLASSES}")

# Check feature dimensions match
if source_data.x.shape[1] != target_data.x.shape[1]:
    print(f"Warning: Feature dimensions don't match! Source: {source_data.x.shape[1]}, "
          f"Target: {target_data.x.shape[1]}")
    # Pad the smaller one with zeros
    max_features = max(source_data.x.shape[1], target_data.x.shape[1])
    if source_data.x.shape[1] < max_features:
        padding = torch.zeros(source_data.num_nodes, max_features - source_data.x.shape[1]).to(DEVICE)
        source_data.x = torch.cat([source_data.x, padding], dim=1)
    if target_data.x.shape[1] < max_features:
        padding = torch.zeros(target_data.num_nodes, max_features - target_data.x.shape[1]).to(DEVICE)
        target_data.x = torch.cat([target_data.x, padding], dim=1)

INPUT_DIM = source_data.x.shape[1]

# # -- 2. Initialize Models --
print("\n=== Initializing Models ===")
feature_extractor = GNNExtractor(INPUT_DIM, HIDDEN_DIM, EMBEDDING_DIM).to(DEVICE)
classifier = NodeClassifier(EMBEDDING_DIM, NUM_CLASSES).to(DEVICE)
discriminator = DomainDiscriminator(EMBEDDING_DIM, HIDDEN_DIM).to(DEVICE)

print(f"GNN Extractor: {INPUT_DIM} -> {HIDDEN_DIM} -> {EMBEDDING_DIM}")
print(f"Classifier: {EMBEDDING_DIM} -> {NUM_CLASSES}")
print(f"Discriminator: {EMBEDDING_DIM} -> {HIDDEN_DIM} -> 1")

# -- 3. Setup Optimizer --
optimizer = optim.Adam(
    list(feature_extractor.parameters()) + 
    list(classifier.parameters()) + 
    list(discriminator.parameters()),
    lr=LEARNING_RATE
)

# # -- 4. Training Loop --
print("\n=== Starting Training ===")
print(f"Epochs: {EPOCHS}, Learning Rate: {LEARNING_RATE}")
print(f"Loss weights - Alpha (adversarial): {ALPHA}, Beta (OT): {BETA}")
print(f"OT sampling: {OT_SAMPLE_SIZE} nodes, computed every {OT_UPDATE_FREQ} epochs\n")

best_target_acc = 0.0
loss_ot_cached = torch.tensor(0.0, device=DEVICE)

for epoch in range(EPOCHS):
    feature_extractor.train()
    classifier.train()
    discriminator.train()
    
    optimizer.zero_grad()
    
    # --- Forward Passes ---
    # Source domain
    source_features = feature_extractor(source_data.x, source_data.edge_index)
    source_preds = classifier(source_features)
    source_domain_preds = discriminator(source_features, alpha=ALPHA)

    # Target domain
    target_features = feature_extractor(target_data.x, target_data.edge_index)
    target_domain_preds = discriminator(target_features, alpha=ALPHA)

    # --- Loss Calculation ---
    # 1. Classification Loss (on source domain only)
    loss_cls = F.cross_entropy(source_preds, source_data.y)

    # 2. Adversarial Loss (Domain Discriminator)
    # Discriminator should predict 1 for source, 0 for target
    # But GRL makes feature extractor try to confuse it
    loss_adv = F.binary_cross_entropy(
        torch.cat([source_domain_preds, target_domain_preds]),
        torch.cat([
            torch.ones_like(source_domain_preds), 
            torch.zeros_like(target_domain_preds)
        ])
    )
    
    # 3. Optimal Transport Loss (Structure-level alignment)
    # Compute periodically to save time
    if epoch % OT_UPDATE_FREQ == 0:
        with torch.no_grad():
            try:
                loss_ot_cached = sampled_gromov_wasserstein_loss(
                    source_adj, target_adj, sample_size=OT_SAMPLE_SIZE
                ).to(DEVICE)
            except Exception as e:
                print(f"Warning: OT computation failed at epoch {epoch}: {e}")
                loss_ot_cached = torch.tensor(0.0, device=DEVICE)
    
    loss_ot = loss_ot_cached

    # --- Total Loss ---
    total_loss = loss_cls + ALPHA * loss_adv + BETA * loss_ot
    
    # --- Backward Pass & Optimization ---
    total_loss.backward()
    optimizer.step()

    # --- Logging and Evaluation ---
    if epoch % 10 == 0:
        feature_extractor.eval()
        classifier.eval()
        
        with torch.no_grad():
            # Evaluate on source domain (sanity check)
            source_eval_features = feature_extractor(source_data.x, source_data.edge_index)
            source_eval_preds = classifier(source_eval_features)
            source_pred_labels = source_eval_preds.argmax(dim=1).cpu().numpy()
            source_true_labels = source_data.y.cpu().numpy()
            source_acc = accuracy_score(source_true_labels, source_pred_labels)
            
            # Evaluate on target domain
            target_eval_features = feature_extractor(target_data.x, target_data.edge_index)
            target_eval_preds = classifier(target_eval_features)
            target_pred_labels = target_eval_preds.argmax(dim=1).cpu().numpy()
            target_true_labels = target_data.y.cpu().numpy()
            target_acc = accuracy_score(target_true_labels, target_pred_labels)
            target_f1 = f1_score(target_true_labels, target_pred_labels, average='binary')
            
            # Track best target accuracy
            if target_acc > best_target_acc:
                best_target_acc = target_acc
        
        print(f"Epoch {epoch:03d} | "
              f"Total: {total_loss:.4f} | "
              f"Cls: {loss_cls:.4f} | "
              f"Adv: {loss_adv:.4f} | "
              f"OT: {loss_ot:.4f} | "
              f"Src Acc: {source_acc:.3f} | "
              f"Tgt Acc: {target_acc:.3f} | "
              f"Tgt F1: {target_f1:.3f}")

print("\n=== Training Finished! ===")
print(f"Best Target Accuracy: {best_target_acc:.4f}")

# -- 5. Final Evaluation on Target Domain --
print("\n=== Final Evaluation on Target Domain ===")
feature_extractor.eval()
classifier.eval()

with torch.no_grad():
    target_features = feature_extractor(target_data.x, target_data.edge_index)
    target_preds = classifier(target_features)
    target_pred_labels = target_preds.argmax(dim=1).cpu().numpy()
    target_probs = F.softmax(target_preds, dim=1)[:, 1].cpu().numpy()
    target_true_labels = target_data.y.cpu().numpy()
    
    accuracy = accuracy_score(target_true_labels, target_pred_labels)
    f1 = f1_score(target_true_labels, target_pred_labels, average='binary')
    
    try:
        auc = roc_auc_score(target_true_labels, target_probs)
        print(f"Accuracy: {accuracy:.4f}")
        print(f"F1-Score: {f1:.4f}")
        print(f"AUC-ROC: {auc:.4f}")
    except:
        print(f"Accuracy: {accuracy:.4f}")
        print(f"F1-Score: {f1:.4f}")
        print("AUC-ROC: N/A (only one class present)")

# # Save the trained models
# print("\n=== Saving Models ===")
# models_dir = '/Users/mohammad/Desktop/Domain-Adaptation/models'
# os.makedirs(models_dir, exist_ok=True)

# torch.save(feature_extractor.state_dict(), 
#           f'{models_dir}/feature_extractor_{SOURCE_DOMAIN}_to_{TARGET_DOMAIN}.pt')
# torch.save(classifier.state_dict(), 
#           f'{models_dir}/classifier_{SOURCE_DOMAIN}_to_{TARGET_DOMAIN}.pt')

# print(f"Models saved to {models_dir}/")
print("\nDone!")