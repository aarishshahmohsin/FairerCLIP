import os
import json
import pandas as pd
import torch
from PIL import Image
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from clip import clip 

# -------------------------------
# Config
# -------------------------------
csv_path = "./dataset_total.csv"   # dataset: image_path, gender, state
device = "cuda" if torch.cuda.is_available() else "cpu"
clip_model_name = "ViTL14"
output_dir = "./features"
os.makedirs(output_dir, exist_ok=True)

# -------------------------------
# Load CLIP
# -------------------------------
model, preprocess = clip.load(clip_model_name, device=device)

# -------------------------------
# Load dataset
# -------------------------------
df = pd.read_csv(csv_path)

# -------------------------------
# Train/test split (80/20 here)
# -------------------------------
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df["gender"])  
# 👆 stratify on gender (can also use "state" or None depending on balance)

# -------------------------------
# Function to process one split
# -------------------------------
def process_split(df_split, split_name, group2idx):
    # ----- build label_s -----
    df_split["s_group"] = df_split["gender"].str.lower() + "_" + df_split["state"].str.lower()
    labels_s = df_split["s_group"].map(group2idx).values

    # ----- image features -----
    image_features = []
    for path in tqdm(df_split["image_path"], desc=f"Extracting {split_name} image features"):
        image = preprocess(Image.open(path).convert("RGB")).unsqueeze(0).to(device)
        with torch.no_grad():
            feat = model.encode_image(image)
            feat = feat / feat.norm(dim=-1, keepdim=True)
        image_features.append(feat.cpu())
    image_features = torch.cat(image_features, dim=0)

    # ----- labels_y = CLIP classification -----
    class_prompts = ["a photo of an Indian person", "a photo of a non-Indian person"]
    text_tokens = clip.tokenize(class_prompts).to(device)

    with torch.no_grad():
        text_features = model.encode_text(text_tokens)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

    with torch.no_grad():
        sims = (image_features.to(device) @ text_features.T.to(device)).cpu().numpy()
        preds = sims.argmax(axis=1)  # 0 = Indian, 1 = Non-Indian 

    labels_y = preds

    # ----- one-hot ±1 -----
    labels_y_oh = torch.nn.functional.one_hot(torch.from_numpy(labels_y), num_classes=2)
    labels_y_oh[labels_y_oh == 0] = -1

    labels_s_oh = torch.nn.functional.one_hot(torch.from_numpy(labels_s), num_classes=len(group2idx))
    labels_s_oh[labels_s_oh == 0] = -1

    # ----- save -----
    torch.save(image_features, os.path.join(output_dir, f"d=faces-s={split_name}-m={clip_model_name}.pt"))
    torch.save(labels_y_oh, os.path.join(output_dir, f"prediction-faces-y-{split_name}.pt"))
    torch.save(labels_s_oh, os.path.join(output_dir, f"prediction-faces-s-{split_name}.pt"))

    torch.save(text_features.cpu(), os.path.join(output_dir, f"text_features_{split_name}_faces_indian_nonindian_{clip_model_name}.pt"))

    print(f"✅ {split_name} features saved")
    print(f"{split_name} labels_s groups: {len(group2idx)}")
    print(f"{split_name} labels_y distribution: {labels_y.sum()} non-Indian / {(labels_y==0).sum()} Indian")


# -------------------------------
# Prepare group2idx (from full dataset, not per split)
# -------------------------------
df["s_group"] = df["gender"].str.lower() + "_" + df["state"].str.lower()
unique_groups = sorted(df["s_group"].unique())
group2idx = {g: i for i, g in enumerate(unique_groups)}

# -------------------------------
# Run for train and test
# -------------------------------
process_split(train_df, "train", group2idx)
process_split(test_df, "test", group2idx)

# save mapping once
with open(os.path.join(output_dir, "group2idx.json"), "w") as f:
    json.dump(group2idx, f, indent=2)

print("✅ All done, features ready in", output_dir)
