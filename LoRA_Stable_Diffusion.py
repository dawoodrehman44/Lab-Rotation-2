import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from transformers import CLIPTokenizer
from diffusers import StableDiffusionPipeline
from peft import get_peft_model, LoraConfig
from accelerate import Accelerator
from tqdm.auto import tqdm

# ==== CONFIGURATION ====
DATA_CSV = "path/to/dataset.csv"
IMAGE_ROOT = "path/to/image/root"
OUTPUT_DIR = "path/to/output/directory"
START_EPOCH = 0
EPOCHS = 10
BATCH_SIZE = 8
LEARNING_RATE = 1e-4
MAX_LENGTH = 77
IMAGE_SIZE = 512
SAVE_EVERY = 1

# ==== DATASET ====
class CaptionDataset(Dataset):
    def __init__(self, dataframe, image_root, tokenizer):
        self.dataframe = dataframe
        self.image_root = image_root
        self.tokenizer = tokenizer
        self.transform = transforms.Compose([
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])
        ])
        self.valid_data = []
        for _, row in dataframe.iterrows():
            img_path = os.path.join(image_root, row["Path"])
            if os.path.exists(img_path):
                self.valid_data.append((img_path, self.generate_caption(row)))

    def generate_caption(self, row):
        findings = ["Finding1", "Finding2", "Finding3"]
        findings_present = [f for f in findings if row.get(f, 0) == 1]
        findings_str = ", ".join(findings_present).lower() if findings_present else "no significant findings"
        return f"Image caption: {findings_str}."

    def __len__(self):
        return len(self.valid_data)

    def __getitem__(self, idx):
        img_path, caption = self.valid_data[idx]
        image = self.transform(Image.open(img_path).convert("RGB"))
        tokenized = self.tokenizer(caption, padding="max_length", truncation=True, max_length=MAX_LENGTH, return_tensors="pt")
        return {
            "pixel_values": image,
            "input_ids": tokenized.input_ids.squeeze(0),
            "attention_mask": tokenized.attention_mask.squeeze(0)
        }

# ==== TRAINING LOOP ====
def train():
    accelerator = Accelerator()
    device = accelerator.device

    df = pd.read_csv(DATA_CSV)
    tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")
    dataset = CaptionDataset(df, IMAGE_ROOT, tokenizer)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    pipe = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5", torch_dtype=torch.float32)
    pipe.to(device)
    unet = pipe.unet

    lora_config = LoraConfig(r=4, lora_alpha=16, target_modules=["to_q", "to_k", "to_v"], lora_dropout=0.1, bias="none")
    unet = get_peft_model(unet, lora_config)

    optimizer = torch.optim.AdamW(unet.parameters(), lr=LEARNING_RATE)
    unet, optimizer, dataloader = accelerator.prepare(unet, optimizer, dataloader)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    for epoch in range(START_EPOCH, EPOCHS):
        unet.train()
        total_loss = 0

        for batch in tqdm(dataloader, desc=f"Epoch {epoch + 1}"):
            latents = pipe.vae.encode(batch["pixel_values"].to(device)).latent_dist.sample() * 0.18215
            noise = torch.randn_like(latents)
            timesteps = torch.randint(0, 1000, (latents.shape[0],), device=device).long()
            noisy_latents = pipe.scheduler.add_noise(latents, noise, timesteps)
            encoder_hidden_states = pipe.text_encoder(batch["input_ids"].to(device)).last_hidden_state
            model_pred = unet(noisy_latents, timesteps, encoder_hidden_states=encoder_hidden_states).sample
            loss = torch.nn.functional.mse_loss(model_pred, noise)

            accelerator.backward(loss)
            optimizer.step()
            optimizer.zero_grad()
            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch + 1} Average Loss: {avg_loss:.4f}")

        if (epoch + 1) % SAVE_EVERY == 0:
            save_path = os.path.join(OUTPUT_DIR, f"lora_unet_epoch_{epoch+1}.pt")
            accelerator.save(unet.state_dict(), save_path)
            print(f"Saved LoRA weights to {save_path}")

if __name__ == "__main__":
    train()
