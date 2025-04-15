# Flux LoRA Merger — Custom Node for ComfyUI

A custom ComfyUI node to **merge up to 4 LoRA models into a UNet**

---

## ✨ Features

- ✅ Merge up to **4 LoRA models**
- ✅ Compatible with UNet models in **FP8 / FP16 / FP32**
- ✅ Three fusion strategies:
  - `additive`: weighted sum of LoRA deltas
  - `average`: average of LoRA deltas
  - `sequential`: apply one after another
- ✅ Uses `load_lora_for_models` (official ComfyUI function)
- ✅ Logs number of keys applied (UNet) and ignored (e.g., text encoder)
- ✅ Displays a clean merge report in both console and UI
- ✅ Option to **save the final merged model** in `.safetensors`
- ✅ Clears GPU VRAM before saving (avoids OOM)
- ✅ Catches memory errors during save and prints helpful warnings

---

## 📥 Inputs

| Parameter        | Type     | Description |
|------------------|----------|-------------|
| `unet_model`     | `MODEL`  | The base UNet model to apply LoRAs to |
| `merge_strategy` | `CHOICE` | `"additive"`, `"average"`, or `"sequential"` |
| `enable_loraX`   | `BOOLEAN`| Whether to enable the corresponding LoRA |
| `loraX`          | `STRING` | Filename of the LoRA (from your `loras/` folder) |
| `loraX_weight`   | `FLOAT`  | Weight multiplier for the LoRA |
| `save_model`     | `BOOLEAN`| Save the merged model to `output/` |
| `save_filename`  | `STRING` | Custom name for the saved `.safetensors` file |

---

## 📤 Outputs

| Output           | Type     | Description |
|------------------|----------|-------------|
| `model`          | `MODEL`  | The merged UNet model |
| `merge_report`   | `STRING` | A table summarizing the applied and ignored keys |

---

## 📊 Merge Report Example

```text
LoRA Merge Report:
| Filename              | UNet Keys | Ignored Keys |
|-----------------------|-----------|---------------|
| style1.safetensors    |   912     |     1014      |
| style2.safetensors    |   912     |     1014      |
| style3.safetensors    |   912     |     1014      |
| style4.safetensors    |   912     |     1014      |

⚠️ 4056 LoRA keys ignored (non-UNet, e.g. text encoder)
# ComfyUI_Flux_Lora_Merger
