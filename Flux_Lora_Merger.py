import os
import io
import gc
import torch
import logging
import contextlib
from safetensors.torch import load_file
from folder_paths import get_filename_list, get_full_path
from comfy.sd import load_lora_for_models
from comfy_extras.nodes_model_merging import save_checkpoint

class FluxLoraMerger:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "unet_model": ("MODEL",),
                "merge_strategy": (["additive", "average", "sequential"],),
                "enable_lora1": ("BOOLEAN", {"default": False}),
                "lora1": (get_filename_list("loras"),),
                "lora1_weight": ("FLOAT", {"default": 1.0}),
                "enable_lora2": ("BOOLEAN", {"default": False}),
                "lora2": (get_filename_list("loras"),),
                "lora2_weight": ("FLOAT", {"default": 1.0}),
                "enable_lora3": ("BOOLEAN", {"default": False}),
                "lora3": (get_filename_list("loras"),),
                "lora3_weight": ("FLOAT", {"default": 1.0}),
                "enable_lora4": ("BOOLEAN", {"default": False}),
                "lora4": (get_filename_list("loras"),),
                "lora4_weight": ("FLOAT", {"default": 1.0}),
                "save_model": ("BOOLEAN", {"default": False}),
                "save_filename": ("STRING", {"default": "flux_lora_merged.safetensors"}),
            }
        }

    RETURN_TYPES = ("MODEL", "STRING",)
    RETURN_NAMES = ("model", "merge_report",)
    FUNCTION = "merge_loras"
    CATEGORY = "flux/dev"

    def merge_with_comfy(self, model, lora_path, weight, ignored_counter, report_list):
        lora_sd = load_file(lora_path)

        comfy_logger = logging.getLogger()
        previous_level = comfy_logger.level
        comfy_logger.setLevel(logging.ERROR)

        try:
            with contextlib.redirect_stdout(io.StringIO()):
                model, _ = load_lora_for_models(model, None, lora_sd, weight, 0.0)
        finally:
            comfy_logger.setLevel(previous_level)

        unet_keys = [k for k in lora_sd if k.startswith("lora_unet")]
        ignored_keys = [k for k in lora_sd if not k.startswith("lora_unet")]

        report_list.append((os.path.basename(lora_path), len(unet_keys), len(ignored_keys)))
        ignored_counter[0] += len(ignored_keys)

        return model

    def merge_loras(self, unet_model, merge_strategy="additive",
                    enable_lora1=False, lora1="", lora1_weight=1.0,
                    enable_lora2=False, lora2="", lora2_weight=1.0,
                    enable_lora3=False, lora3="", lora3_weight=1.0,
                    enable_lora4=False, lora4="", lora4_weight=1.0,
                    save_model=False, save_filename="flux_lora_merged.safetensors"):

        patcher = unet_model
        base_model = patcher.model
        ignored_lora_keys = [0]
        lora_report = []

        lora_list = [
            (enable_lora1, lora1, lora1_weight),
            (enable_lora2, lora2, lora2_weight),
            (enable_lora3, lora3, lora3_weight),
            (enable_lora4, lora4, lora4_weight),
        ]
        active_loras = [(get_full_path("loras", l), w) for e, l, w in lora_list if e and l]
        print(f"[MERGE] Starting {merge_strategy} merge with {len(active_loras)} LoRA(s)")

        if merge_strategy == "sequential":
            for lora_path, weight in active_loras:
                patcher = self.merge_with_comfy(patcher, lora_path, weight, ignored_lora_keys, lora_report)

        elif merge_strategy in ["additive", "average"]:
            threshold = 1e-6
            merged_delta = {}
            base_sd = base_model.state_dict()

            for lora_path, weight in active_loras:
                patcher = self.merge_with_comfy(patcher, lora_path, weight, ignored_lora_keys, lora_report)
                current_sd = patcher.model.state_dict()

                for k in current_sd.keys():
                    if k not in base_sd:
                        continue

                    base_val_fp32 = base_sd[k].to(torch.float32)
                    current_val_fp32 = current_sd[k].to(torch.float32)
                    diff = current_val_fp32 - base_val_fp32

                    if diff.abs().max().item() > threshold:
                        if k not in merged_delta:
                            merged_delta[k] = diff * weight
                        else:
                            merged_delta[k] += diff * weight

            if merge_strategy == "average" and active_loras:
                for k in merged_delta:
                    merged_delta[k] = merged_delta[k] / len(active_loras)

            with torch.no_grad():
                for name, param in base_model.named_parameters():
                    if name in merged_delta:
                        new_value = base_sd[name].to(torch.float32) + merged_delta[name]
                        param.copy_(new_value.to(param.dtype))

        # Rapport fusion (console + UI)
        debug_text = ""
        if lora_report:
            debug_text += "LoRA Merge Report:\n"
            debug_text += "| Filename              | UNet Keys | Ignored Keys |\n"
            debug_text += "|-----------------------|-----------|---------------|\n"
            for name, loaded, ignored in lora_report:
                debug_text += f"| {name:<21} | {loaded:^9} | {ignored:^13} |\n"
                print(f" → {name}: {loaded} UNet keys, {ignored} ignored")

        if ignored_lora_keys[0] > 0:
            msg = f"⚠️ {ignored_lora_keys[0]} LoRA keys ignored (non-UNet, e.g. text encoder)"
            print(msg)
            debug_text += f"\n{msg}"

        # 🔒 Sauvegarde sécurisée avec libération mémoire
        if save_model:
            print("[SAVE] Preparing to save model...")
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            gc.collect()

            output_path = os.path.join(os.getcwd(), "output")
            os.makedirs(output_path, exist_ok=True)

            try:
                save_checkpoint(
                    model=patcher,
                    filename_prefix=os.path.splitext(save_filename)[0],
                    output_dir=output_path,
                    prompt=None,
                    extra_pnginfo=None
                )
                print(f"[SAVE] Model saved to {save_filename}")
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    print("❌ [SAVE ERROR] Torch ran out of memory during model save. Try freeing VRAM and retry.")
                else:
                    raise e

        return (patcher, debug_text)


NODE_CLASS_MAPPINGS = {
    "FluxLoraMerger": FluxLoraMerger
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "FluxLoraMerger": "Flux LoRA Merger"
}
