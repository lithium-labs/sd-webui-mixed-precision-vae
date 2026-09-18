import torch
import modules.scripts as scripts
import gradio as gr
from scripts import vae_blocks
import types

class Script(scripts.Script):
    def __init__(self):
        super().__init__()

    def title(self):
        return "Use mixed fp16/fp32 precision in VAE"

    def show(self, is_img2img):
        return scripts.AlwaysVisible

    def cast_params(self, params, precision):
        for p in params:
            p = p.to(dtype=precision)

    def before_process(self, p, *args, **kwargs):
        # Check if the model has a valid first_stage_model attribute (Forge compatibility guard)
        first_stage = getattr(p.sd_model, 'first_stage_model', None)
        if first_stage is None:
            return

        if hasattr(getattr(first_stage, 'decoder', None), 'mixed_precision'):
            # Already replaced
            return

        precision = first_stage.decoder.conv_in.weight.dtype
        if precision == torch.float32:
            print('Skipping mixed precision VAE extension due to VAE being in fp32 precision')
            return

        # Encoder
        first_stage.encoder.mixed_weights = [
            first_stage.encoder.norm_out,
            first_stage.encoder.conv_out,
            first_stage.encoder.mid.attn_1
        ]
        first_stage.encoder.mixed_precision = False
        first_stage.encoder.precision = precision
        first_stage.encoder.orig_forward = first_stage.encoder.forward
        first_stage.encoder.cast_weights = types.MethodType(vae_blocks.cast_weights, first_stage.encoder)
        first_stage.encoder.forward = types.MethodType(vae_blocks.encoder_forward, first_stage.encoder)

        # Decoder
        first_stage.decoder.mixed_weights = [
            first_stage.decoder.conv_out,
            first_stage.decoder.norm_out
        ]
        first_stage.decoder.mixed_precision = False
        first_stage.decoder.precision = precision
        first_stage.decoder.orig_forward = first_stage.decoder.forward
        first_stage.decoder.cast_weights = types.MethodType(vae_blocks.cast_weights, first_stage.decoder)
        first_stage.decoder.forward = types.MethodType(vae_blocks.decoder_forward, first_stage.decoder)

        for m in first_stage.modules():
            if m.__class__.__name__ == "ResnetBlock":
                mixed_weights = [m.norm1, m.conv2]
                if hasattr(m, 'nin_shortcut'):
                    mixed_weights.extend([m.nin_shortcut])
                if hasattr(m, 'conv_shortcut'):
                    mixed_weights.extend([m.conv_shortcut])
                m.precision = precision
                m.orig_forward = m.forward
                m.mixed_weights = mixed_weights
                m.mixed_precision = False
                m.cast_weights = types.MethodType(vae_blocks.cast_weights, m)
                m.forward = types.MethodType(vae_blocks.replaced_forward, m)
            elif m.__class__.__name__ in ["Upsample", "Downsample"]:
                m.precision = precision
                m.mixed_weights = [m]
                m.orig_forward = m.forward
                m.mixed_precision = False
                m.cast_weights = types.MethodType(vae_blocks.cast_weights, m)
                m.forward = types.MethodType(vae_blocks.wrapped_mixed_forward, m)
                
        print('Mixed precision VAE extension applied')
