import os
import sys
import argparse
import torch


def load_checkpoint(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    obj = torch.load(path, map_location='cpu')
    return obj


def ensure_dict_checkpoint(obj):
    """
    Ensure we return a dict checkpoint we can add keys to.
    If obj is a raw state_dict (OrderedDict), wrap it as {'model': obj}.
    """
    if isinstance(obj, dict):
        return obj
    else:
        return {'model': obj}


def copy_ema_to_target(stage1_ckpt_path, target_ckpt_path, out_path, copy_model=False, only_model_ema=False, allow_synthesize_ema=False):
    """
    Copy EMA from Stage-1 checkpoint into target checkpoint.

    - Expects stage1_ckpt to contain key 'ema' which itself is a dict with keys:
      'module' (state_dict) and optional 'updates'.
    - Optionally also copy 'model' from stage1 to target for full consistency.
    """
    stage1 = load_checkpoint(stage1_ckpt_path)

    # Locate EMA structure
    ema_state = None
    if isinstance(stage1, dict):
        ema_state = stage1.get('ema', None)
    # If EMA missing and allowed to synthesize, build it from stage1['model']
    if (not isinstance(ema_state, dict) or 'module' not in ema_state) and allow_synthesize_ema:
        if isinstance(stage1, dict) and 'model' in stage1:
            ema_state = {'module': stage1['model'], 'updates': 0}
            print("[WARN] Stage-1 lacks 'ema.module'. Synthesizing EMA from 'model' (updates=0).")
        else:
            raise KeyError(
                f"Cannot synthesize EMA: Stage-1 checkpoint at {stage1_ckpt_path} has no 'ema.module' nor 'model'."
            )
    if not isinstance(ema_state, dict) or 'module' not in ema_state:
        raise KeyError(
            f"Stage-1 checkpoint at {stage1_ckpt_path} does not contain a valid EMA state.\n"
            f"Expected a dict under key 'ema' with a 'module' subkey."
        )

    # Load target and ensure dict format
    target = load_checkpoint(target_ckpt_path)
    target = ensure_dict_checkpoint(target)

    if only_model_ema:
        # Create a minimal checkpoint containing only 'model' and 'ema'
        final_obj = {}
        if copy_model and isinstance(stage1, dict) and 'model' in stage1:
            final_obj['model'] = stage1['model']
        else:
            # Prefer target model weights
            if 'model' not in target:
                raise KeyError(
                    f"Target checkpoint at {target_ckpt_path} does not contain 'model' state."
                )
            final_obj['model'] = target['model']
        final_obj['ema'] = ema_state

        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        torch.save(final_obj, out_path)
        print(f"[OK] Wrote minimal (model + ema) checkpoint to {out_path} (source EMA: {stage1_ckpt_path}, target MODEL: {target_ckpt_path})")
    else:
        # Inject EMA into the existing target structure
        target['ema'] = ema_state

        # Optionally sync model too (helps ensure the same weights layout)
        if copy_model and isinstance(stage1, dict) and 'model' in stage1:
            target['model'] = stage1['model']

        # Save
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        torch.save(target, out_path)
        print(f"[OK] Wrote EMA into {out_path} (source: {stage1_ckpt_path}, target: {target_ckpt_path})")


def main():
    parser = argparse.ArgumentParser(description="Copy EMA module from a Stage-1 checkpoint into one or more target ckpts.")
    parser.add_argument('--stage1', type=str, required=True,
                        help='Path to Stage-1 best_stg1.pth containing ema')
    parser.add_argument('--targets', type=str, nargs='+', required=True,
                        help='List of target ckpt paths to receive EMA')
    parser.add_argument('--suffix', type=str, default='_with_ema.pth',
                        help='Suffix for output files (default: _with_ema.pth)')
    parser.add_argument('--copy-model', action='store_true',
                        help='Also copy stage1["model"] into target["model"]')
    parser.add_argument('--only-model-ema', action='store_true',
                        help='Output checkpoint containing ONLY "model" and "ema" keys (drops optimizer/scheduler/etc.)')
    parser.add_argument('--allow-synthesize-ema', action='store_true',
                        help='If stage1 has no ema.module but has model, synthesize ema from model (ema.module = model).')

    args = parser.parse_args()

    stage1_path = args.stage1
    targets = args.targets
    suffix = args.suffix
    copy_model = args.copy_model
    only_model_ema = args.only_model_ema
    allow_synthesize_ema = args.allow_synthesize_ema

    # Validate stage1
    print(f"Inspecting Stage-1 checkpoint: {stage1_path}")
    stage1 = load_checkpoint(stage1_path)
    if not (isinstance(stage1, dict) and isinstance(stage1.get('ema'), dict) and 'module' in stage1['ema']):
        raise RuntimeError("Stage-1 checkpoint missing 'ema.module'; cannot proceed.")
    print("Stage-1 EMA detected.")

    # Process targets
    for target_path in targets:
        if not os.path.exists(target_path):
            print(f"[WARN] Target not found, skipping: {target_path}")
            continue
        base, ext = os.path.splitext(target_path)
        out_path = base + suffix if ext == '' else base + suffix
        # If ext is .pth, we will produce base + _with_ema.pth
        if ext:
            out_path = base + suffix  # e.g., foo.pth -> foo_with_ema.pth
        try:
            copy_ema_to_target(stage1_path, target_path, out_path, copy_model=copy_model, only_model_ema=only_model_ema, allow_synthesize_ema=allow_synthesize_ema)
        except Exception as e:
            print(f"[ERROR] Failed to process {target_path}: {e}")


if __name__ == '__main__':
    main()