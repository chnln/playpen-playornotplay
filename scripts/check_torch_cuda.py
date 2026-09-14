#!/usr/bin/env python3

import argparse
import json
import sys

import torch


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--require-gpu", action="store_true")
    args = parser.parse_args()

    summary = {
        "torch_version": torch.__version__,
        "torch_cuda_version": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "device_count": torch.cuda.device_count(),
    }
    if torch.cuda.device_count() > 0:
        summary["device_names"] = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]

    print(json.dumps(summary, indent=2, sort_keys=True))

    if args.require_gpu and not torch.cuda.is_available():
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
