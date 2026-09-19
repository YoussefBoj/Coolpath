"""Compatibility entry point for the original HRNet download script."""
import sys
from download_checkpoints import main

if __name__ == '__main__':
    raise SystemExit(main(['--models', 'hrnet', *sys.argv[1:]]))
