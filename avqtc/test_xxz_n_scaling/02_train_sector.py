"""Exact full-space direct training for n=12 using all sectors on GPU."""
from scaling import arguments, settings
from sector_training import train


if __name__ == '__main__':
    parser = arguments(__doc__)
    parser.set_defaults(sizes=[12], backend='gpu')
    args = parser.parse_args()
    cfg, sizes = settings(args.config, args.sizes)
    if any(n != 12 for n in sizes):
        parser.error('This entry point is reserved for n=12.')
    if args.backend != 'gpu':
        parser.error('This entry point requires --backend gpu; no automatic CPU fallback.')
    train(cfg, sizes, args.output/'sector_gpu', backend='gpu')
