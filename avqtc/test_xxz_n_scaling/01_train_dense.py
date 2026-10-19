from scaling import arguments, settings, train

if __name__ == '__main__':
    parser = arguments('Train XXZ full-space direct n=6,8,10 using dense matrices.')
    parser.set_defaults(sizes=[6, 8, 10])
    args = parser.parse_args()
    if any(n not in (6, 8, 10) for n in args.sizes):
        parser.error('Use 02_train_sector.py for n=12.')
    cfg, sizes = settings(args.config, args.sizes)
    train(cfg, sizes, args.output, args.backend)
