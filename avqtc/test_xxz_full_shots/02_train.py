from experiment import arguments, train

if __name__ == '__main__':
    args = arguments('Train XXZ full-space direct HST using finite shots; resume per iteration.')
    train(args.config, args.output)
