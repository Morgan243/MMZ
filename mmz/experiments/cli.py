if __name__ == """__main__""":
    #from mmz.experiments.causal_gpt2 import CausalGPT2Pretraining
    from mmz.experiments.hf_causal_gpt2 import HFGPTCausalTrain
    from simple_parsing import ArgumentParser

    parser = ArgumentParser()
    parser.add_arguments(HFGPTCausalTrain, dest="options")
    args = parser.parse_args()
    print(args)
    options: HFGPTCausalTrain = args.options
    options.run()


