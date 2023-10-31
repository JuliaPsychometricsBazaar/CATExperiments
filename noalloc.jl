using ComputerAdaptiveTesting.Aggregators
using PsychometricsBazaarBase.Integrators
using FittedItemBanks.DummyData: dummy_full, std_normal, std_mv_normal
using ComputerAdaptiveTesting.Sim: run_random_comparison
using ComputerAdaptiveTesting.NextItemRules
using FittedItemBanks
using Base.Filesystem
using ComputerAdaptiveTesting
using ComputerAdaptiveTesting.DecisionTree: DecisionTreeGenerationConfig, generate_dt_cat
using ComputerAdaptiveTesting.Sim
using ComputerAdaptiveTesting.NextItemRules
using ComputerAdaptiveTesting.TerminationConditions
using ComputerAdaptiveTesting.Aggregators
using FittedItemBanks
using PsychometricsBazaarBase.Integrators
using PsychometricsBazaarBase.Optimizers
import PsychometricsBazaarBase.IntegralCoeffs
using ItemResponseDatasets.VocabIQ
using RIrtWrappers.Mirt
using Random
using Distributions


include("./utils/RandomItemBanks.jl")

using .RandomItemBanks: clumpy_4pl_item_bank
using InteractiveUtils: @code_warntype
using Profile.Allocs: @profile
using PProf


function main()
    rng = Xoshiro(42)
    params = clumpy_4pl_item_bank(rng, 3, 10)
    integrator = even_grid(-6.0, 6.0, 13)
    ability_estimator = MeanAbilityEstimator(PriorAbilityEstimator(std_normal), integrator)
    next_item_rule = catr_next_item_aliases["MEPV"](ability_estimator)
    config = DecisionTreeGenerationConfig(;
        max_depth=UInt(2),
        next_item=next_item_rule,
        ability_estimator=ability_estimator,
    )
    config = preallocate(config)
    @time generate_dt_cat(config, params)
    @time generate_dt_cat(config, params)
    @profile sample_rate=0.1 generate_dt_cat(config, params)
    PProf.Allocs.pprof()
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end