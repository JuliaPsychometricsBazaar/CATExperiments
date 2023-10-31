using ComputerAdaptiveTesting.Aggregators
using PsychometricsBazaarBase.Integrators
using FittedItemBanks.DummyData: dummy_full, std_normal, std_mv_normal
using ComputerAdaptiveTesting.Sim: run_random_comparison
using ComputerAdaptiveTesting.NextItemRules
using FittedItemBanks
using ComputerAdaptiveTesting
using ComputerAdaptiveTesting.Sim
using ComputerAdaptiveTesting.NextItemRules
using ComputerAdaptiveTesting.TerminationConditions
using ComputerAdaptiveTesting.Aggregators
using ComputerAdaptiveTesting.Responses: BareResponses
using FittedItemBanks
using PsychometricsBazaarBase.Integrators
using PsychometricsBazaarBase.Optimizers
import PsychometricsBazaarBase.IntegralCoeffs
using ItemResponseDatasets.VocabIQ
using RIrtWrappers.Mirt
using Random
using Distributions
using FittedItemBanks.DummyData: dummy_full, std_normal, SimpleItemBankSpec, StdModel4PL, VectorContinuousDomain, OneDimContinuousDomain, BooleanResponse
using CondaPkg
using DataFrames
using Parquet2: writefile

include("./utils/RandomItemBanks.jl")

using .RandomItemBanks


function main()
    rng = Xoshiro(42)
    (item_bank, abilities, responses) = dummy_full(
        rng,
        SimpleItemBankSpec(StdModel4PL(), OneDimContinuousDomain(), BooleanResponse());
        num_questions=100000,
        num_testees=1
    )
    df = DataFrame(
        difficulties=item_bank.inner_bank.inner_bank.difficulties,
        discriminations=item_bank.inner_bank.inner_bank.discriminations,
        guesses=item_bank.inner_bank.guesses,
        slips=item_bank.slips
    )
    writefile("df.parquet", df)
end


if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
