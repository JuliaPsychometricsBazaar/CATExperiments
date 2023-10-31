using ComputerAdaptiveTesting.Aggregators
using ComputerAdaptiveTesting.Aggregators: FunctionProduct
using PsychometricsBazaarBase.Integrators
using PsychometricsBazaarBase.Integrators: IterativeFixedGridIntegrator
using PsychometricsBazaarBase.IntegralCoeffs: SqDev
using FittedItemBanks.DummyData: dummy_full, std_normal, SimpleItemBankSpec, StdModel4PL, VectorContinuousDomain, OneDimContinuousDomain, BooleanResponse
using ComputerAdaptiveTesting.NextItemRules
using ComputerAdaptiveTesting.Responses: BareResponses
using ComputerAdaptiveTesting.Responses: AbilityLikelihood
using ComputerAdaptiveTesting: LogItemBank
using FittedItemBanks
using Random
using PProf
using BenchmarkTools
using Profile


function main()
    integrator = preallocate(FunctionIntegrator(even_grid(-6.0, 6.0, 61)))
    ability_estimator_prior = MeanAbilityEstimator(PriorAbilityEstimator(std_normal), integrator)
    
    state_criterion_prior = AbilityVarianceStateCriterion(ability_estimator_prior)
    item_criterion_prior = ExpectationBasedItemCriterion(ability_estimator_prior, state_criterion_prior)
    mepv_prior = ItemStrategyNextItemRule(ExhaustiveSearch1Ply(false), item_criterion_prior)

    rng = Xoshiro(42)
    (item_bank, abilities, responses) = dummy_full(
        rng,
        SimpleItemBankSpec(StdModel4PL(), OneDimContinuousDomain(), BooleanResponse());
        num_questions=100,
        num_testees=1
    )

    tracked_responses = TrackedResponses(
        BareResponses(ResponseType(item_bank), collect(1:10), collect(responses[1:10])),
        item_bank
    )
    @info "no log" mepv_prior(tracked_responses, item_bank)
    @time mepv_prior(tracked_responses, item_bank)

    log_item_bank = LogItemBank(item_bank)
    log_tracked_responses = TrackedResponses(
        BareResponses(ResponseType(item_bank), collect(1:10), collect(responses[1:10])),
        log_item_bank
    )
    @info "has log" mepv_prior(log_tracked_responses, log_item_bank)
    @time mepv_prior(log_tracked_responses, log_item_bank)
    3@profile mepv_prior(log_tracked_responses, log_item_bank)
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
