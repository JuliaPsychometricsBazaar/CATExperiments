include("../src/hacks.jl")

using Base
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
import RIrtWrappers
using RIrtWrappers.Mirt
using Random
using Distributions
using FittedItemBanks.DummyData: dummy_full, std_normal, SimpleItemBankSpec, StdModel4PL, VectorContinuousDomain, OneDimContinuousDomain, BooleanResponse
using CondaPkg
import SuperFastCat
using Profile

include("./utils/RandomItemBanks.jl")


using .RandomItemBanks

MirtCat = Base.get_extension(RIrtWrappers, :MirtCat)
@assert MirtCat !== nothing

const all_num_items = [20:10:100; 100:100:1000; 1000:1000:10000; 10000:10000:100000]

function item_bank_to_params(item_bank, num_items)
    return [
        item_bank.inner_bank.inner_bank.difficulties[1:num_items],
        item_bank.inner_bank.inner_bank.discriminations[1:num_items],
        item_bank.inner_bank.guesses[1:num_items],
        item_bank.slips[1:num_items],
    ]
end


function noisy_bisect(f, a, b, fa, fb, tolerance)
    while b - a >= tolerance
        mid = 0.5 * (a + b)
        fmid = f(mid)

        if fmid < fa || fmid > fb
            # Monotonicity violated.
            # Reached resolution of noise.
            break
        end

        if fmid < 0
            a, fa = mid, fmid
        else
            b, fb = mid, fmid
        end
    end

    return (a, b)
end

function time_cat_jl(params, responses, lh_grid_tracker, next_item_rule; profile=false)
    item_bank = ItemBank4PL(params...)
    tracked_responses = TrackedResponses(
        BareResponses(ResponseType(item_bank), collect(1:10), collect(responses[1:10])),
        item_bank,
        lh_grid_tracker
    )
    if profile
        next = @profile begin
            track!(tracked_responses)
            next_item_rule(tracked_responses, item_bank)
        end
        @info "cat_jl" next
    else
        t = @timed next_item_rule(tracked_responses, item_bank)
        @info "ComputerAdaptiveTesting.jl" length(item_bank) t.time
    end
end

function time_mirtcat(params_mat, responses)
    mirt_design = MirtCat.make_mirtcat(params_mat, "MEPV", "MEPV")[1]
    for i in 1:10
        mirt_design = MirtCat.next(mirt_design, i, responses[i])
    end
    t2 = @timed MirtCat.next_item(mirt_design)
    @info "mirtCAT" length(params_mat) t2.time
end

function time_superfastcat(params_mat, responses; profile=false)
    state = SuperFastCat.FixedRectSimState(params_mat, i -> responses[i], 11)
    SuperFastCat.precompute!(state)
    for i in 1:10
        SuperFastCat.advance_next_item!(state, i, responses[i])
    end
    @info "superfastcat" state.likelihood
    if profile
        local next
        for _ in 1:1000
            next = @profile SuperFastCat.get_next_item!(state)
        end
        @info "superfastcat" next
    else
        t3 = @timed SuperFastCat.get_next_item!(state)
        @info "superfastcat" size(params_mat, 1) t3.time
    end
end

function mk_catconfig()
    integrator = even_grid(-6.0, 6.0, 61)
    lh_ability_est = PriorAbilityEstimator(std_normal)
    lh_grid_tracker = GriddedAbilityTracker(lh_ability_est, integrator)
    ability_estimator = MeanAbilityEstimator(lh_ability_est, integrator, lh_grid_tracker)
    next_item_rule = preallocate(catr_next_item_aliases["MEPV"](ability_estimator))
    return (lh_grid_tracker, next_item_rule)
end

function main()
    if (length(ARGS) == 0)
        do_cat = true
        do_mirtcat = true
        do_superfastcat = true
    else
        do_cat = "cat" in ARGS
        do_mirtcat = "mirtcat" in ARGS
        do_superfastcat = "superfastcat" in ARGS
    end
    rng = Xoshiro(42)
    (full_item_bank, abilities, responses) = dummy_full(
        rng,
        SimpleItemBankSpec(StdModel4PL(), OneDimContinuousDomain(), BooleanResponse());
        num_questions=100000,
        num_testees=1
    )
    lh_grid_tracker, next_item_rule = mk_catconfig()

    #=
    for idx in 1:100000
        ir = ItemResponse(full_item_bank, idx)
        
        resp.(Ref(ir), integrator.grid)
    end
    =#

    withenv() do
        for num_items in all_num_items
            params = item_bank_to_params(full_item_bank, num_items)
            params_mat = hcat(params...)

            ## ComputerAdaptiveTesting.jl
            if do_cat
                time_cat_jl(params, responses, lh_grid_tracker, next_item_rule)
            end

            ## mirtcat
            if do_mirtcat
                time_mirtcat(params_mat, responses)
            end

            ## superfastcat
            if do_superfastcat
                time_superfastcat(params_mat, responses)
            end
        end
    end
end

function profilecatsuperfastcat()
    Profile.init(; delay=0.0001)
    rng = Xoshiro(42)
    (full_item_bank, abilities, responses) = dummy_full(
        rng,
        SimpleItemBankSpec(StdModel4PL(), OneDimContinuousDomain(), BooleanResponse());
        num_questions=100000,
        num_testees=1
    )
    lh_grid_tracker, next_item_rule = mk_catconfig()

    withenv() do
        params = item_bank_to_params(full_item_bank, 1000)
        params_mat = hcat(params...)

        ## ComputerAdaptiveTesting.jl
        Profile.clear()
        time_cat_jl(params, responses, lh_grid_tracker, next_item_rule; profile=true)
        VSCodeServer.view_profile()

        ## superfastcat
        Profile.clear()
        time_superfastcat(params_mat, responses; profile=true)
        VSCodeServer.view_profile()
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
