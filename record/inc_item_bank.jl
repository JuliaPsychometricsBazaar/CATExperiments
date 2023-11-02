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
using RCall

include("../bench/utils.jl")
include("../utils/RandomItemBanks.jl")


using .RandomItemBanks

MirtCat = Base.get_extension(RIrtWrappers, :MirtCat)
@assert MirtCat !== nothing

const all_num_items = round.(Int, 10 .^ (1.5:0.25:8))

function item_bank_to_params(item_bank, num_items)
    return [
        item_bank.inner_bank.inner_bank.difficulties[1:num_items],
        item_bank.inner_bank.inner_bank.discriminations[1:num_items],
        item_bank.inner_bank.guesses[1:num_items],
        item_bank.slips[1:num_items],
    ]
end

struct CatJLSystem{A, B, C}
    item_bank::A
    tracked_responses::B
    next_item_rule::C
end

function CatJLSystem(params, responses)
    lh_grid_tracker, next_item_rule = mk_catconfig()
    item_bank = ItemBank4PL(params...)
    tracked_responses = TrackedResponses(
        BareResponses(ResponseType(item_bank), collect(1:10), collect(responses[1:10])),
        item_bank,
        lh_grid_tracker
    )
    CatJLSystem(item_bank, tracked_responses, next_item_rule)
end

@inline function (system::CatJLSystem)()
    track!(system.tracked_responses)
    system.next_item_rule(system.tracked_responses, system.item_bank)
end

struct MirtCatSystem{A}
    mirt_design::A
end

function MirtCatSystem(params, responses)
    params_mat = hcat(params...)
    mirt_design = MirtCat.make_mirtcat(params_mat, "MEPV", "MEPV")[1]
    for i in 1:10
        mirt_design = MirtCat.next(mirt_design, i, responses[i])
    end
    MirtCatSystem(mirt_design)
end

@inline function (system::MirtCatSystem)()
    MirtCat.next_item(system.mirt_design)
end

struct SuperFastCatSystem{A}
    state::A
end

function SuperFastCatSystem(params, responses)
    params_mat = hcat(params...)
    state = SuperFastCat.FixedRectSimState(
        params_mat,
        i -> responses[i],
        11;
        quadpts=61,
        theta_lo=-6.0f0,
        theta_hi=6.0f0
    )
    SuperFastCat.precompute!(state)
    for i in 1:10
        SuperFastCat.advance_next_item!(state, i, responses[i])
    end
    SuperFastCatSystem(state)
end

@inline function (system::SuperFastCatSystem)()
    SuperFastCat.get_next_item!(system.state)
end

function mk_catconfig()
    integrator = even_grid(-6.0, 6.0, 61)
    lh_ability_est = LikelihoodAbilityEstimator()
    lh_grid_tracker = GriddedAbilityTracker(lh_ability_est, integrator)
    ability_integrator = AbilityIntegrator(integrator, lh_grid_tracker)
    ability_estimator = MeanAbilityEstimator(lh_ability_est, ability_integrator)
    next_item_rule = preallocate(catr_next_item_aliases["MEPV"](ability_estimator))
    return (lh_grid_tracker, next_item_rule)
end

function run(rw, systems, num_items, params, responses; write=true)
    next_systems = []
    for (system_name, mk_system) in systems
        @info "running" system_name num_items
        system = mk_system(params, responses)
        t = @timed system()
        time = t[:time]
        value = RCall.rcopy(t[:value])
        @info "value" value typeof(value)
        if write
            write_rec(
                rw;
                system_name=system_name,
                quantity="runtime",
                num_items=num_items,
                time=time
            )
            write_rec(
                rw;
                system_name=system_name,
                quantity="result",
                num_items=num_items,
                item_idx=value
            )
        end
        if time < 1.0
            push!(next_systems, (system_name, mk_system))
        end
    end
    return next_systems
end

function main()
    rng = Xoshiro(42)
    (full_item_bank, abilities, responses) = dummy_full(
        rng,
        SimpleItemBankSpec(StdModel4PL(), OneDimContinuousDomain(), BooleanResponse());
        num_questions=10^8,
        num_testees=1
    )

    systems = [
        ("catjl", CatJLSystem),
        ("mirtcat", MirtCatSystem),
        ("superfastcat", SuperFastCatSystem),
    ]

    open_rec_writer(ARGS[1]) do rw
        withenv() do
            # warmup
            run(rw, systems, 32, item_bank_to_params(full_item_bank, 32), responses; write=false)

            for num_items in all_num_items
                next_systems = run(rw, systems, num_items, item_bank_to_params(full_item_bank, num_items), responses)
                @info "iterating" systems
                if isempty(next_systems)
                    break
                end
                systems = next_systems
            end
        end
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
