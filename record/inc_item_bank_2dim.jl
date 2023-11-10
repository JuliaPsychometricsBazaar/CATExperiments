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
using RCall
using QuasiMonteCarlo


include("../bench/utils.jl")
include("../utils/RandomItemBanks.jl")


using .RandomItemBanks

MirtCat = Base.get_extension(RIrtWrappers, :MirtCat)
@assert MirtCat !== nothing

const all_num_items = round.(Int, 10 .^ (1.5:0.25:8))

function item_bank_to_params(item_bank, num_items)
    return (
        item_bank.inner_bank.inner_bank.difficulties[1:num_items],
        item_bank.inner_bank.inner_bank.discriminations[:, 1:num_items],
        item_bank.inner_bank.guesses[1:num_items],
        item_bank.slips[1:num_items],
    )
end

struct CatJLSystem{A, B, C}
    item_bank::A
    tracked_responses::B
    next_item_rule::C
end

function CatJLSystem(params, responses)
    lh_grid_tracker, next_item_rule = mk_catconfig()
    @info "CatJLSystem" next_item_rule 
    item_bank = ItemBankMirt4PL(params...)
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

struct MirtCatSystem{A, R}
    mirt_design::A
    responses::R
end

function MirtCatSystem(params::Tuple, responses)
    params_mat = hcat(params[1], transpose(params[2]), params[3], params[4])
    # DRule as implemented by CatJL is actually postposterior DRule
    mirt_design = MirtCat.make_mirtcat(params_mat)[1]
    for i in 1:10
        mirt_design = MirtCat.next(mirt_design, i, responses[i])
    end
    MirtCatSystem(mirt_design, responses)
end

@inline function (system::MirtCatSystem)()
    idx = MirtCat.next_item(system.mirt_design, "DPrule")
    idx_jl = RCall.rcopy(idx)
    # Fair comparison since we measure the tracking cost for CatJL
    MirtCat.next(system.mirt_design, idx, system.responses[idx_jl])
    idx
end

function mk_catconfig()
    integrator = even_grid([-6.0, -6.0], [6.0, 6.0], 31)
    lh_ability_est = LikelihoodAbilityEstimator()
    #lh_grid_tracker = GriddedAbilityTracker(lh_ability_est, integrator)
    ability_integrator = AbilityIntegrator(integrator)
    ability_estimator = MeanAbilityEstimator(lh_ability_est, ability_integrator)
    next_item_rule = preallocate(NextItemRule(DRuleItemCriterion(ability_estimator)))
    # XXX: This is bad that preallocate breaks maybe_tracked_ability_estimate
    lh_point_tracker = PointAbilityTracker(next_item_rule.criterion.ability_estimator, [NaN, NaN])
    @info "mk_catconfig" lh_point_tracker
    return (lh_point_tracker, next_item_rule)
end

function run(rw, systems, num_items, params, responses; write=true)
    next_systems = []
    for (system_name, mk_system) in systems
        @info "running" system_name num_items
        t = @timed mk_system(params, responses)
        system = t[:value]
        init_time = t[:time]
        if write
            write_rec(
                rw;
                system_name=system_name,
                quantity="init_time",
                num_items=num_items,
                time=init_time
            )
        end
        t = @timed system()
        run_time = t[:time]
        value = RCall.rcopy(t[:value])
        @info "value" value typeof(value)
        total_time = init_time + run_time
        if write
            write_rec(
                rw;
                system_name=system_name,
                quantity="runtime",
                num_items=num_items,
                time=total_time
            )
            write_rec(
                rw;
                system_name=system_name,
                quantity="next_item_rule_runtime",
                num_items=num_items,
                time=run_time
            )
            write_rec(
                rw;
                system_name=system_name,
                quantity="result",
                num_items=num_items,
                item_idx=value
            )
        end
        if total_time < 1.0
            push!(next_systems, (system_name, mk_system))
        end
    end
    return next_systems
end

function main()
    rng = Xoshiro(42)
    (full_item_bank, abilities, responses) = dummy_full(
        rng,
        SimpleItemBankSpec(StdModel4PL(), VectorContinuousDomain(), BooleanResponse()),
        2;
        num_questions=10^8,
        num_testees=1
    )
    @info "main" full_item_bank

    systems = [
        ("catjl", CatJLSystem),
        ("mirtcat", MirtCatSystem),
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
