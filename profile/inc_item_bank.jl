include("../record/inc_item_bank.jl")

using Profile


function profilecatsuperfastcat()
    Profile.init(; delay=0.00002)
    rng = Xoshiro(42)
    (full_item_bank, abilities, responses) = dummy_full(
        rng,
        SimpleItemBankSpec(StdModel4PL(), OneDimContinuousDomain(), BooleanResponse());
        num_questions=100000,
        num_testees=1
    )

    params = item_bank_to_params(full_item_bank, 100000)
    systems = [mk_system(params, responses) for mk_system in [
        CatJLSystem,
        SuperFastCatSystem
    ]]

    ## ComputerAdaptiveTesting.jl
    Profile.clear()
    @profile begin
        systems[1]()
    end
    VSCodeServer.view_profile()

    ## superfastcat
    Profile.clear()
    @profile begin
        systems[2]()
    end
    VSCodeServer.view_profile()
end