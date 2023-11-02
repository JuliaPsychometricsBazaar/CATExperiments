using Parquet2
using CairoMakie
using DataFrames
using Parquet2: Dataset
using AlgebraOfGraphics
using Loess


function load(inf)
    ds = Dataset(inf)
    DataFrame(ds; copycols=false)
end


function main()
    df = load(ARGS[1])
    df[!, :num_items] = Int.(df[!, :num_items])
    df[!, :time] = Float64.(df[!, :time])
    CairoMakie.activate!()
    set_aog_theme!()
    #fig = Figure()
    #ax = Axis(fig[1, 1])
    #scatter!(ax, df[!, :num_items], df[!, :time])
    specs = data(df) * mapping(:num_items, :time, color=:system_name => nonnumeric) * (smooth() + visual(Scatter))
    specs += mapping([0.5], [0.5]) * visual(HLines, linestyle = :dash)
    fig = draw(
        specs;
        axis = (
            limits = (
                (32, 10^7),
                (-0.05, 1.0)
            ),
            xscale = log10,
            xlabel = "Number of items", 
            xtickformat = "{:.2f}",
            ylabel = "Runtime"
        ),
    )
    for ((system_name,), grp_df) in pairs(groupby(df, :system_name))
        model = Loess.loess(grp_df[!, :num_items], grp_df[!, :time]; span=0.75, degree=2)
        xs = round.(Int, 10 .^ range(2, log10(grp_df[!, :num_items][end]), length=1000))
        preds = predict(model, xs)
        min_idx = argmin(abs.(preds .- 0.5))
        num_items = xs[min_idx]
        println("$system_name: 0.5s @$num_items")
    end
    save(ARGS[2], fig)
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
