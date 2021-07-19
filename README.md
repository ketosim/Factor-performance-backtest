# projects
Factor performance backtest


Purposes:

Backtest fundamental and market-oriented factors and analyse the predictiveness of a factor based on different set-ups
Set-ups include sector neutrality, factor/equal weightings and different rebalancing periods


Input:

Fundamental data and market trading data are used to construct factor data


Processes:

Input data (fundmantal and trading data to form a factor, such as P/S, 1Yr sales growth) are cleaned and quantised to construct factor data
Weightings of stocks in each quantile are calcualted. Options for weightings include long-only portfolio and long-short portpolio of the following set-ups: sector neutral equal-weighted, absolute equal-weighted, sector neutral factor-weighted and absolute factor-weighted*
Calculate portfolio returns based on rebalancing set-up and plot the results


Outputs:

Average return for each quantile
Top quintile return minus bottom quintile return (long-short portfolio)
Cumulative returns by quintile with rebalancing
*Sector neutrality and weighting options are designed to cover a wide range of factors -- some sectors are overly represented in a factor quintile by nature. For example, backtesting P/S factor without sector neutrality over the period of 2017-2020 is essentially a long technology and short value stocks strategy. Allowing sector neutrality gives a purer picture of the factor performance over time, free from heavily skewed market trend.

The notebook relies on input data from Quantopian platform.




Stock Correlation visualisation


Input:

daily price and volume data in any time frame


Process:

calulate sparse inverse covariances (covariance.GraphicalLassoCV) of all stocks price(or volume) series
covariances show connectedness of all tickers, i.e. stocks are traded together as a group
group tickers based on covariances (Affinity Propagation, non-equal number clusters)
visulisation: color of nodes showing clusters, thickness of edges showing connectedness (manifold.LocallyLinearEmbedding)
Variations:

input 1 daily_change = close_price - open_price
input 2 daily_volatility = today's high - today's low
input 3 volume = today's volume - 10 day moving average volume


Uses:

Given a basket, say Saas stocks, see who is traded away from the basket
Given a market event, see how the affected stocks react differently
Compare monthly trading behaviours to identify market shifts


Others:

symbol dictionary generator for bulk analyses
