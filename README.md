# projects

Factor performance introduction
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
