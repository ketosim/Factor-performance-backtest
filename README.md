# Factor Performance Backtest

## **Purpose**
- Backtest fundamental and market-oriented factors and analyze the predictiveness of a factor based on different setups.
- Explore setups including sector neutrality, factor/equal weightings, and various rebalancing periods.

## **Input**
- Fundamental data and market trading data are used to construct factor data.

## **Processes**
1. **Data Construction:**
    - Input data (fundamental and trading data) is cleaned and quantized to construct factor data. Examples of factors include:
      - Price-to-Sales Ratio (P/S)
      - 1-Year Sales Growth

2. **Weightings of Stocks in Each Quantile:**
    - Calculate weightings based on different setups:
      - **Long-only portfolio**
      - **Long-short portfolio**
    - Weighting options include:
      - Sector-neutral, equal-weighted
      - Absolute equal-weighted
      - Sector-neutral, factor-weighted
      - Absolute factor-weighted

3. **Calculate Portfolio Returns:**
    - Calculate portfolio returns based on the selected rebalancing setup.
    - Options for rebalancing periods (e.g., monthly, quarterly).

4. **Visualize Results:**
    - Plot cumulative returns by quantile.
    - Provide detailed analysis of long-short portfolios and factor predictiveness.

## **Outputs**
- **Average Return:** Return for each quantile.
- **Top Quintile Minus Bottom Quintile:** Return difference for long-short portfolios.
- **Cumulative Returns:** Returns by quantile, including rebalancing effects.

### **Note:**
- Sector neutrality and weighting options are designed to account for cases where some sectors are naturally overrepresented in certain factor quintiles. For example, backtesting the P/S factor without sector neutrality from 2017-2020 primarily involves a long technology/short value strategy. Applying sector neutrality offers a clearer assessment of the factor’s performance over time.

- **Platform Dependency:** This notebook relies on input data from the Quantopian platform.

---

# Stock Correlation Visualization

## **Input**
- Daily price and volume data over any time frame.

## **Process**
1. **Calculate Sparse Inverse Covariances:**
    - Use `covariance.GraphicalLassoCV` on stock price or volume series to compute covariances.
    - This shows the connectedness of stocks, indicating how they are traded together as a group.

2. **Cluster Tickers:**
    - Group tickers based on covariances using **Affinity Propagation**, which does not require specifying the number of clusters.

3. **Visualize Connectedness:**
    - Color nodes by cluster membership.
    - Adjust edge thickness to reflect the strength of connectedness between tickers using **`manifold.LocallyLinearEmbedding`**.

## **Variations:**
- **Input 1:** Daily change = Close price - Open price
- **Input 2:** Daily volatility = Today’s high - Today’s low
- **Input 3:** Volume = Today’s volume - 10-day moving average volume

## **Uses:**
- **Analyze Basket Behavior:**
    - Given a basket (e.g., SaaS stocks), identify which stocks are traded differently from the group.

- **Monitor Market Events:**
    - Detect how affected stocks respond differently during significant market events.

- **Identify Market Shifts:**
    - Compare monthly trading behavior to detect shifts in market dynamics.

---

# Symbol Dictionary Generator for Bulk Analyses
- Generate symbol dictionaries to facilitate large-scale analyses across multiple tickers and datasets.
