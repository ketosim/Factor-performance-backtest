# quantopian has stopped supporting import

import scipy.stats
import numpy as np
import pandas as pd
import datetime as dt
from datetime import date
import statsmodels.api as sm
import matplotlib.pyplot as plt
from statsmodels import regression
from quantopian.pipeline import Pipeline
from quantopian.research import run_pipeline
from quantopian.pipeline.data import morningstar
from quantopian.pipeline.data.builtin import USEquityPricing
from quantopian.pipeline.data.morningstar import Fundamentals
from quantopian.pipeline.factors import CustomFactor, Returns
from quantopian.pipeline.classifiers.fundamentals import Sector
from quantopian.pipeline.filters import QTradableStocksUS, Q1500US
from pandas.tseries.offsets import CustomBusinessDay, Day, BusinessDay

def make_pipeline():
    """
    Collect data via quantopian pipeline. 
    
    Output:
    A MultiIndex DataFrame indexed by date (level 0) and asset (level 1),
    including factors and sector code 
    """
    pipe = Pipeline()
    
    sector = Sector()
    
#    non_financial= (sector!= 103)
#    non_biotech = (sector!= 206)
    tech = sector.eq(311)
    financial = sector.eq(103)
    pe = Fundamentals.pe_ratio.latest
    ps = Fundamentals.ps_ratio.latest
    factor_filtered = ps.percentile_between(0.5,99.5)
    

    base_universe = (Q1500US() & factor_filtered)
    pipe.set_screen(base_universe)
    pipe.add(sector,'sector')
    pipe.add(ps, 'factor')   
    
#    eps_growth = Fundamentals.diluted_eps_growth.latest
#    earning_yield = Fundamentals.earning_yield.latest

    
    return pipe
  

def df_input_ready(start_date, end_date):
    pipe = make_pipeline()
    results = run_pipeline(pipe, start_date=start_date, end_date=end_date)
    results= results.reset_index()
    results.set_index(['level_0', 'level_1'], inplace=True)
    results.index=results.index.rename(['date', 'asset'])
    
    return results 
  
  
def quantize_factor(factor_data,
                    group_name,
                    factor_name,
                    quantiles=5,
                    by_group=True,
                    
                   ):
    """
    Construct factor quantiles, default=5

    Parameters
    ----------
    factor_data : DataFrame with date index, asset index, sector and factor
    
    group_name : default == 'sector' from data pipeline
        can be specified when other source data being used 
        
    factor_name : default == 'factor'from data pipeline
        can be specified when other source data being used 
        
    quantiles : equal-sized groups, quintile as default
    
    by_group : if true, quintiles are sector neutral 
        the highest factor stocks from each sector will be selected 
        for the top quintile
        

    Output:
        quantised factor df
    
    """
    
    def quantile_calc(x, _quantiles, ):
        return pd.qcut(x, _quantiles, labels=False) + 1
        
                
        

    grouper = [factor_data.index.get_level_values('date')]
    if by_group:
        grouper.append(str(group_name))

    factor_quantile = factor_data.groupby(grouper)[str(factor_name)] \
        .apply(quantile_calc, quantiles,)
    factor_quantile.name = 'quantile'
    factor = factor_quantile.dropna()
    return pd.DataFrame(factor)
  
  

  def plot_quantile_statistics_table(factor_data,q_name,factor_name):
    quantile_stats = factor_data.groupby(str(q_name)) \
        .agg(['min', 'max', 'mean', 'std', 'count'])[str(factor_name)]
    quantile_stats['count %'] = quantile_stats['count'] \
        / quantile_stats['count'].sum() * 100.
    
    print(quantile_stats)
    
    
ef factor_weights(factor_data,
                   factor_name,
                   group_name,
                   demeaned=False,
                   group_adjust=True,
                   equal_weight=False):
    """
    Computes asset weights by factor values and dividing by the sum of their
    absolute value (achieving gross leverage of 1). Positive factor values will
    results in positive weights and negative values in negative weights.

    Parameters
    ----------
    factor_data : DataFrame with date index, asset index, sector and factor
    
    group_name : default == 'sector' from data pipeline
        can be specified when other source data being used 
        
    factor_name : default == 'factor'from data pipeline
        can be specified when other source data being used 
        
    demeaned : bool
        Should this computation happen on a long short portfolio? if True,
        weights are computed by demeaning factor values and dividing by the sum
        of their absolute value (achieving gross leverage of 1). The sum of
        positive weights will be the same as the negative weights (absolute
        value), suitable for a dollar neutral long-short portfolio
        
    group_adjust : bool
        Should this computation happen on a group neutral portfolio? If True,
        compute group neutral weights: each group will weight the same and
        if 'demeaned' is enabled the factor values demeaning will occur on the
        group level.
        
    equal_weight : bool, optional
        If True the assets will be equal-weighted instead of factor-weighted
        

    Returns
    -------
    pd.Series
        Assets weighted by factor value.
    """

    def to_weights(group,_demeaned, _equal_weight):

        if _equal_weight:
            group = group.copy()

            if _demeaned:
                # top assets positive weights, bottom ones negative
                group = group - group.median()

            negative_mask = group < 0
            group[negative_mask] = -1.0
            positive_mask = group > 0
            group[positive_mask] = 1.0

            if _demeaned:
                # positive weights must equal negative weights
                if negative_mask.any():
                    group[negative_mask] /= negative_mask.sum()
                if positive_mask.any():
                    group[positive_mask] /= positive_mask.sum()
                    
        elif _demeaned:
            group = group - group.mean()

        return group / group.abs().sum()

    grouper = [factor_data.index.get_level_values('date')]
    if group_adjust:
        grouper.append(str(group_name))
        # grouper includes date and sector

    weights = factor_data.groupby(grouper)[str(factor_name)] \
        .apply(to_weights, demeaned, equal_weight)

    if group_adjust:
        weights = weights.groupby(level='date').apply(to_weights, False, False)
        # here the weights are adjsuted by no. of sectors again
        # e.g. appl wieghts is halved since the sample has two sectors 

    return weights
# if demeaned=False, equal_weight=False, it means it's a long-only portfolio and 
# long weights are factor values and dividing by the sum of their absolute value 

# if demeaned=True(default), equal_weight=False, it means it is a long-short 
# portfolio, their weights are factor determined 

# if group_adjust=True, grouper includes date and sector
# weights.apple_to_weights AGAIN 
# the weights are adjsuted by no. of sectors 
# e.g. appl wieghts is halved since the sample has two sectors 


def calculate_forward_returns(prices_df,period_num,factor_df):
    factor_dateindex = factor_df.index.levels[0]
    factor_dateindex = factor_dateindex.intersection(prices_df.index)
    prices = prices_df.filter(items=factor_df.index.levels[1])
    returns = prices_df.pct_change(period_num)
    forward_returns =  returns.shift(-period_num).reindex(factor_dateindex)
    
    raw_values_dict = {}
    # it is one instance with 5d period forward returns
    #'5D' is the label that goes to the column name in the final df
    # raw_values_dict 
    # {'5D': array([-0.02609076, -0.01439414,  0.00532871, ..., -0.00274361,
    #        -0.03166992,  0.02343352])}
    label = '{}D'.format(str(period_num))
    column_list = []
    column_list.append(label)
    raw_values_dict[label] = np.concatenate(forward_returns.values)
    
    forward_ret = pd.DataFrame.from_dict(raw_values_dict)

    forward_ret.set_index(
        pd.MultiIndex.from_product(
            [factor_dateindex, prices.columns],
            names=['date', 'asset']
        ),inplace=True)
#df = df.reindex(factor.index)
    forward_ret.index.levels[0].freq = '1D'
    return forward_ret

  
  
def merged_data(factor_df,prices,forward_period):
    # merge forward return, factor, quantile, sector
    quantile_copy = quantize_factor(factor_df,'sector','factor').copy()
    merged_data = calculate_forward_returns(prices,int(forward_period),factor_df).copy()
    merged_data['quantile'] = quantile_copy
    merged_data['sector'] = factor_df['sector']
    merged_data['factor'] = factor_df['factor']
    merged_data = merged_data.reset_index()
    merged_data['date'] = pd.to_datetime(merged_data['date'])
    merged_data.set_index(['date', 'asset'], inplace=True)
    merged_data = merged_data.dropna()
    
    
    return merged_data
  
  
def demean_forward_returns(factor_data,forward_ret_col_name, grouper=None):
    """
    Convert forward returns to returns relative to mean
    period wise all-universe or group returns.
    group-wise normalization incorporates the assumption of a
    group neutral portfolio constraint and thus allows allows the
    factor to be evaluated across groups.

    For example, if AAPL 5 period return is 0.1% and mean 5 period
    return for the Technology stocks in our universe is 0.5% in the
    same period, the group adjusted 5 period return for AAPL in this
    period is -0.4%.

    Parameters
    ----------
    factor_data : pd.DataFrame - MultiIndex
        Forward returns indexed by date and asset.
        Separate column for each forward return window.
    grouper : list
        If True, demean according to group.

    Returns
    -------
    adjusted_forward_returns : pd.DataFrame - MultiIndex
        DataFrame of the same format as the input, but with each
        security's returns normalized by group.
    """

    factor_data = factor_data.copy()

    if not grouper:
        grouper = factor_data.index.get_level_values('date')

    cols = str(forward_ret_col_name)
    factor_data[cols] = factor_data.groupby(grouper)[cols] \
        .transform(lambda x: x - x.mean())

    return factor_data

# demean_forward_returns(data,'5D', grouper='sector')


def mean_return_by_quantile(factor_data,forward_ret_col,
                            by_date=True,
                            by_group=False,
                            demeaned=False,
                            group_adjust=False):
    
    
    """
    Computes mean returns for factor quantiles across
    provided forward returns columns.

    Returns
    -------
    mean_ret : pd.DataFrame
        Mean period wise returns by specified factor quantile.
    std_error_ret : pd.DataFrame
        Standard error of returns by specified quantile.
    """

    if group_adjust:
        grouper = [factor_data.index.get_level_values('date')] + ['sector']
        factor_data = demean_forward_returns(factor_data,str(forward_ret_col), grouper)
    elif demeaned:
        factor_data = demean_forward_returns(factor_data,str(forward_ret_col))
    else:
        factor_data = factor_data.copy()

    grouper = ['quantile', factor_data.index.get_level_values('date')]

    if by_group:
        grouper.append('sector')

    group_stats = factor_data.groupby(grouper)[str(forward_ret_col)] \
        .agg(['mean', 'std', 'count'])
    

    mean_ret = group_stats.T.xs('mean').T
    if not by_date:
        grouper = [mean_ret.index.get_level_values('quantile')]
        if by_group:
            grouper.append(mean_ret.index.get_level_values('sector'))
        group_stats = mean_ret.groupby(grouper)\
            .agg(['mean', 'std', 'count'])
        mean_ret = group_stats.T.xs('mean').T

    std_error_ret = group_stats.T.xs('std').T \
        / np.sqrt(group_stats.T.xs('count').T)
    
    mean_ret=pd.DataFrame(mean_ret)
    std_error_ret=pd.DataFrame(std_error_ret)
    

    return mean_ret, std_error_ret
  
  
def weighted_sum_return_by_quantile(factor_data,forward_ret_col,
                            by_date,
                            by_group,
                            demeaned,
                            group_adjust):
    
     
    """
    Computes mean returns for factor quantiles across
    provided forward returns columns.

    Returns
    -------
    mean_ret : pd.DataFrame
        Mean period wise returns by specified factor quantile.
    std_error_ret : pd.DataFrame
        Standard error of returns by specified quantile.
    """

    if group_adjust:
        grouper = [factor_data.index.get_level_values('date')] + ['sector']
        factor_data = demean_forward_returns(factor_data,str(forward_ret_col), grouper)
    elif demeaned:
        factor_data = demean_forward_returns(factor_data,str(forward_ret_col))
    else:
        factor_data = factor_data.copy()

    grouper = ['quantile', factor_data.index.get_level_values('date')]

    if by_group:
        grouper.append('sector')

    group_stats = factor_data.groupby(grouper)[str(forward_ret_col)] \
        .agg(['sum', 'std', 'count'])
    

    sum_ret= group_stats.T.xs('sum').T
    if not by_date:
        grouper = [mean_ret.index.get_level_values('quantile')]
        if by_group:
            grouper.append(mean_ret.index.get_level_values('sector'))
        group_stats = mean_ret.groupby(grouper)\
            .agg(['sum', 'std', 'count'])
        sum_ret = group_stats.T.xs('sum').T

    std_error_ret = group_stats.T.xs('std').T \
        / np.sqrt(group_stats.T.xs('count').T)
    
    sum_ret=pd.DataFrame(sum_ret)
    std_error_ret=pd.DataFrame(std_error_ret)
    

    return sum_ret
  
  
def mean_quant_ret_cal(df_input, 
                            forward_period, 
                            group_adjust,
                            equal_weight,
                            demeaned,
                            by_group,
                            cumulative_ret = True,
                            group_name='sector'
                               
                           ):
    # running pipeline is a standalone execution, to avoid multiple calls
    # call df_input=df_input_ready() before this
    # Given the factor is ready
    # forward_period: int 5,10,22...252
    # factor_col_name = 'pb' or other factors from df_input
    # group_name = 'sector'
    # group_adjust = True when factor should be 
    # loaded across all sectors evenlly, unless factors are momentum or others 
    # equal_weighted = False to amplify factor effects
    # demeaned = True --> long/short portfolio
       # by_group : bool
   #     If True, compute quantile bucket returns separately for each group.
    
    prices = get_pricing(df_input.index.levels[1],
                            start_date = start_date,
                            end_date = dt.datetime.today() - dt.timedelta(days=2),
                            fields = 'open_price',
                            frequency='daily'  )
    
    print ('Pricing data fetched')
    
    
    forward_ret_col_name = '{}D'.format(str(forward_period))
    
    
    data=merged_data(df_input,prices,
                     forward_period)

    
    weights = factor_weights(data,
                             'factor',
                             'sector',
                            group_adjust=group_adjust,
                            equal_weight=equal_weight,
                            demeaned=False)
    

    
    weighted_returns = data[forward_ret_col_name].multiply(weights, axis=0)
    data['weighted_returns'] = weighted_returns
    
    
    print ('Forward returns merged')
    
    sum_quant_ret = weighted_sum_return_by_quantile(data,
                                'weighted_returns',
                                by_date=True,
                                by_group=by_group,
                                demeaned=demeaned,
                                group_adjust=group_adjust)
    
    
    
        
    mean_quant_ret, std_quantile = mean_return_by_quantile(data,
                                forward_ret_col_name,
                                by_date=True,
                                by_group=by_group,
                                demeaned=demeaned,
                                group_adjust=group_adjust)
    
    std1 = std_quantile.xs(1, level='quantile')
    std5 = std_quantile.xs(5, level='quantile')
    joint_std_err = np.sqrt(std1**2 + std5**2)

    
    quant_ret_spread = mean_quant_ret.xs(1,level='quantile') \
            - mean_quant_ret.xs(5,level='quantile')
    

    
    return sum_quant_ret, mean_quant_ret, quant_ret_spread, std_quantile, joint_std_err
    
    
    
def plot_quantile_performance(forward_period,mean_quant_ret,quant_ret_spread,sum_quant_ret, joint_std_err):
    DECIMAL_TO_BPS = 10000

    fig2 = plt.figure(figsize=(16, 24))
    spec2 = plt.GridSpec(ncols=1, nrows=3)
    f2_ax1 = fig2.add_subplot(spec2[0, :])

    q1= mean_quant_ret.xs(1,level='quantile')*DECIMAL_TO_BPS
    q1.plot(ax=f2_ax1)
    
    q2= mean_quant_ret.xs(2,level='quantile')*DECIMAL_TO_BPS
    q2.plot(ax=f2_ax1)
    
    q3= mean_quant_ret.xs(3,level='quantile')*DECIMAL_TO_BPS
    q3.plot(ax=f2_ax1)
    
    q4= mean_quant_ret.xs(4,level='quantile')*DECIMAL_TO_BPS
    q4.plot(ax=f2_ax1)

    q5= mean_quant_ret.xs(5,level='quantile')*DECIMAL_TO_BPS
    q5.plot(title='Quantiles Mean Returns (bps)',ax=f2_ax1)
    


    f2_ax2 = fig2.add_subplot(spec2[1, :])
    q5_q1=quant_ret_spread*DECIMAL_TO_BPS
    joint_std_err_bps = joint_std_err*DECIMAL_TO_BPS
    
    upper = q5_q1.values + (joint_std_err_bps * 1)
    Y1 = upper[upper.columns[0]]
    lower = q5_q1.values - (joint_std_err_bps * 1)
    Y2 = lower[lower.columns[0]]
    f2_ax2.fill_between(quant_ret_spread.index,
                        Y1,
                        Y2,
                        alpha=0.3)
    
    
    q5_q1.plot(title='Top Minus Bottom Quantile Mean Return ({} Period Forward Return)' \
               .format(str(forward_period)),ax=f2_ax2)

    
    f2_ax3 = fig2.add_subplot(spec2[2, :])
    cumul_1 = (1 + sum_quant_ret.xs(1,level='quantile')).cumprod() - 1
    cumul_2 = (1 + sum_quant_ret.xs(2,level='quantile')).cumprod() - 1
    cumul_3 = (1 + sum_quant_ret.xs(3,level='quantile')).cumprod() - 1
    cumul_4 = (1 + sum_quant_ret.xs(4,level='quantile')).cumprod() - 1
    cumul_5 = (1 + sum_quant_ret.xs(5,level='quantile')).cumprod() - 1
    
    cumul_1.plot(ax=f2_ax3)
    cumul_2.plot(ax=f2_ax3)
    cumul_3.plot(ax=f2_ax3)
    cumul_4.plot(ax=f2_ax3)
    cumul_5.plot(ax=f2_ax3, title='Cumulative Quantile Returns')
    
    
    f2_ax1.legend(loc='upper left')
    f2_ax1.set(xlabel="",ylabel='Quantile Mean Return (bps)')
    f2_ax1_l=f2_ax1.legend()

    f2_ax1_l.get_texts()[0].set_text('Q1')
    f2_ax1_l.get_texts()[1].set_text('Q2')
    f2_ax1_l.get_texts()[2].set_text('Q3')
    f2_ax1_l.get_texts()[3].set_text('Q4')
    f2_ax1_l.get_texts()[4].set_text('Q5')
    
    f2_ax2.legend(loc='upper left')
    f2_ax2.set(xlabel="",ylabel=\
               '{}D Cumulative Factor Return Spread (bps)'.format(str(forward_period))
               )

    
    f2_ax2_l=f2_ax2.legend()
    f2_ax2_l.get_texts()[0].set_text('mean return  Q1-Q5')
    
    f2_ax3.legend(loc='upper left')
    f2_ax3.set(xlabel="",ylabel='Quantile Cumulative Return')
    f2_ax3_l=f2_ax3.legend()

    f2_ax3_l.get_texts()[0].set_text('Q1')
    f2_ax3_l.get_texts()[1].set_text('Q2')
    f2_ax3_l.get_texts()[2].set_text('Q3')
    f2_ax3_l.get_texts()[3].set_text('Q4')
    f2_ax3_l.get_texts()[4].set_text('Q5')
    
    
    
    
def plot_mean_cumul_ret_by_q(start_date,
                        end_date,
                        forward_period,
                        group_adjust,
                        equal_weight,
                        demeaned,
                        by_group=False
                       ):
    df_input=df_input_ready(start_date = start_date,end_date = end_date)
    sum_quant_ret, mean_quant_ret, quant_ret_spread, std_quantile, joint_std_err =\
            mean_quant_ret_cal(df_input, 
                           forward_period,
                           group_adjust=group_adjust,
                            equal_weight=equal_weight,
                            demeaned=demeaned,
                            by_group=by_group,
                          )
    plot_quantile_performance(forward_period,mean_quant_ret,quant_ret_spread,sum_quant_ret,joint_std_err)

    
    
def plot_mean_cumul_ret_by_q_with_df_input_ready(
                        df_input,
                        forward_period,
                        group_adjust,
                        equal_weight,
                        demeaned,
                        by_group=False
                       ):
    
    sum_quant_ret, mean_quant_ret, quant_ret_spread, std_quantile, joint_std_err =\
            mean_quant_ret_cal(df_input,
                    forward_period,
                           group_adjust=group_adjust,
                            equal_weight=equal_weight,
                            demeaned=demeaned,
                            by_group=by_group,
                          )
    plot_quantile_performance(forward_period,mean_quant_ret,quant_ret_spread,sum_quant_ret,joint_std_err)
