
%matplotlib inline
from datetime import datetime
import matplotlib.pyplot as plt
plt.style.use('Solarize_Light2')


from matplotlib import style
import pandas as pd
import pandas_datareader.data as web
import yfinance as yf
from matplotlib import pyplot as plt
from mpl_finance import candlestick_ohlc
import numpy as np
import sys
from matplotlib.collections import LineCollection
from sklearn import cluster, covariance, manifold

print(__doc__)
yf.pdr_override()

def nasdaq_symbol_dict_generator(year_onward, sector ='all',industry='all'):
    nasdaq_tickers = pd.read_csv('https://old.nasdaq.com/screening/companies-by-name.aspx?letter=0&exchange=nasdaq&render=download')  
    nasdaq_tickers =nasdaq_tickers[nasdaq_tickers['IPOyear']<int(year_onward)]
    if not sector == 'all': 
        nasdaq_tickers=nasdaq_tickers[nasdaq_tickers['Sector'] ==str(sector)]
    if not industry == 'all':
        nasdaq_tickers=nasdaq_tickers[nasdaq_tickers['industry'] ==str(industry)]
            
    nasdaq_tickers['Name']=nasdaq_tickers['Name'].str.replace(", Inc.| Inc.| Limited.| Ltd.","")
    symbol_dict=dict(zip(nasdaq_tickers.Symbol,nasdaq_tickers.Name))
    #print(nasdaq_tickers)
    return symbol_dict
    
def nyse_symbol_dict_generator(year_onward, sector ='all',industry='all'):
    nyse_tickers = pd.read_csv('https://old.nasdaq.com/screening/companies-by-name.aspx?letter=0&exchange=nyse&render=download')  
    nyse_tickers =nyse_tickers[nyse_tickers['IPOyear']<int(year_onward)]
    if not sector == 'all': 
        nyse_tickers=nyse_tickers[nyse_tickers['Sector'] ==str(sector)]
    if not industry == 'all':
        nyse_tickers=nyse_tickers[nyse_tickers['industry'] ==str(industry)]
            
    nyse_tickers['Name']=nyse_tickers['Name'].str.replace(", Inc.| Inc.| Limited.| Ltd.","")
    symbol_dict=dict(zip(nyse_tickers.Symbol,nyse_tickers.Name))
    print(nyse_tickers)
    return symbol_dict
    
    
    
def ticker_dict_generator(tickers):
    symbol_dict = {}
    for ticker in tickers:
        try:
            name = str(yf.Ticker(str(ticker)).info['shortName'])
            symbol_dict[str(ticker).upper()] = str(name)
        except:
            print(ticker,'dictionary has failed')
            pass
        
    return symbol_dict
    
    
    
def visualisation(symbol_dict, start, end):
    quotes = []
    start = datetime.strptime(str(start), '%Y-%m-%d').date()
    end = datetime.strptime(str(end), '%Y-%m-%d').date()
    symbols, names = np.array(sorted(symbol_dict.items())).T
    #print(symbols)
    #print(names)
    j=0
    length=0
    first_run = 0
    
    for symbol in symbols:
    #print('Fetching quote history for %r' % symbol, file=sys.stderr)
        df = web.get_data_yahoo('{}'.format(symbol),start,end).dropna()
        length = len(df.index)
        
        if (length>0 |first_run==0):
            model_length=length
            print('model_length: ', model_length)
            first_run = 1
            
        if len(df.index) == model_length:
            quotes.append(df)
        
        else: 
            i, = np.where(symbols == str(symbol))
            print(i, symbol, ' download has failed')
            symbols =np.delete(symbols, i, 0)
            names = np.delete(names,i,0)
            j+=1
        
    print(j,'tickers are deleted')
    
    if len(symbols)<30:
        tall= 8
    else:
        tall = 12
    
    #print(quotes)
    close_prices = np.vstack([q['Close'] for q in quotes])
    open_prices = np.vstack([q['Open'] for q in quotes])
    variation = close_prices - open_prices
    
    #print(variation)
    #print(variation.shape)
    edge_model = covariance.GraphicalLassoCV(cv=5)
    #standardisation
    X = variation.copy().T
    X /= X.std(axis=0)
    #print(X)
    #print(X.shape)
    edge_model.fit(X)
    

    _, labels = cluster.affinity_propagation(edge_model.covariance_)
    #print(labels)
    n_labels = labels.max()

    for i in range(n_labels + 1):
        print('Cluster %i: %s' % ((i + 1), ', '.join(names[labels == i])))
    #for w in range(n_labels + 1):
        #print('Cluster %i: %s' % ((w + 1), ', '.join(symbols[labels == w])))    
        
    
    node_position_model = manifold.LocallyLinearEmbedding(
    n_components=2, eigen_solver='dense', n_neighbors=6)

    embedding = node_position_model.fit_transform(X.T).T

    plt.figure(1, facecolor='w', figsize=(10, tall))
    plt.clf()
    ax = plt.axes([0., 0., 1., 1.])
    plt.axis('off')

    partial_correlations = edge_model.precision_.copy()
    d = 1 / np.sqrt(np.diag(partial_correlations))
    partial_correlations *= d
    partial_correlations *= d[:, np.newaxis]
    non_zero = (np.abs(np.triu(partial_correlations, k=1)) > 0.02)

    plt.scatter(embedding[0], embedding[1], s=100 * d ** 2, c=labels,
                cmap=plt.cm.nipy_spectral)

    start_idx, end_idx = np.where(non_zero)

    segments = [[embedding[:, start], embedding[:, stop]]
                for start, stop in zip(start_idx, end_idx)]
    values = np.abs(partial_correlations[non_zero])
    lc = LineCollection(segments,
                        zorder=0, cmap=plt.cm.hot_r,
                        norm=plt.Normalize(0, .7 * values.max()))
    lc.set_array(values)
    lc.set_linewidths(15 * values)
    ax.add_collection(lc)

    for index, (name, label, (x, y)) in enumerate(
            zip(names, labels, embedding.T)):

        dx = x - embedding[0]
        dx[index] = 1
        dy = y - embedding[1]
        dy[index] = 1
        this_dx = dx[np.argmin(np.abs(dy))]
        this_dy = dy[np.argmin(np.abs(dx))]
        if this_dx > 0:
            horizontalalignment = 'left'
            x = x + .002
        else:
            horizontalalignment = 'right'
            x = x - .002
        if this_dy > 0:
            verticalalignment = 'bottom'
            y = y + .002
        else:
            verticalalignment = 'top'
            y = y - .002
        plt.text(x, y, name, size=10,
                 horizontalalignment=horizontalalignment,
                 verticalalignment=verticalalignment,
                 bbox=dict(facecolor='w',
                           edgecolor=plt.cm.nipy_spectral(label / float(n_labels)),
                           alpha=.6))

    plt.xlim(embedding[0].min() - .15 * embedding[0].ptp(),
             embedding[0].max() + .10 * embedding[0].ptp(),)
    plt.ylim(embedding[1].min() - .03 * embedding[1].ptp(),
             embedding[1].max() + .03 * embedding[1].ptp())
             
  def visualisation_volume(symbol_dict, start, end):
    quotes = []
    start = datetime.strptime(str(start), '%Y-%m-%d').date()
    end = datetime.strptime(str(end), '%Y-%m-%d').date()
    symbols, names = np.array(sorted(symbol_dict.items())).T
    print(symbols)
    #print(names)
    j=0
    length=0
    first_run = 0
    
    for symbol in symbols:
    #print('Fetching quote history for %r' % symbol, file=sys.stderr)
        df = web.get_data_yahoo('{}'.format(symbol),start,end)
        df['10ma'] = df['Volume'].rolling(window=10).mean()
        #df['ab_vol'] = df['Volume']- df['10ma']
        df.dropna(inplace=True)
        length = len(df.index)
        
        if (length>0 |first_run==0):
            model_length=length
            print('model_length: ', model_length)
            first_run = 1
            
        if len(df.index) == model_length:
            quotes.append(df)
        
        else: 
            i, = np.where(symbols == str(symbol))
            print(len(df.index))
            print(i, symbol, ' download has failed')
            symbols =np.delete(symbols, i, 0)
            names = np.delete(names,i,0)
            j+=1
        
    print(j,'ticker(s) being deleted')
    #print(symbols)
    #print(names)    
    
    if len(symbols)<30:
        tall= 8
    else:
        tall = 12
    
    
    vol = np.vstack([q['Volume'] for q in quotes])
    avg_vol = np.vstack([q['10ma'] for q in quotes])
    variation = vol - avg_vol
    
    edge_model = covariance.GraphicalLassoCV(cv=5)
    # standardisation
    X = variation.copy().T
    X /= X.std(axis=0)
    edge_model.fit(X)
    

    _, labels = cluster.affinity_propagation(edge_model.covariance_)
    #print(labels)
    n_labels = labels.max()

    for i in range(n_labels + 1):
        print('Cluster %i: %s' % ((i + 1), ', '.join(names[labels == i])))
    #for w in range(n_labels + 1):
        #print('Cluster %i: %s' % ((w + 1), ', '.join(symbols[labels == w])))    
        
    
    node_position_model = manifold.LocallyLinearEmbedding(
    n_components=2, eigen_solver='dense', n_neighbors=6)

    embedding = node_position_model.fit_transform(X.T).T

    plt.figure(1, facecolor='w', figsize=(10, tall))
    plt.clf()
    ax = plt.axes([0., 0., 1., 1.])
    plt.axis('off')

    partial_correlations = edge_model.precision_.copy()
    d = 1 / np.sqrt(np.diag(partial_correlations))
    partial_correlations *= d
    partial_correlations *= d[:, np.newaxis]
    non_zero = (np.abs(np.triu(partial_correlations, k=1)) > 0.02)

    plt.scatter(embedding[0], embedding[1], s=100 * d ** 2, c=labels,
                cmap=plt.cm.nipy_spectral)

    start_idx, end_idx = np.where(non_zero)

    segments = [[embedding[:, start], embedding[:, stop]]
                for start, stop in zip(start_idx, end_idx)]
    values = np.abs(partial_correlations[non_zero])
    lc = LineCollection(segments,
                        zorder=0, cmap=plt.cm.hot_r,
                        norm=plt.Normalize(0, .7 * values.max()))
    lc.set_array(values)
    lc.set_linewidths(15 * values)
    ax.add_collection(lc)

    for index, (name, label, (x, y)) in enumerate(
            zip(names, labels, embedding.T)):

        dx = x - embedding[0]
        dx[index] = 1
        dy = y - embedding[1]
        dy[index] = 1
        this_dx = dx[np.argmin(np.abs(dy))]
        this_dy = dy[np.argmin(np.abs(dx))]
        if this_dx > 0:
            horizontalalignment = 'left'
            x = x + .002
        else:
            horizontalalignment = 'right'
            x = x - .002
        if this_dy > 0:
            verticalalignment = 'bottom'
            y = y + .002
        else:
            verticalalignment = 'top'
            y = y - .002
        plt.text(x, y, name, size=10,
                 horizontalalignment=horizontalalignment,
                 verticalalignment=verticalalignment,
                 bbox=dict(facecolor='w',
                           edgecolor=plt.cm.nipy_spectral(label / float(n_labels)),
                           alpha=.6))

    plt.xlim(embedding[0].min() - .15 * embedding[0].ptp(),
             embedding[0].max() + .10 * embedding[0].ptp(),)
    plt.ylim(embedding[1].min() - .03 * embedding[1].ptp(),
             embedding[1].max() + .03 * embedding[1].ptp())      
             
             
def visualisation_high_minus_low(symbol_dict, start, end):
    quotes = []
    start = datetime.strptime(str(start), '%Y-%m-%d').date()
    end = datetime.strptime(str(end), '%Y-%m-%d').date()
    symbols, names = np.array(sorted(symbol_dict.items())).T
    print(symbols)
    #print(names)
    j=0
    length=0
    first_run = 0
    
    for symbol in symbols:
        #print('first_run',first_run)
    #print('Fetching quote history for %r' % symbol, file=sys.stderr)
        df = web.get_data_yahoo('{}'.format(symbol),start,end)
        length = len(df.index)
        
        if (length>0 |first_run==0):
            model_length=length
            print('model_length: ', model_length)
            first_run = 1
            
        if len(df.index) == model_length:
            quotes.append(df)
        
        else: 
            i, = np.where(symbols == str(symbol))
            print(i, symbol, ' download has failed')
            symbols =np.delete(symbols, i, 0)
            names = np.delete(names,i,0)
            j+=1
        
    print(j,'ticker(s) being deleted')
    #print(symbols)
    #print(names)    
    
    if len(symbols)<30:
        tall= 8
    else:
        tall = 12
    
    
    close_prices = np.vstack([q['High'] for q in quotes])
    open_prices = np.vstack([q['Low'] for q in quotes])
    variation = close_prices - open_prices
    
    edge_model = covariance.GraphicalLassoCV(cv=5)
    # standardisation
    X = variation.copy().T
    X /= X.std(axis=0)
    edge_model.fit(X)
    

    _, labels = cluster.affinity_propagation(edge_model.covariance_)
    #print(labels)
    n_labels = labels.max()

    for i in range(n_labels + 1):
        print('Cluster %i: %s' % ((i + 1), ', '.join(names[labels == i])))
    #for w in range(n_labels + 1):
        #print('Cluster %i: %s' % ((w + 1), ', '.join(symbols[labels == w])))    
        
    
    node_position_model = manifold.LocallyLinearEmbedding(
    n_components=2, eigen_solver='dense', n_neighbors=6)

    embedding = node_position_model.fit_transform(X.T).T

    plt.figure(1, facecolor='w', figsize=(10, tall))
    plt.clf()
    ax = plt.axes([0., 0., 1., 1.])
    plt.axis('off')
    #ax.text(0., 0., 'High minus low \n ', style='italic',
       # bbox={'facecolor': 'red', 'alpha': 0.5, 'pad': 10})
    #ax.set_xlabel('High minus low \n time span: ', str(start), str(end), symbols)
                 

    partial_correlations = edge_model.precision_.copy()
    d = 1 / np.sqrt(np.diag(partial_correlations))
    partial_correlations *= d
    partial_correlations *= d[:, np.newaxis]
    non_zero = (np.abs(np.triu(partial_correlations, k=1)) > 0.02)

    plt.scatter(embedding[0], embedding[1], s=100 * d ** 2, c=labels,
                cmap=plt.cm.nipy_spectral)

    start_idx, end_idx = np.where(non_zero)

    segments = [[embedding[:, start], embedding[:, stop]]
                for start, stop in zip(start_idx, end_idx)]
    values = np.abs(partial_correlations[non_zero])
    lc = LineCollection(segments,
                        zorder=0, cmap=plt.cm.hot_r,
                        norm=plt.Normalize(0, .7 * values.max()))
    lc.set_array(values)
    lc.set_linewidths(15 * values)
    ax.add_collection(lc)

    for index, (name, label, (x, y)) in enumerate(
            zip(names, labels, embedding.T)):

        dx = x - embedding[0]
        dx[index] = 1
        dy = y - embedding[1]
        dy[index] = 1
        this_dx = dx[np.argmin(np.abs(dy))]
        this_dy = dy[np.argmin(np.abs(dx))]
        if this_dx > 0:
            horizontalalignment = 'left'
            x = x + .002
        else:
            horizontalalignment = 'right'
            x = x - .002
        if this_dy > 0:
            verticalalignment = 'bottom'
            y = y + .002
        else:
            verticalalignment = 'top'
            y = y - .002
        plt.text(x, y, name, size=10,
                 horizontalalignment=horizontalalignment,
                 verticalalignment=verticalalignment,
                 bbox=dict(facecolor='w',
                           edgecolor=plt.cm.nipy_spectral(label / float(n_labels)),
                           alpha=.6))

    plt.xlim(embedding[0].min() - .15 * embedding[0].ptp(),
             embedding[0].max() + .10 * embedding[0].ptp(),)
    plt.ylim(embedding[1].min() - .03 * embedding[1].ptp(),
             embedding[1].max() + .03 * embedding[1].ptp())
             
   
