#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 30 22:24:28 2023

@author: kunyu
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct  6 23:53:10 2023

@author: kunyu
"""

import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
import os
import seaborn as sns
import tarfile

import fire

#%%
Tickfile_Dir = "/Users/kunyu/Desktop/Exploratory Trading"
Underlying_csv = "/Users/kunyu/Desktop/Exploratory Trading/derived_underlying.future.csv"

Underlyings = pd.read_csv(Underlying_csv).drop_duplicates(subset=['name'], keep="last").set_index("name")
#print(Underlyings['multiplier'])
spilandTrade1 = pd.read_csv('/Users/kunyu/Desktop/Exploratory Trading/spiland.bin.2023-09-15am/spiland.Trade.csv')
spilandTrade2 = pd.read_csv('/Users/kunyu/Desktop/Exploratory Trading/spiland.bin.2023-09-15pm/spiland.Trade.csv')
Spiland = pd.concat([spilandTrade1, spilandTrade2])
#print(Spiland)


CALENDAR_PATH = "/Users/kunyu/Desktop/Exploratory Trading/derived_calendar.csv"

Calendar = pd.read_csv(CALENDAR_PATH,
                       parse_dates=['date','prevTradingDay','nextTradingDay'],
                       index_col='date')

def get_trades_for_date(instr = 'i2401'):
    data_instr = Spiland[Spiland['instrument'] == instr]
    trades = data_instr[['recvTime', 'instrument', 'strategy', 'side', 'volume', 'price']]
    
    return trades
"""    
tdate = pd.Timestamp("2023-09-15")
trades = get_trades_for_date(tdate, instr='i2401')
print(trades)
"""


def prod_orders_on_date(server:str, tdate:pd.Timestamp):
    pass

def prod_sodPos_for_date(server:str, tdate:pd.Timestamp):
    pass

def get_prod_pos_orders_for_date(server:str, tdate: pd.Timestamp) :
# -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
  """ 
  Inputs:
    server
    tradingDay
  Outputs:
    instrmentPosition,
    strategyPosition, 
    orders, 
    trades
  """

  instP, stgyP, orders, trades = prod_orders_on_date(server, tdate)
  instP_prev, stgyP_prev = prod_sodPos_for_date(server, tdate)
  return pd.concat([instP_prev, instP]), pd.concat([stgyP_prev, stgyP]), orders, trades

#
def get_instr_price(tradingDay=pd.Timestamp("2022-02-23"), instr="ni2203", interval='60s'):
    
    tradingDay = pd.Timestamp(tradingDay)
    
    underly = "".join([c.upper() for c in instr if not c.isdigit()])
    
    tickfile_n = f"{Tickfile_Dir}/{tradingDay:%Y}/{tradingDay:%Y%m%d}n_{underly}.csv"
    tickfile_d = f"{Tickfile_Dir}/{tradingDay:%Y}/{tradingDay:%Y%m%d}d_{underly}.csv"

    usecols=['InstrumentID','TradingDay','LastPrice','PreClosePrice','UpdateTime']
    datecols = ['TradingDay', 'UpdateTime']


    tick_prc = pd.DataFrame()
    if os.path.exists(tickfile_n):
        tick_prc_n = pd.read_csv(tickfile_n, parse_dates=datecols, usecols=usecols)
        #print(tick_prc_n['InstrumentID'].unique())  # Debugging print statement
        tick_prc = pd.concat([tick_prc, tick_prc_n])
    if os.path.exists(tickfile_d):
        tick_prc_d = pd.read_csv(tickfile_d, parse_dates=datecols, usecols=usecols)
        #print(tick_prc_n['InstrumentID'].unique())  # Debugging print statement
        tick_prc = pd.concat([tick_prc, tick_prc_d])

    if tick_prc.empty:
        print(f"Cannot get tick files for {tradingDay:%Y-%m-%d} for {instr}")
        return None
    
    flds = ["UpdateTime", "InstrumentID", "LastPrice"]
    tick_prc = tick_prc.loc[tick_prc['InstrumentID'] == instr, flds]
    #print(tick_prc['InstrumentID'].unique())
    tick_prc.rename(columns={'UpdateTime':'timestamp', 'InstrumentID': 'instrument', 'LastPrice':'price'}, inplace=True)
    tick_prc = tick_prc.set_index('timestamp').sort_index().resample(interval).last()
    # Handle NaN values (choose either Fill Forward or Drop NaN Rows)
    tick_prc['instrument'].fillna(method='ffill', inplace=True)  # Fill Forward

    #print("After Resampling:", tick_prc['instrument'].head().unique())

    
    return tick_prc
def get_underly_early_prc(tradingDay=pd.Timestamp("2022-01-05"), instr="CJ205", nrows=100):

    usecols=['InstrumentID','TradingDay','LastPrice','PreClosePrice','OpenPrice','UpdateTime']
    datecols = ['TradingDay', 'UpdateTime']
    tradingDay = pd.Timestamp(tradingDay)
    
    underly = "".join([c.upper() for c in instr if not c.isdigit()])
    
    tickfile_n = f"{Tickfile_Dir}/{tradingDay:%Y}/{tradingDay:%Y%m%d}n_{underly}.csv"
    tickfile_d = f"{Tickfile_Dir}/{tradingDay:%Y}/{tradingDay:%Y%m%d}d_{underly}.csv"
    
    tick_prc = pd.DataFrame()
    if os.path.exists(tickfile_n):
        tick_prc_n = pd.read_csv(tickfile_n, parse_dates=datecols,nrows=nrows,usecols=usecols)
        tick_prc = pd.concat([tick_prc, tick_prc_n])
    if os.path.exists(tickfile_d):
        tick_prc_d = pd.read_csv(tickfile_d, parse_dates=datecols,nrows=nrows, usecols=usecols)
        tick_prc = pd.concat([tick_prc, tick_prc_d])

    if tick_prc.empty:
        print(f"Cannot get tick files for {tradingDay:%Y-%m-%d} for {instr}")
        return None
    
    tick_prc.rename(columns={'UpdateTime':'timestamp', 'LastPrice':'price'}, inplace=True)
    tick_prc = tick_prc[tick_prc['OpenPrice'] > 0]
    tick_prc = tick_prc.sort_values('timestamp').drop_duplicates(subset=['InstrumentID','TradingDay'],keep='first')
    
    return tick_prc

#price_data = get_instr_price(tradingDay="2023-09-15", instr="i2401", interval='60s')
#early_tick_data = get_underly_early_prc(tradingDay="2023-09-15", instr="i2401", nrows=100)
"""
if price_data['instrument'].notnull().all():
    print("All values in 'instrument' are non-missing.")
else:
    print("There are missing values in 'instrument'.")
"""
def get_pre_close_prices(tdate:pd.Timestamp, underlyings:list):
    pre_close_lst = []
    for instr in underlyings:
        pre_close_lst.append(get_underly_early_prc(tdate, instr))
    pre_close = pd.concat(pre_close_lst)
    return pre_close

#pre_close_prices = get_pre_close_prices(pd.Timestamp("2023-09-15"), ["i2401"])


#%%
def pnl_one_instrument(trade_df):
    
    #print(trade_df.columns)
    assert 1 == len(trade_df['instrument'].unique())
    
    try:
        trds = trade_df.set_index('timestamp').sort_index()
    except:
        trds = trade_df.sort_values('timestamp')
    
    trds['sgn'] = 1
    trds.loc[trds['side']=='S', 'sgn'] = -1
    #print(trds.columns)
    
    
    
    trds['sgnQty'] = trds['sgn'] * trds['volume'] *trds['multi']
    trds['pay'] = -trds['price'] * trds['sgnQty']
    trds['fee'] = np.abs(trds['pay']) * 0e-4  # 1 bps per round trip

    trds['pos'] = trds['sgnQty'].cumsum()
    trds['val'] = trds['pos'] * trds['price'] #value at last trade price
    trds['rmb'] = (trds['pay'] - trds['fee']).cumsum()

    trds['pnl'] = trds['val'] + trds['rmb']
    return trds


#%%

def pnl_day_instr(trade_df, tradingDay, instr):
    aa = pnl_one_instrument(trade_df[trade_df['instrument']==instr])
    #print(aa[['rmb','pos','multi']])
    for col in ['rmb', 'pos', 'multi']:
        aa[col] = aa[col].astype(float)
    
    tradingDay = pd.Timestamp(tradingDay)
    mktprc = get_instr_price(tradingDay, instr, interval='60s')
    #print(mktprc)
    mktprc.index = pd.to_datetime(mktprc.index)
    aa.index = pd.to_datetime(aa.index)
    
    #df = pd.merge(mktprc, aa[['instrument', 'price', 'sgn', 'pay', 'rmb', 'pos', 'multi']], left_index=True, right_index=True, how='outer')
    
    #df = df.rename(columns={'price_x': 'mkt_price', 'price_y': 'trade_price'})

    #df = pd.merge_asof(mktprc, aa[['sgn', 'pay', 'rmb','pos','multi']], left_index=True, right_index=True)
    #df = pd.concat([mktprc, aa[['instrument','price','sgn', 'pay', 'rmb','pos','multi']]])
    df = pd.merge_asof(mktprc.sort_index(), aa[['instrument','price','sgn', 'pay', 'rmb','pos','multi']].sort_index(), 
                       left_index=True, right_index=True, direction='backward', suffixes=('_mkt', '_trade'))
    #df = df.sort_index()
    #print(df.columns)
    
    df[['rmb','pos','multi']] = df[['rmb','pos','multi']].fillna(method='ffill')
    df[['rmb','pos','multi']] = df[['rmb','pos','multi']].fillna(0)
    #print(df[['rmb','pos','multi']])
    
    df['val'] = df['pos'] * df['price_mkt']
    df['pnl'] = df['val'] + df['rmb']
    return df

def _plot_pnl_tseries(prev_df: pd.DataFrame, prc_pos_df:pd.DataFrame, tdate:pd.Timestamp, instr:str,
        server:str=None, stgy:str=None):
    curve = prc_pos_df.dropna(subset=['price_mkt'])
    curve = curve.reset_index()
    
    prev_df = prev_df.reset_index()
    prev_df.index = pd.to_datetime(prev_df.index)

    # for curve's x axis scale and labels.
    is_hour = curve['timestamp'].apply(lambda x: (x.second==0)  & (x.minute in [0, 30]))
    xticks = is_hour.index[is_hour==True]
    xticklabels = curve.loc[xticks,'timestamp'].apply(lambda x: x.strftime('%m-%d %H:%M'))
    
 

    fig, axes = plt.subplots(2,1, figsize=(10,10))
    #fig, ax1 = plt.subplots(figsize=(10,6))
    ax_prc = axes[0]
    ax_prc.plot(curve.index, curve['price_mkt'], 'gray', drawstyle='steps-post',linewidth=0.6)
    ax_prc.set_ylabel('Price')

    buy_trds = prev_df.loc[prev_df['side']== 'B'].copy()
    sell_trds = prev_df.loc[prev_df['side'] == 'S'].copy()
    
    buy_trds['timestamp'] = pd.to_datetime(buy_trds['timestamp'])
    sell_trds['timestamp'] = pd.to_datetime(sell_trds['timestamp'])

    
    #print(buy_trds)


    # Find the closest matching index in curve for each timestamp in buy_trds and sell_trds
    buy_indices = [curve['timestamp'].sub(ts).abs().idxmin() for ts in buy_trds['timestamp']]
    sell_indices = [curve['timestamp'].sub(ts).abs().idxmin() for ts in sell_trds['timestamp']]

    # Scatter the buy/sell signals using the mapped indices and using volume as size
    size_unit = 4 / 1e2
    ax_prc.scatter(buy_indices, buy_trds['price'], s=size_unit*buy_trds['volume']*buy_trds['multi'] * 10, marker='^', color='r')
    ax_prc.scatter(sell_indices, sell_trds['price'], s=size_unit*sell_trds['volume']*sell_trds['multi'] * 10, marker='v', color='g')
    
    
    #ax_prc.scatter(buy_trds.index, buy_trds['price'], s=size_unit*curve['pay'].abs(),marker='^',color='r')
    #ax_prc.scatter(sell_trds.index, sell_trds['price'], s=size_unit*curve['pay'].abs(),marker='v',color='g')

    ax_prc.set_xticks(xticks);
    ax_prc.set_xticklabels(xticklabels, rotation=30);
    ax_prc.grid(True)
    ax_prc.set_title(f"{stgy} {server} {tdate:%Y-%m-%d} : {instr}")
    
    #print(curve['sgn'])

    # pnl and pos curves
    ax_pos = axes[1]
    ax_pos.plot(curve.index, curve['val'], 'red', drawstyle='steps-post',linewidth=1.2, label='position')
    ax_pos.legend()
    ax_pos.set_ylabel('Position')

    ax_pnl = ax_pos.twinx()
    ax_pnl.plot(curve.index, curve['pnl'], 'blue', drawstyle='steps-post',linewidth=1.0, label='pnl')
    ax_pnl.legend()
    ax_pnl.set_ylabel('PnL')

    # stitch ou the session breaks.
    ax_pos.set_xticks(xticks);
    ax_pos.set_xticklabels(xticklabels, rotation=30);
    ax_pos.grid(True)

    plt.show()
    

#%%
#plot mark-out

def _plot_mark_out(tradingDay: pd.Timestamp, instr: str, trade_time: pd.Timestamp, window_before: int, window_after: int, interval):
    
    
    
    tradingDay = pd.Timestamp(tradingDay)
    mktprc = get_instr_price(tradingDay, instr, interval= f'{interval}s')
    #print(mktprc)
    mktprc.index = pd.to_datetime(mktprc.index)

    #p0 = mktprc.loc[trade_time, 'price']
    p0 = mktprc.iloc[mktprc.index.get_loc(trade_time, method='nearest')]['price']


    start_time = mktprc[mktprc.index <= trade_time].index[-window_before]

    end_time = mktprc[mktprc.index >= trade_time].index[window_after]

    trade_window = mktprc[(mktprc.index >= start_time) & (mktprc.index <= end_time)].copy()
    
    trade_window['normalized_price'] = trade_window['price'] / p0
    trade_window['returns'] = trade_window['normalized_price'].pct_change()
    


    
    plt.figure(figsize=(10, 4))  
    plt.plot(trade_window.index, trade_window['returns'])
    plt.axvline(x=trade_time, color='r', linestyle='--', label='trade time')
    plt.legend()
    plt.title(f'Mark Out Curve for {interval}s interval')
    plt.ylabel('Returns')
    plt.xlabel('Timestamp')
    plt.xticks(rotation=45)
    plt.grid(True)
    plt.tight_layout()

def plot_mark_out_for_intervals(tradingDay, instr, trade_time, window_before, window_after, intervals):
    for interval in intervals:
        print(f"Plotting mark-out curve for {tradingDay.date()} at {trade_time.time()} with interval: {interval}s")
        _plot_mark_out(tradingDay, instr, trade_time, window_before, window_after, interval)
        plt.show()




#%%

def main(
    stgy:str = 'melrose',
    tdate:str = "2023-09-15",
    underlying:str = 'i',
    server:str = None

    ):
    print(f"\nPlot trades and pnl for {stgy} on {tdate}")

    if tdate is None:
        tdate = pd.Timestamp.today().floor('d')
    else: 
        tdate = pd.Timestamp(tdate)
    #pdate = Calendar.loc[tdate, 'prevTradingDay']

    trades = get_trades_for_date(instr='i2401')

    trades = trades[(trades['strategy']==stgy) ]
    #print(trades)

    df = (trades[['recvTime','instrument', 'side','volume','price']]
            .rename(columns={'recvTime':'timestamp'})
            .set_index('timestamp')
            .sort_index()
           )

    df['underlying'] = df['instrument'].apply(lambda x: "".join([c.upper() for c in x if not c.isdigit()]))
    df['multi'] = df['underlying'].apply(lambda x: Underlyings['multiplier'][x])
    
    #print(df)
    instrs = df['instrument'].unique()
       
    
    
    print(f"\nAvailable instruments:\n{instrs}")
    if underlying is not None:
        instrs = [c for c in instrs if underlying in c]
     

    for instr in instrs:
      tmp = df[df['instrument']==instr]
      print(f"\n Sod pos and trades for {instr}:")
      #print(tmp)
     
      res = pnl_day_instr(tmp, tdate, instr)
      #print(res)
      
      _plot_pnl_tseries(df, res, tdate, instr, server, stgy)
      #_plot_mark_out(tdate, instr, pd.Timestamp('2023-09-14 21:54:11'), 3,3)
      plot_mark_out_for_intervals(
          tradingDay= tdate,
          instr= instr,
          trade_time=pd.Timestamp('2023-09-14 21:54:11'),
          window_before=3,
          window_after=3,
          intervals=[1, 5, 10, 60, 120]
          )

if __name__ == "__main__":
    fire.Fire(main)