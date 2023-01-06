import numpy as np
import pandas as pd
import yfinance as yf
import datetime
import matplotlib.pyplot as plt

# constants
RISKY_ASSET = 'AMZN' # can be changed for any chosen stock
START_DATE = '2020-01-01'
END_DATE = '2022-12-31'
FINAL_TRAIN_DATE = datetime.date(2022, 6, 30)
FIRST_TEST_DATE = FINAL_TRAIN_DATE + datetime.timedelta(days=1)

# fetching data from yahoo finance
df = yf.download(RISKY_ASSET, start=START_DATE, end=END_DATE)

adj_close = df['Adj Close']
returns = adj_close.pct_change().dropna()
print(f'Average daily return: {100 * returns.mean():.2f}%')
returns.plot(title=f'{RISKY_ASSET} returns: {START_DATE} - {END_DATE}')

# splitting the data
train = returns[START_DATE:FINAL_TRAIN_DATE.strftime('%Y-%m-%d')]
test = returns[FIRST_TEST_DATE.strftime('%Y-%m-%d'):END_DATE]

T = len(test)
N = len(test)
S_0 = adj_close[train.index[-1].date().strftime('%Y-%m-%d')]
N_SIM = 100
mu = train.mean()
sigma = train.std()

# defining the sim functions to break the continuous format to discrete form
def simulate_gbm(s_0, mu, sigma, n_sims, T, N):
    dt = T/N
    dW = np.random.normal(scale = np.sqrt(dt), size=(n_sims, N))
    W = np.cumsum(dW, axis=1)
    time_step = np.linspace(dt, T, N)
    time_steps = np.broadcast_to(time_step, (n_sims, N))
    S_t = s_0 * np.exp((mu - 0.5 * sigma ** 2) * time_steps + sigma * W)
    S_t = np.insert(S_t, 0, s_0, axis=1)
    return S_t

# training based on the params
gbm_simulations = simulate_gbm(S_0, mu, sigma, N_SIM, T, N)

# prepare objects for plotting
LAST_TRAIN_DATE = train.index[-1].date()
FIRST_TEST_DATE = test.index[0].date()
LAST_TEST_DATE = test.index[-1].date()
PLOT_TITLE = (f'{RISKY_ASSET} Simulation '
              f'({FIRST_TEST_DATE}:{LAST_TEST_DATE})')
selected_indices = adj_close[LAST_TRAIN_DATE:LAST_TEST_DATE].index
index = [date.date() for date in selected_indices]
gbm_simulations_df = pd.DataFrame(np.transpose(gbm_simulations), index=index)

# plotting
ax = gbm_simulations_df.plot(alpha=0.2, legend=False)
line_1, = ax.plot(index, gbm_simulations_df.mean(axis=1), color='red')
line_2, = ax.plot(index, adj_close[LAST_TRAIN_DATE:LAST_TEST_DATE], color='blue')
ax.set_title(PLOT_TITLE, fontsize=16)
ax.legend((line_1, line_2), ('mean', 'actual'))

plt.show()



