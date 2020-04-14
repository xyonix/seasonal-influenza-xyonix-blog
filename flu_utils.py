'''utils to generate N-BEAS model for seasonal influenza'''
import os
from datetime import date
import numpy as np
import torch
from torch import optim
from torch.nn import functional
from nbeats_pytorch.model import NBeatsNet
from sklearn.metrics import mean_squared_error, mean_absolute_error
import seaborn
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

#pylint: disable=too-many-arguments
#pylint: disable=too-many-locals
#pylint: disable=not-callable

# state mappings
US_STATE_ABBREVIATIONS = {
    'Alabama': 'AL',
    'Alaska': 'AK',
    'American Samoa': 'AS',
    'Arizona': 'AZ',
    'Arkansas': 'AR',
    'California': 'CA',
    'Colorado': 'CO',
    'Connecticut': 'CT',
    'Delaware': 'DE',
    'District of Columbia': 'DC',
    'Florida': 'FL',
    'Georgia': 'GA',
    'Guam': 'GU',
    'Hawaii': 'HI',
    'Idaho': 'ID',
    'Illinois': 'IL',
    'Indiana': 'IN',
    'Iowa': 'IA',
    'Kansas': 'KS',
    'Kentucky': 'KY',
    'Louisiana': 'LA',
    'Maine': 'ME',
    'Maryland': 'MD',
    'Massachusetts': 'MA',
    'Michigan': 'MI',
    'Minnesota': 'MN',
    'Mississippi': 'MS',
    'Missouri': 'MO',
    'Montana': 'MT',
    'Nebraska': 'NE',
    'Nevada': 'NV',
    'New Hampshire': 'NH',
    'New Jersey': 'NJ',
    'New Mexico': 'NM',
    'New York': 'NY',
    'North Carolina': 'NC',
    'North Dakota': 'ND',
    'Northern Mariana Islands':'MP',
    'Ohio': 'OH',
    'Oklahoma': 'OK',
    'Oregon': 'OR',
    'Pennsylvania': 'PA',
    'Puerto Rico': 'PR',
    'Rhode Island': 'RI',
    'South Carolina': 'SC',
    'South Dakota': 'SD',
    'Tennessee': 'TN',
    'Texas': 'TX',
    'Utah': 'UT',
    'Vermont': 'VT',
    'Virgin Islands': 'VI',
    'Virginia': 'VA',
    'Washington': 'WA',
    'West Virginia': 'WV',
    'Wisconsin': 'WI',
    'Wyoming': 'WY'
}


# model training, inference, and save/load

def data_generator(x_full, y_full, batch_size):
    '''data generator'''

    def split(arr, size):
        arrays = []
        while len(arr) > size:
            slice_ = arr[:size]
            arrays.append(slice_)
            arr = arr[size:]
        arrays.append(arr)
        return arrays

    while True:
        for batch_sample in split((x_full, y_full), batch_size):
            yield batch_sample

def train_100_grad_steps(data, device, net, optimiser, training_losses, test_losses,
                         report_interval, checkpoint_path):
    '''train model for 100 gradient steps'''
    gap = ' '*20
    global_step = load(net, optimiser, checkpoint_path)
    for x_train_batch, y_train_batch in data:
        global_step += 1
        optimiser.zero_grad()
        net.train()
        _, forecast = net(torch.tensor(x_train_batch, dtype=torch.float).to(device))
        loss = functional.mse_loss(forecast,
                                   torch.tensor(y_train_batch, dtype=torch.float).to(device))
        loss.backward()
        optimiser.step()
        if global_step % report_interval == 0:
            training_loss = loss.item()
            msg = f'gradient_step = {str(global_step).zfill(6)}{gap}'
            msg += f'training_loss = {loss.item():.6f}{gap}'
            msg += f'test_loss = {test_losses[-1]:.6f}'
            print(msg)
            training_losses.append(training_loss)
        if global_step > 0 and global_step % report_interval == 0:
            with torch.no_grad():
                save(net, optimiser, global_step, checkpoint_path)
            break
    return training_losses, test_losses

def load(model, optimiser, checkpoint_path):
    '''load pytorch model'''
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimiser.load_state_dict(checkpoint['optimizer_state_dict'])
        grad_step = checkpoint['grad_step']
        return grad_step
    return 0

def save(model, optimiser, grad_step, checkpoint_path):
    '''save pytorch model'''
    torch.save({
        'grad_step': grad_step,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimiser.state_dict(),
    }, checkpoint_path)

def eval_test(backcast_length, forecast_length, net, norm_constant,
              test_losses, x_test, y_test):
    '''evaluate model on test data and produce some plots'''
    net.eval()
    _, forecast = net(torch.tensor(x_test, dtype=torch.float))
    test_losses.append(functional.mse_loss(forecast,
                                           torch.tensor(y_test, dtype=torch.float)).item())
    pred = forecast.detach().numpy()
    subplots = [141, 142, 143, 144]
    plt.figure(1, figsize=(20, 2))
    for plot_id, i in enumerate(np.random.choice(range(len(x_test)), size=4, replace=False)):
        ff_norm = pred[i] * norm_constant
        xx_norm = x_test[i] * norm_constant
        yy_norm = y_test[i] * norm_constant
        plt.subplot(subplots[plot_id])
        plt.grid(which='major')
        plot_scatter(range(0, backcast_length), xx_norm, color='cornflowerblue')
        plot_scatter(range(backcast_length, backcast_length + forecast_length),
                     yy_norm, color='lime')
        plot_scatter(range(backcast_length, backcast_length + forecast_length),
                     ff_norm, color='red')
        plt.title('test sample %s' % (plot_id+1))
    plt.show()

def train_and_score_model(state, ili_data,
                          horizon=12,
                          lookback=120,
                          split=0.7,
                          batch_size=10,
                          model_dir='saved_models',
                          covid_19_onset='2019-11-01',
                          device=torch.device('cpu'),
                          report_interval=100,
                          num_training_intervals=20):
    '''traing a state flu model'''

    # check state data exists
    supported_states = ili_data.STATE.unique().tolist()
    assert state in supported_states, f'{state} not present in the ili_data'

    # establish checkpoint
    os.makedirs(model_dir, exist_ok=True)
    checkpoint_path = f'{model_dir}/nbeats-training-checkpoint.th'

    if os.path.isfile(checkpoint_path):
        os.remove(checkpoint_path)

    # map inputs to commonly used variables
    forecast_length = horizon # num weeks in forecast horizon
    backcast_length = lookback
    norm_constant = 1

    # get state data and eliminate COVID-19 period
    state_data_df = ili_data.loc['2000-01-01':covid_19_onset].reset_index()
    state_data_df.set_index(['STATE', 'DATE'], inplace=True)
    state_data = state_data_df.loc[state].values

    # create n-beats training data
    x_train_batch, ylist = [], []
    for i in range(backcast_length, len(state_data) - forecast_length):
        x_train_batch.append(state_data[i - backcast_length:i])
        ylist.append(state_data[i:i + forecast_length])

    x_train_batch = np.array(x_train_batch)[..., 0]
    ylist = np.array(ylist)[..., 0]

    cut = int(len(x_train_batch) * split)
    x_train, y_train = x_train_batch[:cut], ylist[:cut]
    x_test, y_test = x_train_batch[cut:], ylist[cut:]

    # create n-beats model
    net = NBeatsNet(stack_types=[NBeatsNet.GENERIC_BLOCK, NBeatsNet.GENERIC_BLOCK],
                    forecast_length=forecast_length,
                    thetas_dims=[7, 8],
                    nb_blocks_per_stack=3,
                    backcast_length=backcast_length,
                    hidden_layer_units=512, #128,
                    share_weights_in_stack=False,
                    device=device)

    # set optimizer
    optimiser = optim.Adam(net.parameters())

    # create data generator
    data = data_generator(x_train, y_train, batch_size)

    # train model
    test_losses = []
    training_losses = []

    for i in range(num_training_intervals):
        eval_test(backcast_length, forecast_length, net, norm_constant,
                  test_losses, x_test, y_test)

        training_losses, test_losses = train_100_grad_steps(data,
                                                            device,
                                                            net,
                                                            optimiser,
                                                            training_losses,
                                                            test_losses,
                                                            report_interval,
                                                            checkpoint_path)

    _, forecast = net(torch.tensor(x_test, dtype=torch.float))
    y_pred = forecast.detach().numpy()

    return y_test, y_pred, checkpoint_path

# file I/O
def state_model_path(state, lookback, horizon, model_base_dir='saved_models'):
    '''defines path to store state model'''
    return os.path.join(model_base_dir, f'horizon_{horizon}/lookback_{lookback}/{state}')

def load_inference_data(state, lookback, horizon=12, model_path=None):
    '''load inference data stored on disk as numpy (NPZ) files'''
    if model_path is None:
        model_path = state_model_path(state, lookback, horizon)

    assert os.path.exists(model_path)

    model_file = os.path.join(model_path, f'{state}_inference.npz')
    data = np.load(model_file)
    y_pred = data['y_pred']
    y_true = data['y_test']

    return y_true, y_pred

# inference metrics
def symmetric_mean_absolute_percentage_error(y_true, y_pred):
    '''symmetric mean absolute percentage error'''
    assert y_true.shape == y_pred.shape
    horizon = y_true.shape[1]
    numerator = np.abs(y_true - y_pred)
    denominator = (np.abs(y_true) + np.abs(y_pred))/2.0
    return np.mean(1./horizon * np.sum(numerator/denominator, axis=1))

def efficacy_metrics(y_true, y_pred):
    '''efficacy metrics'''
    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    smape = symmetric_mean_absolute_percentage_error(y_true, y_pred)
    return mae, mse, smape

# plotting
def plot_scatter(*args, **kwargs):
    '''plot training progress'''
    plt.plot(*args, **kwargs)
    plt.scatter(*args, **kwargs, s=10)

def ili_plot(series, **kwargs):
    '''plot ILI series'''
    plt.plot(series, linewidth=2, color='cornflowerblue', **kwargs)
    axis = plt.gca()
    axis.grid(True)
    date_format = mdates.DateFormatter('%Y')
    axis.xaxis.set_major_formatter(date_format)
    plt.tick_params(labelsize=18)
    axis.axvspan(date(2019, 12, 1), date.today(), alpha=0.2, color='red')

def plot_error_heatmap(err_hist,
                       ylabels=None,
                       xlabels=range(1, 25),
                       y_axis_label='forecast error',
                       x_axis_label='weeks out',
                       title=None,
                       cmap='Blues',
                       figsize=(20, 5),
                       fontsize=16,
                       fig_path=None):
    """Plot error heatmap"""
    if ylabels is None:
        ylabels = list(np.flip(np.arange(-9, 10, 1), axis=0))

    fig = plt.figure(figsize=figsize)
    if not title is None:
        axis = plt.axes()
        axis.set_title(title)

    seaborn.heatmap(err_hist, annot=True, cmap=cmap, yticklabels=ylabels, xticklabels=xlabels)
    plt.yticks(rotation=0)
    plt.ylabel(y_axis_label, fontsize=fontsize)
    plt.xlabel(x_axis_label, fontsize=fontsize)
    if fig_path is not None:
        plt.savefig(fig_path, bbox_inches='tight')
        plt.close(fig)

def compute_error_histogram(y_true, y_pred, bins=np.arange(-9.5, 10.5, 1.0)):
    """
    Compute the error histogram for a single prediction set
    :param y_true: y_true
    :param y_pred: y_pred
    :param bins: bins
    :return: error histogram
    """
    diff = y_true - y_pred
    err_hist = np.zeros((len(bins) - 1, diff.shape[1]), dtype=np.float)
    for i in range(diff.shape[1]):
        hist, _ = np.histogram(diff[:, i], bins=bins)
        err_hist[:, i] = np.flip(np.round(hist.astype(np.float) / np.sum(hist), 2), axis=0)
    return err_hist

def plot_error_histogram(y_test, y_pred, title=None, label_start=-2, label_stop=2, label_step=0.5,
                         fig_path=None, y_axis_label='forecast error', x_axis_label='weeks out',
                         cmap='Blues'):
    '''plot error histogram using labels to define y-axis'''
    centers = np.arange(label_start, label_stop+label_step, label_step)
    halfspan = label_step/2.0

    bins = np.arange(centers[0]-halfspan, centers[-1]+halfspan*2, halfspan*2)
    err_hist = compute_error_histogram(y_test, y_pred, bins=bins)
    forecast_length = y_test.shape[1]

    plot_error_heatmap(err_hist,
                       xlabels=[*range(1, forecast_length+1)],
                       ylabels=list(reversed(centers)),
                       x_axis_label=x_axis_label,
                       y_axis_label=y_axis_label,
                       title=title,
                       cmap=cmap,
                       fig_path=fig_path)
