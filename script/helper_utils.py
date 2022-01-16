import matplotlib.pyplot as plt


def plot_error(train_rmse, test_rmse, num_epochs):
    plt.style.use('seaborn-whitegrid')

    x = list(range(num_epochs))
    fig = plt.figure()
    ax = plt.axes()

    plt.plot(x, train_rmse, label='train_rmse')
    plt.plot(x, test_rmse, label='test_rmse')

    leg = ax.legend()

    plt.savefig("test_image.png")
