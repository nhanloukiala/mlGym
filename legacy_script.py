__author__ = 'nhan trinh'

import numpy as np
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import itertools
from sklearn.decomposition import PCA as pca
from sklearn import manifold
from sklearn.linear_model import LinearRegression
from sklearn import preprocessing as prep
from sklearn.cross_validation import KFold
from scipy.stats import norm

def scatterplot_matrix(data, attNames, **kwargs):
    rows, atts = data.shape
    fig, axes = plt.subplots(nrows = atts, ncols =atts, figsize=(30,30))
    fig.subplots_adjust(hspace = 0.05 , wspace = 0.05)

    for ax in axes.flat:
        ax.xaxis.set_visible(False)
        ax.yaxis.set_visible(False)

        if ax.is_first_col():
            ax.yaxis.set_ticks_position('left')
        if ax.is_last_col():
            ax.yaxis.set_ticks_position('right')
        if ax.is_first_row():
            ax.xaxis.set_ticks_position('top')
        if ax.is_last_row():
            ax.xaxis.set_ticks_position('bottom')

    for i, j in zip(*np.triu_indices_from(axes, k=1)):
        for x, y in [(i,j), (j,i)]:
            axes[x,y].plot(data[y], data[x], **kwargs)

    # Label the diagonal subplots...
    for i, label in enumerate(attNames):
        axes[i,i].annotate(label, (0.5, 0.5), xycoords='axes fraction',
                ha='center', va='center')

    for i, j in zip(range(atts), itertools.cycle((-1, 0))):
        axes[j,i].xaxis.set_visible(True)
        axes[i,j].yaxis.set_visible(True)

    return fig

def get_color_list():
    import config
    return config.result

def plot_by_category(data, attNames, **kwargs):
    perm = itertools.permutations(attNames[:-1], 2)
    colors = get_color_list()

    #plotting pair by pair
    for x, y in perm:
        fig, ax = plt.subplots()
        projected = data[[x, y, attNames[11]]]
        grouped_projected = projected.groupby(attNames[11])

        for i, j in grouped_projected:
            j.plot(ax=ax, kind='scatter', x=x, y=y, label=i, color=colors[str(i)])

        plt.show()
        # pdb.set_trace()

def standardize_dataset(dataset, **kwargs):
    stdizer = prep.StandardScaler().fit(dataset)
    std_data = stdizer.transform(dataset)
    return std_data

def l2_norm(dataset, **kwargs):
    return prep.Normalizer(norm='l2', copy=True).fit_transform(dataset)

def scatter(x, y, color, **kwargs):
    plt.scatter(x, y, color = color, s=5)
    plt.title(kwargs['title'])

    legends = []
    col_dict = get_color_list()
    for x in col_dict:
        legends.append(mpatches.Patch(color=col_dict[x],label=x))
    plt.legend(handles=legends)
    plt.show()

def plot(trained, labels, attNames, **kwargs):
    r_color = get_color_list()
    colors = [r_color[str(x)] for x in labels]
    scatter(trained[:, 0], trained[: , 1], color = colors, **kwargs)
    # merge data with colors
    # print p.transform(p.components_)
    # labels = np.expand_dims(dataset[:, 11], axis=1)
    # final_data = np.concatenate((trained, labels), axis=1)

def pca_cal(dataset, labels, attNames, **kwargs):
    p = pca(n_components=2)
    trained = p.fit_transform(dataset)
    plot(trained, labels, attNames, **kwargs)
# def mds(dataset, labels, attNames, **kwargs):

def mds(dataset, labels, attNames, **kwargs):
    mds = manifold.MDS(n_components=2, max_iter=300)
    trained = mds.fit_transform(dataset)
    plot(trained, labels, attNames, **kwargs)

# Assignment 2

def linear_reg(dataset, labels, **kwargs):
    trainer = LinearRegression()
    model = trainer.fit(dataset, labels)

    return model, model.coef_, model.intercept_

def plot_error(list_data, mean, sd, message):
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    list_data = sorted(list_data)
    ax.plot(list_data, norm.pdf(list_data, mean, sd), 'r', label="Error")
    plt.xlabel("Error")
    plt.title(message)
    plt.text(0.33, 0.64, r'$\sigma=%s\ \mu=%s$' % (round(sd,2),mean))
    plt.text(0.7, 0.4, r'sample size=%s' % (len(list_data)))
    # ax.hist(list_data, int(len(list_data) ** 0.5),normed=1, facecolor='g')
    plt.show()

def generate_errors(external_errors, internal_errors, model, test, train):
    # out sample
    test_predict = model.predict(test[:, :-1])
    test_error = test[:, 11] - test_predict
    external_errors = np.append(external_errors, test_error)
    # in sample
    train_predict = model.predict(train[:, :-1])
    train_error = train[:, 11] - train_predict
    internal_errors = np.append(internal_errors, train_error)
    return external_errors, internal_errors

def displayResult(vec, message, plot_message) :
    mean_errors, sd_errors, var_errors = getStats(vec)
    print message
    print mean_errors
    print sd_errors
    print var_errors
    plot_error(vec, mean=mean_errors, sd=sd_errors, message=plot_message)

def getStats(vec):
    mean_errors = np.mean(vec)
    sd_errors = np.std(vec)
    var_errors = np.var(vec)

    return mean_errors, sd_errors, var_errors

def gen_cross_validation_indices(dataset):
    kf = KFold(dataset.shape[0], n_folds=5)
    return kf
