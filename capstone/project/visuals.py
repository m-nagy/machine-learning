###########################################
# Suppress matplotlib user warnings
# Necessary for newer version of matplotlib
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="matplotlib")
#
# Display inline matplotlib plots with IPython
from IPython import get_ipython
get_ipython().run_line_magic('matplotlib', 'inline')
###########################################

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import pandas as pd
import numpy as np


def pca_results(good_data, pca):
    '''
    Create a DataFrame of the PCA results
    Includes dimension feature weights and explained variance
    Visualizes the PCA results
    '''

    # Dimension indexing
    dimensions = dimensions = ['Dimension {}'.format(
        i) for i in range(1, len(pca.components_)+1)]

    # PCA components
    components = pd.DataFrame(
        np.round(pca.components_, 4), columns=list(good_data.keys()))
    components.index = dimensions

    # PCA explained variance
    ratios = pca.explained_variance_ratio_.reshape(len(pca.components_), 1)
    variance_ratios = pd.DataFrame(
        np.round(ratios, 4), columns=['Explained Variance'])
    variance_ratios.index = dimensions

    # Create a bar plot visualization
    fig, ax = plt.subplots(figsize=(14, 8))

    # Plot the feature weights as a function of the components
    components.plot(ax=ax, kind='bar')
    ax.set_ylabel("Feature Weights")
    ax.set_xticklabels(dimensions, rotation=0)

    # Display the explained variance ratios
    for i, ev in enumerate(pca.explained_variance_ratio_):
        ax.text(i-0.40, ax.get_ylim()[1] + 0.05,
                "Explained Variance\n          %.4f" % (ev))

    # Return a concatenated DataFrame
    return pd.concat([variance_ratios, components], axis=1)


def plot_data(df0):
    # Plot data
    fig, axes = plt.subplots(5, 3, figsize=(17, 20))
    plt.subplots_adjust(wspace=0.20, hspace=0.20, top=0.95)
    plt.suptitle("Heart Disease Data", fontsize=20)

    # Marginal feature distributions compared for disease and no-disease
    # (likelihoods)
    axes[0, 0].hist(df0[df0.num > 0].age, color=["crimson"],
                    histtype="step", label="disease")
    axes[0, 0].hist(df0[df0.num == 0].age, color=["chartreuse"],
                    histtype="step", label="no disease")
    axes[0, 0].set_xlabel("Age (years)")
    axes[0, 0].set_ylabel("Number of Patients")
    axes[0, 0].legend(prop={'size': 10}, loc="upper left")

    axes[0, 1].hist(df0[df0.num > 0].sex, color=["crimson"],
                    histtype="step", label="disease")
    axes[0, 1].hist(df0[df0.num == 0].sex, color=["chartreuse"],
                    histtype="step", label="no disease")
    axes[0, 1].set_xlabel("Sex (0=female,1=male)")
    axes[0, 1].set_ylabel("Number of Patients")
    axes[0, 1].legend(prop={'size': 10}, loc="upper left")
    axes[0, 2].hist(df0[df0.num > 0].cp, color=["crimson"],
                    histtype="step", label="disease")
    axes[0, 2].hist(df0[df0.num == 0].cp, color=["chartreuse"],
                    histtype="step", label="no disease")
    axes[0, 2].set_xlabel("Type of Chest Pain [cp]")
    axes[0, 2].set_ylabel("Number of Patients")
    axes[0, 2].legend(prop={'size': 10}, loc="upper left")

    axes[1, 0].hist(df0[df0.num > 0].restbp, color=["crimson"],
                    histtype="step", label="disease")
    axes[1, 0].hist(df0[df0.num == 0].restbp, color=[
                    "chartreuse"], histtype="step", label="no disease")
    axes[1, 0].set_xlabel("Resting Blood Pressure [restbp]")
    axes[1, 0].set_ylabel("Number of Patients")
    axes[1, 0].legend(prop={'size': 10}, loc="upper right")
    axes[1, 1].hist(df0[df0.num > 0].chol, color=["crimson"],
                    histtype="step", label="disease")
    axes[1, 1].hist(df0[df0.num == 0].chol, color=["chartreuse"],
                    histtype="step", label="no disease")
    axes[1, 1].set_xlabel("Serum Cholesterol [chol]")
    axes[1, 1].set_ylabel("Number of Patients")
    axes[1, 1].legend(prop={'size': 10}, loc="upper right")
    axes[1, 2].hist(df0[df0.num > 0].fbs, color=["crimson"],
                    histtype="step", label="disease")
    axes[1, 2].hist(df0[df0.num == 0].fbs, color=["chartreuse"],
                    histtype="step", label="no disease")
    axes[1, 2].set_xlabel("Fasting blood sugar [fbs]")
    axes[1, 2].set_ylabel("Number of Patients")
    axes[1, 2].legend(prop={'size': 10}, loc="upper right")
    axes[2, 0].hist(df0[df0.num > 0].restecg, color=[
                    "crimson"], histtype="step", label="disease")
    axes[2, 0].hist(df0[df0.num == 0].restecg, color=["chartreuse"],
                    histtype="step", label="no disease")
    axes[2, 0].set_xlabel("Rest ECG")
    axes[2, 0].set_ylabel("Number of Patients")
    axes[2, 0].legend(prop={'size': 10}, loc="upper center")
    axes[2, 1].hist(df0[df0.num > 0].thalach, color=[
                    "crimson"], histtype="step", label="disease")
    axes[2, 1].hist(df0[df0.num == 0].thalach, color=["chartreuse"],
                    histtype="step", label="no disease")
    axes[2, 1].set_xlabel("thalach")
    axes[2, 1].set_ylabel("Number of Patients")
    axes[2, 1].legend(prop={'size': 10}, loc="upper left")
    axes[2, 2].hist(df0[df0.num > 0].exang, color=["crimson"],
                    histtype="step", label="disease")
    axes[2, 2].hist(df0[df0.num == 0].exang, color=["chartreuse"],
                    histtype="step", label="no disease")
    axes[2, 2].set_xlabel("exang")
    axes[2, 2].set_ylabel("Number of Patients")
    axes[2, 2].legend(prop={'size': 10}, loc="upper right")
    axes[3, 0].hist(df0[df0.num > 0].oldpeak, color=[
                    "crimson"], histtype="step", label="disease")
    axes[3, 0].hist(df0[df0.num == 0].oldpeak, color=["chartreuse"],
                    histtype="step", label="no disease")
    axes[3, 0].set_xlabel("oldpeak")
    axes[3, 0].set_ylabel("Number of Patients")
    axes[3, 0].legend(prop={'size': 10}, loc="upper right")
    axes[3, 1].hist(df0[df0.num > 0].slope, color=["crimson"],
                    histtype="step", label="disease")
    axes[3, 1].hist(df0[df0.num == 0].slope, color=["chartreuse"],
                    histtype="step", label="no disease")
    axes[3, 1].set_xlabel("slope")
    axes[3, 1].set_ylabel("Number of Patients")
    axes[3, 1].legend(prop={'size': 10}, loc="upper right")
    axes[3, 2].hist(df0[df0.num > 0].ca, color=["crimson"],
                    histtype="step", label="disease")
    axes[3, 2].hist(df0[df0.num == 0].ca, color=["chartreuse"],
                    histtype="step", label="no disease")
    axes[3, 2].set_xlabel("ca")
    axes[3, 2].set_ylabel("Number of Patients")
    axes[3, 2].legend(prop={'size': 10}, loc="upper right")
    axes[4, 0].hist(df0[df0.num > 0].thal.tolist(), color=[
                    "crimson"], histtype="step", label="disease")
    axes[4, 0].hist(df0[df0.num == 0].thal, color=["chartreuse"],
                    histtype="step", label="no disease")
    axes[4, 0].set_xlabel("thal")
    axes[4, 0].set_ylabel("Number of Patients")
    axes[4, 0].legend(prop={'size': 10}, loc="upper right")
    axes[4, 1].axis("off")
    axes[4, 2].axis("off")
    plt.show()
