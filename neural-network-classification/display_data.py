#!/usr/bin/env python3

import sys
import argparse
import logging
import os.path

import pandas as pd
import numpy as np

import math
import matplotlib.pyplot as plt

def get_data(filename):
    """
    Assumes column 0 is the instance index stored in the
    csv file.  If no such column exists, remove the
    index_col=0 parameter.
    """
    data = pd.read_csv(filename, index_col=0)
    return data

def get_basename(filename):
    root, ext = os.path.splitext(filename)
    dirname, basename = os.path.split(root)
    logging.info("root: {}  ext: {}  dirname: {}  basename: {}".format(root, ext, dirname, basename))
    return basename

def get_feature_and_label_names(my_args, data):
    label_column = my_args.label
    feature_columns = my_args.features

    if label_column in data.columns:
        label = label_column
    else:
        label = ""

    features = []
    if feature_columns is not None:
        for feature_column in feature_columns:
            if feature_column in data.columns:
                features.append(feature_column)

    # no features specified, so add all non-labels
    if len(features) == 0:
        for feature_column in data.columns:
            if feature_column != label:
                features.append(feature_column)

    return features, label

def display_feature_histograms(my_args, data, figure_number):
    """
    Display a histogram for every feature and the label, if identified.
    """
    feature_columns, label_column = get_feature_and_label_names(my_args, data)

    total_count = len(feature_columns)
    if label_column:
        total_count += 1
    size = int(math.ceil(math.sqrt(total_count)))
    
    fig = plt.figure(figure_number, figsize=(14, 14))
    fig.suptitle( "Feature Histograms" )
    n_max = 1
    all_ax = []
    for i in range(1, len(feature_columns)+1):
        feature_column = feature_columns[i-1]
        if feature_column in data.columns:
            ax = fig.add_subplot(size, size, i)
            ax.set_yscale("log")
            n, bins, patches = ax.hist(data[feature_column], bins=20)
            if max(n) > n_max:
                n_max = max(n)
            ax.set_xlabel(feature_column)
            ax.locator_params(axis='x', tight=True, nbins=5)
            all_ax.append(ax)
            if feature_column == "trtbps":
                plt.xticks(rotation=90, fontsize=5)
                ax.set_xticks(ax.get_xticks()[::2])
        else:
            logging.warn("feature_column: '{}' not in data.columns: {}".format(feature_column, data.columns))


    if label_column:
        ax = fig.add_subplot(size, size, total_count)
        ax.set_yscale("log")
        n, bins, patches = ax.hist(data[label_column], bins=20)
        if max(n) > n_max:
            n_max = max(n)
        ax.set_xlabel(label_column)
        ax.locator_params(axis='x', tight=True, nbins=5)
        all_ax.append(ax)
        if feature_column == "trtbps":
            plt.xticks(rotation=90, fontsize=5)
            ax.set_xticks(ax.get_xticks()[::2])
        
    for ax in all_ax:
        ax.set_ylim(bottom=1.0, top=n_max)

    fig.tight_layout()
    basename = get_basename(my_args.data_file)
    figure_name = "{}-histogram-{}.{}".format(basename, "-".join(feature_columns), "pdf")
    fig.savefig(figure_name)
    plt.close(fig)
    return

def display_label_vs_features(my_args, data, figure_number):
    """
    Display a plot of label vs feature for every feature and the label, if identified.
    """
    feature_columns, label_column = get_feature_and_label_names(my_args, data)

    total_count = len(feature_columns)
    if label_column:
        total_count += 1
    size = int(math.ceil(math.sqrt(total_count)))

    all_ax = []

    fig = plt.figure(figure_number, figsize=(15, 15))

    
    fig.suptitle( "Label vs. Features" )
    for i in range(1, len(feature_columns)+1):
        feature_column = feature_columns[i-1]
        if feature_column in data.columns:
            ax = fig.add_subplot(size, size, i)
            ax.scatter(feature_column, label_column, data=data, s=1)
            ax.set_xlabel(feature_column)
            ax.set_ylabel(label_column)
            ax.locator_params(axis='both', tight=True, nbins=5)
            all_ax.append(ax)
            if feature_column == "trtbps":
                plt.xticks(rotation=90, fontsize=5)
                ax.set_xticks(ax.get_xticks()[::2])
        else:
            logging.warn("feature_column: '{}' not in data.columns: {}".format(feature_column, data.columns))
            
    if label_column:
        ax = fig.add_subplot(size, size, total_count)
        ax.scatter(label_column, label_column, data=data, s=1)
        ax.set_xlabel(label_column)
        ax.set_ylabel(label_column)
        ax.locator_params(axis='both', tight=True, nbins=5)
        all_ax.append(ax)
        if feature_column == "trtbps":
            plt.xticks(rotation=90, fontsize=5)
            ax.set_xticks(ax.get_xticks()[::2])

    fig.tight_layout()
    basename = get_basename(my_args.data_file)
    figure_name = "{}-scatter-{}.{}".format(basename, "-".join(feature_columns), "pdf")
    fig.savefig(figure_name)
    plt.close(fig)

    return

def parse_args(argv):
    parser = argparse.ArgumentParser(prog=argv[0], description='Create Data Plots')
    parser.add_argument('action', default='all',
                        choices=[ "label-vs-features", "feature-histograms",
                                  "all" ], 
                        nargs='?', help="desired action")
    parser.add_argument('--data-file',               '-d', default="",    type=str,   help="csv file of data to display")
    parser.add_argument('--features',  '-f', default=None, action="extend", nargs="+", type=str,
                        help="column names for features")
    parser.add_argument('--label',               '-l', default="label",    type=str,   help="column name for label")

    my_args = parser.parse_args(argv[1:])

    #
    # Do any special fixes/checks here
    #
    
    return my_args

def main(argv):
    my_args = parse_args(argv)
    logging.basicConfig(level=logging.WARN)

    filename = my_args.data_file
    if os.path.exists(filename) and os.path.isfile(filename):
        data = get_data(filename)

        if my_args.action in ("all", "label-vs-features"):
            display_label_vs_features(my_args, data, 1)
        if my_args.action in ("all", "feature-histograms"):
            display_feature_histograms(my_args, data, 2)

    else:
        print(filename + " doesn't exist, or is not a file.")
    
    return

if __name__ == "__main__":
    main(sys.argv)
    