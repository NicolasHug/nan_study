#!/usr/bin/python3

'''Run batch of experiments for a given Boolean function in given dimension,
and construct plots showing accuracy of the NaN classifier.'''

import sys
import os
import pickle
import matplotlib.pyplot as plt
import argparse

import accuracy


functions = {
        'monk2' : accuracy.monk2,
        'isEven' : accuracy.isEven,
        'kOfm' : lambda x: True, # redefined later because we need m
        'xor' : lambda x: x[-1] ^ x[-2],
        'or' : lambda x: x[-1] or x[-2],
        'and' : lambda x: x[-1] and x[-2],
        'andFourLast' : lambda x: x[-1] and x[-2] and x[-3] and x[-4],
        'firstAndLast' : lambda x: x[0] and x[-1],
        'monkAllButOne' : lambda x: sum(x) == m-1
        }

desc = ('Run batch of experiments for a given Boolean function in given ' +
        'dimension, and construct plots showing accuracy of the NaN ' +
        'classifier.')
parser = argparse.ArgumentParser(description=desc)
parser.add_argument('m', type=int, default=8, nargs='?',
                    help='dimension of universe X.  Default is 8',
                    choices=range(4, 12), metavar='<dimension>')
parser.add_argument('fname', type=str, default='xor', nargs='?',
                    help='Name of the Boolean function to use. Accepted ' +
                    'values are ' + ', '.join(functions.keys()) + '. Default '
                    + 'is xor.', choices=functions, metavar='<function name>')
parser.add_argument('-nExp', type=int, default=100, nargs='?',
                    help='number of experiences. Default is 100.',
                    metavar='<number of exp>')
parser.add_argument('--show', dest='show', action='store_const', const=True,
                    default=False, help='show plots on matplotlib window')
parser.add_argument('--savefig', dest='save_figure', action='store_const',
                    const=True, default=False, help='save figure. ' + 
                    'File will be in the plots folder.')
parser.add_argument('-format', type=str, default='pdf', nargs='?',
                    help='format for saving of images. Default is pdf.',
                    metavar='<format>')
args = parser.parse_args()


# update of k of m function now that m is defined
functions['kOfm'] = lambda x: sum(x) >= args.m / 2

# values of |S|
ns =  [3, 4, 5, 7, 10, 15, 20, 30, 40, 50, 75, 100]

d = {}
val_names = ['avgAccNanMV', 'avgAccNanThry1MV', 'avgErr1MV', 'avgAccNnS',
        'avgAccNnAEStarMV', 'avgLambdaMV', 'avgWMV', 'avgWMVEst',
        'avgGammaMV', 'avgAccNanThry2MV', 'avgErr2MV', 'avgAccNanThry3MV',
        'avgErr3MV', 'avgPropAMV', 'avgPropBMV']

for k in val_names:
    d[k] = []

d['m'] = args.m
d['fname'] = args.fname
f = functions[args.fname]
d['n_exp'] = args.nExp

# for each size of S, launch nExp experiments and retrieve results
for n in ns:
    try:
        res = accuracy.main(d['m'], n, d['n_exp'], f)
        for k in val_names:
            d[k].append(res[k])
    except accuracy.Sn_Too_Small:
        for k in val_names:
            d[k].append(0)

def ecai_plots():

    ax1 = fig.add_subplot(1, 2, 1)
    ax1.plot(ns, d['avgAccNanMV'], linestyle='--', marker='x', ms=2, mew=8,
             label='NaN$_S$')
    ax1.plot(ns, d['avgAccNnS'], linestyle='--', marker='o', label='NN$_S$')

    ax2 = fig.add_subplot(1, 2, 2)
    ax2.plot(ns, d['avgAccNanThry3MV'], linestyle='--', marker='x', ms=2,
             mew=8, label='thry')
    ax2.plot(ns, d['avgWMV'], linestyle='--', marker='^', label='$\omega$')
    #ax2.plot(ns, d['avgWMVEst'], linestyle='--', marker='d',
    #        label='$\hat{\omega}$')

    ax2.plot(ns, d['avgGammaMV'], linestyle='--', marker='s',
             label='$\gamma$')

    for ax in ax1, ax2:
        ax.set_title('')
        ax.yaxis.set_ticks([.6, .8, 1])
        ax.legend(loc='lower right')
        ax.xaxis.set_ticks_position('none')
        ax.set_ylim([.6, 1])
        ax.yaxis.set_ticks_position('none')

fig = plt.figure(figsize=(14, 3))
ecai_plots()

if args.save_figure:
    if not os.path.exists('./plots'):
        os.makedirs('./plots')
    plt.savefig('./plots/' + d['fname'] + 'm' + str(d['m']) + '.' + args.format,
            dpi=fig.dpi, bbox_inches='tight')
if args.show:
    plt.show()
