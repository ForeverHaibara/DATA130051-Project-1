'''
Author: Zehao Zhang 20307130201
Homepage: https://github.com/ForeverHaibara/DATA130051-Project-1 
'''

from matplotlib import pyplot as plt
import numpy as np

def plotter(result, logy = True, trunc = 0, savepath = None):
    '''
    Plot the training result automatically.

    Parameters
    ----------
    logy: bool
        Whether or not to take the logarithm.

    trunc: int
        Only plot the truncated part of the training losses.

    savepath: str
        Save the figure to the path.
    '''
    print(f"Validation Accuracy = {result['acc']}")

    plt.figure(figsize=(15,7))
    plt.subplot(2,1,1, title = 'Loss')
    if not logy or min(np.min(np.array(result['loss_valid'])), np.min(np.array(result['loss']))) < 0:
        # take the logarithm
        # training loss
        plt.plot(result['loss'][trunc:], linewidth = 1)

        # validation loss
        plt.plot(np.arange(0,len(result['loss']),1250), 
                    result['loss_valid'], 'ro-', linewidth = 2)
    else:
        plt.semilogy(result['loss'][trunc:], linewidth=1)
        plt.semilogy(np.arange(0,len(result['loss']),1250), 
                    result['loss_valid'], 'ro-', linewidth = 2)
    plt.legend(['Training Loss','Validation Loss'])

    # validation accuracy
    plt.subplot(2,1,2, title = 'Acc')
    plt.plot(result['acc'], '.-')
    plt.plot(np.linspace(-2, len(result['acc']), 500), np.full(500,result['acc'][-1]),'--',c='gray')
    plt.xlim(-.5, len(result['acc'])-.5)
    plt.yticks(np.linspace(result['acc'][0],result['acc'][-1],10))
    if savepath: plt.savefig(savepath)
    plt.show()

