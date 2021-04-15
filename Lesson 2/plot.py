import numpy as np
import matplotlib.pyplot as plt

def image_preds(image, probs):
    ''' This function is for viewing an image and its predicted class.
    '''
    probs = probs.data.numpy().squeeze()

    fig, (ax1, ax2) = plt.subplots(figsize=(6,9), ncols=2)
    ax1.imshow(image.resize_(1, 28, 28).numpy().squeeze())
    ax1.axis('off')
    ax2.barh(np.arange(10), probs)
    ax2.set_aspect(0.1)
    ax2.set_yticks(np.arange(10))
    ax2.set_yticklabels(np.arange(10))
    ax2.set_title('Returned Class Probabilities')
    ax2.set_xlim(0, 1.1)

    plt.tight_layout()